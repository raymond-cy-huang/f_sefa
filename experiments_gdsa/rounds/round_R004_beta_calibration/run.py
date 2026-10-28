"""Calibrate sinh nonlinearity across bases using matched beta on shared-W samples."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch


ROUND_DIR = Path(__file__).resolve().parent
ROOT = ROUND_DIR.parents[2]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


legacy = load_module("gdsa_exp02_for_r004", ROOT / "experiments_gdsa/02_generator_jacobian_ro/run.py")
bases = load_module("gdsa_r003_bases", ROOT / "experiments_gdsa/rounds/round_R003_basis_validation/run.py")


@dataclass(frozen=True)
class Config:
    num_samples: int = 100
    seed: int = 42
    raw_alphas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0, 2.0)
    betas: tuple[float, ...] = (0.1, 0.3, 0.5, 0.7, 1.0)
    output_dir: str = str(ROUND_DIR / "outputs_a")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--raw-alphas", type=float, nargs="+", default=list(defaults.raw_alphas))
    parser.add_argument("--betas", type=float, nargs="+", default=list(defaults.betas))
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    return Config(
        num_samples=values.num_samples,
        seed=values.seed,
        raw_alphas=tuple(values.raw_alphas),
        betas=tuple(values.betas),
        output_dir=values.output_dir,
    )


def configure(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def nonlinear_metrics(coordinates: np.ndarray, alpha: float) -> dict[str, float | int]:
    argument = alpha * coordinates
    derivative = np.cosh(argument)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        mapped = np.sinh(argument) / alpha
        ratio = np.divide(mapped, coordinates, out=np.ones_like(mapped), where=coordinates != 0)
    metric_scales = derivative**2
    per_sample_condition = np.max(metric_scales, axis=1) / np.min(metric_scales, axis=1)
    return {
        "argument_abs_q99": float(np.quantile(np.abs(argument), 0.99)),
        "argument_abs_max": float(np.max(np.abs(argument))),
        "mapping_ratio_abs_median": float(np.median(np.abs(ratio))),
        "mapping_ratio_abs_q99": float(np.quantile(np.abs(ratio), 0.99)),
        "derivative_median": float(np.median(derivative)),
        "derivative_q99": float(np.quantile(derivative, 0.99)),
        "sample_metric_condition_median": float(np.median(per_sample_condition)),
        "sample_metric_condition_max": float(np.max(per_sample_condition)),
        "nonfinite_count": int(
            np.size(mapped)
            - np.count_nonzero(np.isfinite(mapped) & np.isfinite(derivative) & np.isfinite(metric_scales))
        ),
    }


def main() -> None:
    config = parse_args()
    configure(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    legacy_config = legacy.Config(num_seeds=config.num_samples, seed=config.seed)
    generator = legacy.load_generator(legacy_config.model)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    samples = legacy.sample_w(generator, config.num_samples, legacy_config).numpy().astype(np.float64)
    np.save(output_dir / "shared_w_samples.npy", samples.astype(np.float32))
    sample_sha = hashlib.sha256((output_dir / "shared_w_samples.npy").read_bytes()).hexdigest()

    builders = {
        "identity": bases.identity_transform,
        "legendre": bases.legendre_transform,
        "chebyshev_first_kind": bases.chebyshev_transform,
    }
    distribution_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []
    matched_rows: list[dict[str, object]] = []
    q99_by_basis: dict[str, float] = {}

    for name, builder in builders.items():
        transform, _ = builder(samples.shape[1])
        coordinates = samples @ transform.T
        absolute = np.abs(coordinates)
        q99 = float(np.quantile(absolute, 0.99))
        q99_by_basis[name] = q99
        distribution_rows.append(
            {
                "basis": name,
                "coordinate_mean": float(np.mean(coordinates)),
                "coordinate_std": float(np.std(coordinates)),
                "coordinate_abs_median": float(np.median(absolute)),
                "coordinate_abs_q90": float(np.quantile(absolute, 0.90)),
                "coordinate_abs_q95": float(np.quantile(absolute, 0.95)),
                "coordinate_abs_q99": q99,
                "coordinate_abs_max": float(np.max(absolute)),
            }
        )
        for alpha in config.raw_alphas:
            raw_rows.append(
                {"basis": name, "alpha": alpha, "beta_empirical": alpha * q99, **nonlinear_metrics(coordinates, alpha)}
            )
        for beta in config.betas:
            alpha = beta / q99
            metrics = nonlinear_metrics(coordinates, alpha)
            matched_rows.append(
                {
                    "basis": name,
                    "beta_target": beta,
                    "alpha_calibrated": alpha,
                    "beta_abs_error": abs(float(metrics["argument_abs_q99"]) - beta),
                    **metrics,
                }
            )

    write_csv(output_dir / "coordinate_distributions.csv", distribution_rows)
    write_csv(output_dir / "raw_alpha_metrics.csv", raw_rows)
    write_csv(output_dir / "matched_beta_metrics.csv", matched_rows)
    q99_values = list(q99_by_basis.values())
    relative_spread = (max(q99_values) - min(q99_values)) / min(q99_values)
    max_beta_error = max(float(row["beta_abs_error"]) for row in matched_rows)
    nonfinite = sum(int(row["nonfinite_count"]) for row in raw_rows + matched_rows)
    summary = {
        "status": "PASS" if max_beta_error <= 1e-12 and nonfinite == 0 else "FAIL",
        "hypothesis_outcome": "SUPPORTED" if relative_spread >= 0.05 else "UNSUPPORTED",
        "config": asdict(config),
        "shared_w_samples_sha256": sample_sha,
        "q99_abs_coordinate_by_basis": q99_by_basis,
        "q99_relative_spread": relative_spread,
        "maximum_matched_beta_abs_error": max_beta_error,
        "total_nonfinite_count": nonfinite,
        "interpretation": "Track A calibration only; matched beta is not semantic evidence.",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
