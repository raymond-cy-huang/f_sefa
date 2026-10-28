"""Run deterministic GDSA proxy measurements and path-energy refinement."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch

from energy import build_legendre_transform, build_path


ROUND_DIR = Path(__file__).resolve().parent
ROOT = ROUND_DIR.parents[2]
LEGACY_PATH = ROOT / "experiments_gdsa/02_generator_jacobian_ro/run.py"
spec = importlib.util.spec_from_file_location("gdsa_exp02_legacy", LEGACY_PATH)
legacy = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = legacy
spec.loader.exec_module(legacy)


@dataclass(frozen=True)
class Config:
    num_seeds: int = 2
    seed: int = 42
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0)
    step_counts: tuple[int, ...] = (1, 2, 5, 10, 20, 50)
    target_tau: float = 0.5
    proxy_steps: int = 5
    jacobian_epsilon: float = 0.01
    response_resolution: int = 64
    output_dir: str = str(ROUND_DIR / "outputs_run_a")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-seeds", type=int, default=defaults.num_seeds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(defaults.alphas))
    parser.add_argument("--step-counts", type=int, nargs="+", default=list(defaults.step_counts))
    parser.add_argument("--target-tau", type=float, default=defaults.target_tau)
    parser.add_argument("--proxy-steps", type=int, default=defaults.proxy_steps)
    parser.add_argument("--jacobian-epsilon", type=float, default=defaults.jacobian_epsilon)
    parser.add_argument("--response-resolution", type=int, default=defaults.response_resolution)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    if values.num_seeds <= 0 or any(step <= 0 for step in values.step_counts):
        parser.error("num-seeds and step-counts must be positive")
    return Config(
        num_seeds=values.num_seeds,
        seed=values.seed,
        alphas=tuple(values.alphas),
        step_counts=tuple(values.step_counts),
        target_tau=values.target_tau,
        proxy_steps=values.proxy_steps,
        jacobian_epsilon=values.jacobian_epsilon,
        response_resolution=values.response_resolution,
        output_dir=values.output_dir,
    )


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def configure_determinism(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def main() -> None:
    config = parse_args()
    configure_determinism(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    legacy_config = legacy.Config(
        num_seeds=config.num_seeds,
        seed=config.seed,
        alphas=config.alphas,
        target_tau=config.target_tau,
        num_steps=config.proxy_steps,
        jacobian_epsilon=config.jacobian_epsilon,
        response_resolution=config.response_resolution,
        output_dir=str(output_dir),
    )
    generator = legacy.load_generator(legacy_config.model)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    target, semantic = legacy.extract_target_direction(generator, legacy_config.semantic)
    transform = build_legendre_transform(len(target))
    points = legacy.sample_w(generator, config.num_seeds, legacy_config).numpy().astype(np.float64)
    methods = ("B1_euclidean", "B2_fixed_metric", "B3_dynamic_metric")

    refinement_rows: list[dict[str, object]] = []
    proxy_rows: list[dict[str, object]] = []
    for seed_index, point in enumerate(points):
        for alpha in config.alphas:
            for steps in config.step_counts:
                for method in methods:
                    result = build_path(
                        point, target, transform, alpha, config.target_tau, steps, method
                    )
                    refinement_rows.append(
                        {
                            "seed_index": seed_index,
                            "alpha": alpha,
                            "num_steps": steps,
                            "method": method,
                            "target_progress": result.target_progress,
                            "target_progress_error": abs(result.target_progress - config.target_tau),
                            "legacy_left_energy": result.legacy_left_energy,
                            "standard_midpoint_energy": result.midpoint_energy,
                            "latent_displacement_l2": float(np.linalg.norm(result.points[-1] - point)),
                        }
                    )

            base_rms, base_response = legacy.directional_jacobian(
                generator, point, target, config.jacobian_epsilon, config.response_resolution
            )
            for method in methods:
                result = build_path(
                    point,
                    target,
                    transform,
                    alpha,
                    config.target_tau,
                    config.proxy_steps,
                    method,
                )
                start_rms, start_response = legacy.directional_jacobian(
                    generator,
                    result.points[0],
                    result.directions[0],
                    config.jacobian_epsilon,
                    config.response_resolution,
                )
                end_rms, end_response = legacy.directional_jacobian(
                    generator,
                    result.points[-1],
                    result.directions[-1],
                    config.jacobian_epsilon,
                    config.response_resolution,
                )
                proxy_rows.append(
                    {
                        "seed_index": seed_index,
                        "alpha": alpha,
                        "method": method,
                        "target_generator_jacobian_rms": base_rms,
                        "start_generator_jacobian_rms": start_rms,
                        "end_generator_jacobian_rms": end_rms,
                        "start_response_cosine_to_target": legacy.response_cosine(start_response, base_response),
                        "end_response_cosine_to_start": legacy.response_cosine(end_response, start_response),
                    }
                )

    repeat_point = torch.as_tensor(points[:1], dtype=torch.float32)
    repeat_first = legacy.decode_low(generator, repeat_point, config.response_resolution)
    repeat_second = legacy.decode_low(generator, repeat_point, config.response_resolution)
    repeat_max = float((repeat_first - repeat_second).abs().max().cpu())

    write_csv(output_dir / "refinement_metrics.csv", refinement_rows)
    write_csv(output_dir / "proxy_metrics.csv", proxy_rows)
    maximum_progress_error = max(float(row["target_progress_error"]) for row in refinement_rows)
    alpha_zero_errors = [
        abs(float(row["standard_midpoint_energy"]) - 0.25)
        for row in refinement_rows
        if float(row["alpha"]) == 0.0
    ]
    summary = {
        "status": "PASS"
        if repeat_max == 0.0
        and maximum_progress_error <= 1e-10
        and max(alpha_zero_errors) <= 1e-12
        else "FAIL",
        "config": asdict(config),
        "semantic": semantic,
        "deterministic_algorithms": True,
        "repeat_decode_max_abs_difference": repeat_max,
        "maximum_target_progress_error": maximum_progress_error,
        "maximum_alpha_zero_analytic_energy_error": max(alpha_zero_errors),
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": "Measurement-standardization evidence only; semantic results remain proxy-only.",
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
