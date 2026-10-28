"""Validate discrete Identity, Legendre, and Chebyshev transforms."""

from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.polynomial import chebyshev, legendre
from scipy.special import roots_chebyt


ROUND_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Config:
    dimension: int = 512
    num_samples: int = 100
    seed: int = 42
    output_dir: str = str(ROUND_DIR / "outputs")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=defaults.dimension)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    if values.dimension <= 0 or values.num_samples <= 0:
        parser.error("dimension and num-samples must be positive")
    return Config(**vars(values))


def identity_transform(dimension: int) -> tuple[np.ndarray, dict[str, object]]:
    return np.eye(dimension, dtype=np.float64), {
        "domain": "finite coordinate index",
        "measure": "Euclidean counting measure",
        "nodes": "canonical coordinates",
        "normalization": "unit coordinate vectors",
    }


def legendre_transform(dimension: int) -> tuple[np.ndarray, dict[str, object]]:
    nodes, weights = legendre.leggauss(dimension)
    orders = np.arange(dimension, dtype=np.float64)
    norms = 2.0 / (2.0 * orders + 1.0)
    basis = legendre.legvander(nodes, dimension - 1)
    transform = np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]
    return transform, {
        "domain": "[-1, 1]",
        "measure": "dx",
        "nodes": "roots of P_n (Gauss-Legendre)",
        "normalization": "||P_k||^2 = 2/(2k+1)",
        "quadrature_weight_sum": float(weights.sum()),
    }


def chebyshev_transform(dimension: int) -> tuple[np.ndarray, dict[str, object]]:
    nodes, weights = chebyshev.chebgauss(dimension)
    norms = np.full(dimension, np.pi / 2.0, dtype=np.float64)
    norms[0] = np.pi
    basis = chebyshev.chebvander(nodes, dimension - 1)
    transform = np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]

    # Independent checks against the DLMF closed form and SciPy implementation.
    indices = np.arange(1, dimension + 1, dtype=np.float64)
    formula_nodes = np.cos((2.0 * indices - 1.0) * np.pi / (2.0 * dimension))
    formula_weights = np.full(dimension, np.pi / dimension)
    scipy_nodes, scipy_weights = roots_chebyt(dimension)
    metadata = {
        "domain": "[-1, 1]",
        "measure": "dx/sqrt(1-x^2)",
        "nodes": "roots of T_n (Gauss-Chebyshev first kind)",
        "normalization": "||T_0||^2=pi; ||T_k||^2=pi/2 for k>0",
        "quadrature_weight_sum": float(weights.sum()),
        "formula_node_max_abs_error": float(
            np.max(np.abs(np.sort(nodes) - np.sort(formula_nodes)))
        ),
        "formula_weight_max_abs_error": float(np.max(np.abs(weights - formula_weights))),
        "scipy_node_max_abs_error": float(
            np.max(np.abs(np.sort(nodes) - np.sort(scipy_nodes)))
        ),
        "scipy_weight_max_abs_error": float(np.max(np.abs(weights - scipy_weights))),
    }
    return transform, metadata


def evaluate(
    name: str,
    transform64: np.ndarray,
    metadata: dict[str, object],
    samples64: np.ndarray,
) -> list[dict[str, object]]:
    identity = np.eye(transform64.shape[0])
    orthogonality_error = float(np.max(np.abs(transform64.T @ transform64 - identity)))
    condition_number = float(np.linalg.cond(transform64))
    rows: list[dict[str, object]] = []
    for dtype in (np.float64, np.float32):
        transform = transform64.astype(dtype)
        samples = samples64.astype(dtype)
        started = time.perf_counter()
        coordinates = samples @ transform.T
        reconstructed = coordinates @ transform
        elapsed = time.perf_counter() - started
        residual = reconstructed.astype(np.float64) - samples.astype(np.float64)
        sample_norms = np.linalg.norm(samples.astype(np.float64), axis=1)
        relative = np.linalg.norm(residual, axis=1) / sample_norms
        # An orthogonal transform must preserve pairwise angle.
        before = samples64[0] @ samples64[1]
        after = coordinates[0].astype(np.float64) @ coordinates[1].astype(np.float64)
        denominator = np.linalg.norm(samples64[0]) * np.linalg.norm(samples64[1])
        rows.append(
            {
                "basis": name,
                "dtype": np.dtype(dtype).name,
                "orthogonality_max_abs_error_float64": orthogonality_error,
                "condition_number_float64": condition_number,
                "roundtrip_relative_error_max": float(np.max(relative)),
                "roundtrip_relative_error_mean": float(np.mean(relative)),
                "roundtrip_max_abs_error": float(np.max(np.abs(residual))),
                "pair_cosine_numerator_error": float(abs(after - before) / denominator),
                "runtime_seconds": elapsed,
                "transform_memory_bytes": int(transform.nbytes),
                "domain": metadata["domain"],
                "measure": metadata["measure"],
                "nodes": metadata["nodes"],
                "normalization": metadata["normalization"],
            }
        )
    return rows


def main() -> None:
    config = parse_args()
    rng = np.random.default_rng(config.seed)
    samples = rng.standard_normal((config.num_samples, config.dimension))
    builders = {
        "identity": identity_transform,
        "legendre": legendre_transform,
        "chebyshev_first_kind": chebyshev_transform,
    }
    rows: list[dict[str, object]] = []
    metadata: dict[str, dict[str, object]] = {}
    for name, builder in builders.items():
        transform, details = builder(config.dimension)
        metadata[name] = details
        rows.extend(evaluate(name, transform, details, samples))

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "basis_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    failures = []
    for row in rows:
        if float(row["orthogonality_max_abs_error_float64"]) > 1e-10:
            failures.append(f"{row['basis']}: orthogonality")
        tolerance = 1e-10 if row["dtype"] == "float64" else 1e-5
        if float(row["roundtrip_relative_error_max"]) > tolerance:
            failures.append(f"{row['basis']}/{row['dtype']}: roundtrip")
    cheb = metadata["chebyshev_first_kind"]
    for key in (
        "formula_node_max_abs_error",
        "formula_weight_max_abs_error",
        "scipy_node_max_abs_error",
        "scipy_weight_max_abs_error",
    ):
        if float(cheb[key]) > 1e-12:
            failures.append(f"chebyshev_first_kind: {key}")

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "config": asdict(config),
        "metadata": metadata,
        "failures": failures,
        "interpretation": (
            "Numerical representation validation only. Orthogonal linear transforms "
            "preserve Euclidean geometry and do not imply semantic differences."
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
