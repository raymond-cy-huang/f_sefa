"""Experiment 01: validate the Legendre + sinh coordinate map and inverse."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial import legendre
from scipy.differentiate import jacobian


EXPERIMENT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    dimension: int = 512
    num_samples: int = 100
    alphas: tuple[float, ...] = (0.0, 0.1, 0.5, 1.0, 2.0)
    seed: int = 42
    jacobian_dimension: int = 8
    output_dir: str = str(EXPERIMENT_DIR / "outputs")


def parse_args() -> ExperimentConfig:
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dimension", type=int, default=defaults.dimension)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=list(defaults.alphas)
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--jacobian-dimension", type=int, default=defaults.jacobian_dimension
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    if values.dimension <= 0 or values.num_samples <= 0:
        parser.error("dimension and num-samples must be positive")
    if values.jacobian_dimension <= 0:
        parser.error("jacobian-dimension must be positive")
    if any(alpha < 0 for alpha in values.alphas):
        parser.error("alphas must be non-negative")
    return ExperimentConfig(
        dimension=values.dimension,
        num_samples=values.num_samples,
        alphas=tuple(values.alphas),
        seed=values.seed,
        jacobian_dimension=values.jacobian_dimension,
        output_dir=values.output_dir,
    )


def build_legendre_transform(dimension: int, dtype=np.float64) -> np.ndarray:
    """Return the weighted, normalized Gauss–Legendre transform L."""
    nodes, weights = legendre.leggauss(dimension)
    orders = np.arange(dimension, dtype=np.float64)
    norms = 2.0 / (2.0 * orders + 1.0)
    basis = legendre.legvander(nodes, dimension - 1)
    transform = np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]
    return transform.astype(dtype, copy=False)


def linear_coordinates(transform: np.ndarray, latent: np.ndarray) -> np.ndarray:
    """Apply L along the first axis while preserving SciPy batch axes."""
    return np.einsum("ij,j...->i...", transform, latent)


def inverse_linear_coordinates(
    transform: np.ndarray, coordinates: np.ndarray
) -> np.ndarray:
    return np.einsum("ji,j...->i...", transform, coordinates)


def sinh_forward(
    latent: np.ndarray, transform: np.ndarray, alpha: float
) -> np.ndarray:
    coordinates = linear_coordinates(transform, latent)
    if alpha == 0.0:
        return coordinates
    return np.sinh(alpha * coordinates) / alpha


def sinh_inverse(
    mapped: np.ndarray, transform: np.ndarray, alpha: float
) -> np.ndarray:
    if alpha == 0.0:
        coordinates = mapped
    else:
        coordinates = np.arcsinh(alpha * mapped) / alpha
    return inverse_linear_coordinates(transform, coordinates)


def forward_jacobian(
    latent: np.ndarray, transform: np.ndarray, alpha: float
) -> np.ndarray:
    coordinates = linear_coordinates(transform, latent)
    if alpha == 0.0:
        scales = np.ones_like(coordinates)
    else:
        scales = np.cosh(alpha * coordinates)
    return scales[:, None] * transform


def inverse_jvp(
    mapped: np.ndarray,
    mapped_tangent: np.ndarray,
    transform: np.ndarray,
    alpha: float,
) -> np.ndarray:
    if alpha == 0.0:
        coordinate_tangent = mapped_tangent
    else:
        scales = np.sqrt(1.0 + (alpha * mapped) ** 2)
        coordinate_tangent = mapped_tangent / scales
    return inverse_linear_coordinates(transform, coordinate_tangent)


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    numerator = np.linalg.norm(candidate - reference, axis=0)
    denominator = np.maximum(np.linalg.norm(reference, axis=0), 1e-15)
    return numerator / denominator


def cosine(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    numerator = np.sum(reference * candidate, axis=0)
    denominator = np.linalg.norm(reference, axis=0) * np.linalg.norm(
        candidate, axis=0
    )
    return numerator / np.maximum(denominator, 1e-15)


def mapping_metrics(
    latent_samples: np.ndarray,
    transform: np.ndarray,
    alpha: float,
    dtype_name: str,
    rng: np.random.Generator,
) -> dict[str, float | int | str | bool]:
    mapped = sinh_forward(latent_samples, transform, alpha)
    reconstructed = sinh_inverse(mapped, transform, alpha)
    relative_errors = relative_l2(latent_samples, reconstructed)
    cosines = cosine(latent_samples, reconstructed)

    latent = latent_samples[:, 0]
    tangent = rng.normal(size=latent.shape).astype(latent.dtype)
    tangent /= np.linalg.norm(tangent)
    forward_matrix = forward_jacobian(latent, transform, alpha)
    mapped_tangent = forward_matrix @ tangent
    tangent_reconstructed = inverse_jvp(
        sinh_forward(latent, transform, alpha),
        mapped_tangent,
        transform,
        alpha,
    )
    tangent_relative_error = float(
        np.linalg.norm(tangent_reconstructed - tangent) / np.linalg.norm(tangent)
    )

    metric = forward_matrix.T @ forward_matrix
    metric_symmetry_error = float(np.max(np.abs(metric - metric.T)))
    metric_eigenvalues = np.linalg.eigvalsh(metric.astype(np.float64))

    return {
        "dtype": dtype_name,
        "alpha": alpha,
        "all_finite": bool(np.all(np.isfinite(mapped))),
        "max_abs_mapped": float(np.max(np.abs(mapped))),
        "mean_relative_l2": float(np.mean(relative_errors)),
        "p95_relative_l2": float(np.quantile(relative_errors, 0.95)),
        "max_relative_l2": float(np.max(relative_errors)),
        "min_cosine": float(np.min(cosines)),
        "jvp_inverse_relative_error": tangent_relative_error,
        "metric_symmetry_error": metric_symmetry_error,
        "metric_min_eigenvalue": float(metric_eigenvalues[0]),
        "metric_max_eigenvalue": float(metric_eigenvalues[-1]),
        "metric_condition_number": float(
            metric_eigenvalues[-1] / metric_eigenvalues[0]
        ),
    }


def validate_numerical_jacobian(dimension: int, alpha: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    transform = build_legendre_transform(dimension, np.float64)
    point = rng.normal(size=dimension)

    result = jacobian(
        lambda value: sinh_forward(value, transform, alpha),
        point,
        tolerances={"atol": 1e-10, "rtol": 1e-8},
        maxiter=15,
    )
    analytical = forward_jacobian(point, transform, alpha)
    difference = result.df - analytical
    return {
        "dimension": dimension,
        "alpha": alpha,
        "all_converged": bool(np.all(result.success)),
        "max_abs_error": float(np.max(np.abs(difference))),
        "relative_frobenius_error": float(
            np.linalg.norm(difference) / np.linalg.norm(analytical)
        ),
    }


def acceptance(rows: list[dict], orthogonality_error: float, jacobian_check: dict):
    failures = []
    if orthogonality_error > 1e-10:
        failures.append(
            f"float64 orthogonality error {orthogonality_error:.3e} > 1e-10"
        )
    for row in rows:
        if not row["all_finite"]:
            failures.append(
                f"non-finite mapping for dtype={row['dtype']} alpha={row['alpha']}"
            )
        threshold = 1e-10 if row["dtype"] == "float64" else 1e-5
        if row["max_relative_l2"] > threshold:
            failures.append(
                f"round-trip error for dtype={row['dtype']} alpha={row['alpha']} "
                f"is {row['max_relative_l2']:.3e} > {threshold:.1e}"
            )
        if row["metric_min_eigenvalue"] <= 0.0:
            failures.append(
                f"metric is not positive definite for dtype={row['dtype']} "
                f"alpha={row['alpha']}"
            )
    if not jacobian_check["all_converged"]:
        failures.append("SciPy numerical Jacobian did not converge")
    if jacobian_check["max_abs_error"] > 1e-7:
        failures.append(
            "Jacobian maximum absolute error "
            f"{jacobian_check['max_abs_error']:.3e} > 1e-7"
        )
    return failures


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config.seed)
    base_samples = rng.normal(size=(config.dimension, config.num_samples))

    transform64 = build_legendre_transform(config.dimension, np.float64)
    identity64 = np.eye(config.dimension, dtype=np.float64)
    orthogonality_error = float(
        np.max(np.abs(transform64.T @ transform64 - identity64))
    )
    transform_condition_number = float(np.linalg.cond(transform64))

    rows = []
    for dtype in (np.float64, np.float32):
        transform = transform64.astype(dtype)
        samples = base_samples.astype(dtype)
        dtype_rng = np.random.default_rng(config.seed)
        for alpha in config.alphas:
            rows.append(
                mapping_metrics(
                    samples,
                    transform,
                    alpha,
                    np.dtype(dtype).name,
                    dtype_rng,
                )
            )

    diagnostic_alpha = max(config.alphas)
    jacobian_check = validate_numerical_jacobian(
        config.jacobian_dimension, diagnostic_alpha, config.seed
    )
    failures = acceptance(rows, orthogonality_error, jacobian_check)

    summary = {
        "status": "PASS" if not failures else "FAIL",
        "config": asdict(config),
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "legendre": {
            "orthogonality_max_abs_error": orthogonality_error,
            "condition_number": transform_condition_number,
        },
        "jacobian_check": jacobian_check,
        "failures": failures,
    }

    write_csv(output_dir / "roundtrip_metrics.csv", rows)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"metrics={output_dir / 'roundtrip_metrics.csv'}")
    print(f"summary={output_dir / 'summary.json'}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
