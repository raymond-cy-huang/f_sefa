"""MVV for training-free explicit orthogonal semantic mapping.

This script numerically validates a simple analytical inverse for a nonlinear
latent trajectory:

    V(t) = w0 + alpha1 * phi1(t) * n1 + alpha2 * phi2(t) * n2

where n1 and n2 mimic orthogonal SeFa semantic directions and phi1, phi2 are
explicit Legendre basis functions on [-1, 1]. The inverse uses only vector dot
products and discrete inner products, with no optimizer, network, or training.
"""

import argparse
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration for the pure numerical validation."""

    latent_dim: int = 512
    num_samples: int = 100
    alpha1: float = 2.0
    alpha2: float = -1.5
    seed: int = 7
    t_min: float = -1.0
    t_max: float = 1.0
    max_basis_abs: float = 1.0


def parse_args() -> ExperimentConfig:
    """Parses optional command line overrides."""

    parser = argparse.ArgumentParser(
        description=(
            "Validate a training-free analytical inverse for explicit "
            "orthogonal nonlinear semantic mapping."
        )
    )
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--alpha1", type=float, default=2.0)
    parser.add_argument("--alpha2", type=float, default=-1.5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t_min", type=float, default=-1.0)
    parser.add_argument("--t_max", type=float, default=1.0)
    return ExperimentConfig(**vars(parser.parse_args()))


def normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Returns a unit vector, guarding against degenerate random draws."""

    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def make_orthonormal_pair(
    latent_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Creates two orthonormal vectors with Gram-Schmidt orthogonalization."""

    n1 = normalize(rng.standard_normal(latent_dim))

    raw_n2 = rng.standard_normal(latent_dim)
    raw_n2 = raw_n2 - np.dot(raw_n2, n1) * n1
    n2 = normalize(raw_n2)

    return n1, n2


def legendre_basis(t: np.ndarray, max_abs: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Evaluates the first two non-constant Legendre polynomials on [-1, 1].

    P1(t) = t
    P2(t) = 0.5 * (3t^2 - 1)

    The explicit boundary check prevents silent numerical blow-up if callers
    accidentally sample outside the intended orthogonal interval.
    """

    if np.any(t < -1.0) or np.any(t > 1.0):
        raise ValueError("Legendre basis is constrained to t in [-1, 1].")

    phi1 = t
    phi2 = 0.5 * (3.0 * t**2 - 1.0)

    if not np.all(np.isfinite(phi1)) or not np.all(np.isfinite(phi2)):
        raise FloatingPointError("Basis evaluation produced non-finite values.")
    if max(np.max(np.abs(phi1)), np.max(np.abs(phi2))) > max_abs + 1e-12:
        raise FloatingPointError("Basis magnitude exceeded the configured bound.")

    return phi1, phi2


def forward_trajectory(
    w0: np.ndarray,
    n1: np.ndarray,
    n2: np.ndarray,
    phi1: np.ndarray,
    phi2: np.ndarray,
    alpha1: float,
    alpha2: float,
) -> np.ndarray:
    """Constructs V with shape (num_samples, latent_dim)."""

    return (
        w0[None, :]
        + alpha1 * phi1[:, None] * n1[None, :]
        + alpha2 * phi2[:, None] * n2[None, :]
    )


def discrete_inner_product(values_a: np.ndarray, values_b: np.ndarray, dt: float) -> float:
    """Approximates integral a(t)b(t)dt with a uniform Riemann sum."""

    return float(dt * np.sum(values_a * values_b))


def analytical_inverse(
    V: np.ndarray,
    w0: np.ndarray,
    direction: np.ndarray,
    basis_values: np.ndarray,
    dt: float,
) -> float:
    """Recovers alpha_i using only projections and discrete integration."""

    displacement = V - w0[None, :]
    projected_curve = displacement @ direction

    numerator = discrete_inner_product(projected_curve, basis_values, dt)
    denominator = discrete_inner_product(basis_values, basis_values, dt)
    if abs(denominator) < 1e-12:
        raise ZeroDivisionError("Basis norm is too small for stable inversion.")

    return numerator / denominator


def main() -> None:
    """Runs the minimum viable numerical validation."""

    config = parse_args()
    rng = np.random.default_rng(config.seed)

    # 1. Simulate a latent anchor and two SeFa-like orthonormal directions.
    w0 = rng.standard_normal(config.latent_dim)
    n1, n2 = make_orthonormal_pair(config.latent_dim, rng)

    # 2. Sample the bounded orthogonal interval and evaluate explicit bases.
    t = np.linspace(config.t_min, config.t_max, config.num_samples)
    t = np.clip(t, -1.0, 1.0)
    dt = float((t[-1] - t[0]) / (len(t) - 1))
    phi1, phi2 = legendre_basis(t, max_abs=config.max_basis_abs)

    # 3. Forward map: bend SeFa's linear directions into a nonlinear curve.
    V = forward_trajectory(
        w0=w0,
        n1=n1,
        n2=n2,
        phi1=phi1,
        phi2=phi2,
        alpha1=config.alpha1,
        alpha2=config.alpha2,
    )

    # 4. Analytical inverse: recover coefficients with dot products and sums.
    estimated_alpha1 = analytical_inverse(V, w0, n1, phi1, dt)
    estimated_alpha2 = analytical_inverse(V, w0, n2, phi2, dt)

    true_alphas = np.array([config.alpha1, config.alpha2])
    estimated_alphas = np.array([estimated_alpha1, estimated_alpha2])
    mse = float(np.mean((true_alphas - estimated_alphas) ** 2))

    # Diagnostics make the numerical assumptions visible.
    direction_dot = float(np.dot(n1, n2))
    phi_cross = discrete_inner_product(phi1, phi2, dt)
    phi1_norm = discrete_inner_product(phi1, phi1, dt)
    phi2_norm = discrete_inner_product(phi2, phi2, dt)

    print("=== Explicit Orthogonal Mapping MVV ===")
    print(f"Latent dimension: {config.latent_dim}")
    print(f"Number of t samples: {config.num_samples}")
    print(f"Trajectory V shape: {V.shape}")
    print()
    print("Ground truth coefficients:")
    print(f"  alpha1 = {config.alpha1:.12f}")
    print(f"  alpha2 = {config.alpha2:.12f}")
    print()
    print("Analytically recovered coefficients:")
    print(f"  estimated_alpha1 = {estimated_alpha1:.12f}")
    print(f"  estimated_alpha2 = {estimated_alpha2:.12f}")
    print()
    print("Error:")
    print(f"  MSE = {mse:.16e}")
    print()
    print("Numerical diagnostics:")
    print(f"  dot(n1, n2)              = {direction_dot:.16e}")
    print(f"  discrete <phi1, phi2>    = {phi_cross:.16e}")
    print(f"  discrete <phi1, phi1>    = {phi1_norm:.16e}")
    print(f"  discrete <phi2, phi2>    = {phi2_norm:.16e}")


if __name__ == "__main__":
    main()
