"""Matrix-free semantic-response operator primitives.

The implementation is intentionally independent of StyleGAN.  It accepts JVP
and VJP callables, which allows the same algebra to be tested on exact linear
controls before attaching an autograd-backed generator in a later pilot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


Vector = np.ndarray


@dataclass
class SemanticResponseOperator:
    """Coordinate-aware wrapper around ``J.T @ J`` without materializing J."""

    q: np.ndarray
    jvp_w: Callable[[Vector], Vector]
    vjp_w: Callable[[Vector], Vector]

    def __post_init__(self) -> None:
        self.q = np.asarray(self.q, dtype=np.float64)
        if self.q.ndim != 2 or self.q.shape[0] != self.q.shape[1]:
            raise ValueError("q must be a square matrix")
        error = np.linalg.norm(self.q.T @ self.q - np.eye(self.q.shape[0]), ord=2)
        if error > 1e-10:
            raise ValueError(f"q is not orthogonal: spectral error {error:.3e}")

    @property
    def dimension(self) -> int:
        return self.q.shape[0]

    def to_h(self, vector_w: Vector) -> Vector:
        return self.q @ np.asarray(vector_w, dtype=np.float64)

    def to_w(self, vector_h: Vector) -> Vector:
        return self.q.T @ np.asarray(vector_h, dtype=np.float64)

    def apply_a(self, vector_h: Vector) -> Vector:
        return np.asarray(self.jvp_w(self.to_w(vector_h)), dtype=np.float64)

    def apply_adjoint(self, output_vector: Vector) -> Vector:
        return self.to_h(np.asarray(self.vjp_w(output_vector), dtype=np.float64))

    def k_mv(self, vector_h: Vector) -> Vector:
        return self.apply_adjoint(self.apply_a(vector_h))

    def explicit_k(self) -> np.ndarray:
        eye = np.eye(self.dimension)
        return np.column_stack([self.k_mv(eye[:, index]) for index in range(self.dimension)])

    def solve_full(self, target_h: Vector, mu: float) -> Vector:
        if mu <= 0:
            raise ValueError("mu must be positive")
        target_h = np.asarray(target_h, dtype=np.float64)
        solution = np.linalg.solve(self.explicit_k() + mu * np.eye(self.dimension), target_h)
        denominator = float(target_h @ solution)
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("non-positive normalization denominator")
        return solution / denominator

    def solve_diag(self, target_h: Vector, diagonal: Vector, mu: float) -> Vector:
        if mu <= 0:
            raise ValueError("mu must be positive")
        target_h = np.asarray(target_h, dtype=np.float64)
        diagonal = np.asarray(diagonal, dtype=np.float64)
        if diagonal.shape != target_h.shape or np.any(diagonal < 0):
            raise ValueError("diagonal must be non-negative and match target")
        solution = target_h / (diagonal + mu)
        denominator = float(target_h @ solution)
        if denominator <= np.finfo(np.float64).eps:
            raise ValueError("non-positive normalization denominator")
        return solution / denominator

    def estimate_diag(self, probes: int, seed: int) -> Vector:
        """Unbiased Hutchinson diagonal estimator using Rademacher probes."""

        if probes <= 0:
            raise ValueError("probes must be positive")
        generator = np.random.default_rng(seed)
        estimate = np.zeros(self.dimension, dtype=np.float64)
        for _ in range(probes):
            z = generator.choice((-1.0, 1.0), size=self.dimension)
            estimate += z * self.k_mv(z)
        return estimate / probes


def explicit_linear_operator(jacobian: np.ndarray, q: np.ndarray) -> SemanticResponseOperator:
    jacobian = np.asarray(jacobian, dtype=np.float64)
    return SemanticResponseOperator(
        q=np.asarray(q, dtype=np.float64),
        jvp_w=lambda vector: jacobian @ vector,
        vjp_w=lambda vector: jacobian.T @ vector,
    )

