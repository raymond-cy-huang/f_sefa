"""Hilbert geometry for Legendre-coefficient function-space directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


def legendre_gram_diagonal(dimension: int) -> np.ndarray:
    """Return G_kk = integral_{-1}^{1} P_k(x)^2 dx."""
    if dimension <= 0:
        raise ValueError("dimension must be positive.")
    orders = np.arange(dimension, dtype=np.float64)
    return 2.0 / (2.0 * orders + 1.0)


def hilbert_inner(left: np.ndarray, right: np.ndarray, gram_diag: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    return float(np.dot(left * gram_diag, right))


def hilbert_norm(vector: np.ndarray, gram_diag: np.ndarray) -> float:
    value = hilbert_inner(vector, vector, gram_diag)
    return float(np.sqrt(max(value, 0.0)))


@dataclass(frozen=True)
class GeometryDiagnostics:
    ambient_dimension: int
    known_rank: int
    complement_dimension: int
    max_known_complement_inner: float
    max_complement_gram_error: float


class HilbertComplement:
    """G-unit sphere inside the G-orthogonal complement of known directions.

    Function-space elements are represented by their raw Legendre coefficients.
    If L = sqrt(G), an orthonormal basis B of col(LU)^perp gives Q=L^-1 B,
    hence U^T G Q=0 and Q^T G Q=I. Candidate functions are v=Qz with
    ||z||_2=1.
    """

    def __init__(self, known_directions: np.ndarray, gram_diag: np.ndarray | None = None):
        known = np.asarray(known_directions, dtype=np.float64)
        if known.ndim != 2:
            raise ValueError("known_directions must have shape [dimension, count].")
        dimension, count = known.shape
        if count == 0 or count >= dimension:
            raise ValueError("known direction count must be in [1, dimension-1].")
        if not np.all(np.isfinite(known)):
            raise ValueError("known_directions contains non-finite values.")

        gram = (
            legendre_gram_diagonal(dimension)
            if gram_diag is None
            else np.asarray(gram_diag, dtype=np.float64)
        )
        if gram.shape != (dimension,) or np.any(gram <= 0):
            raise ValueError("gram_diag must be a positive vector of ambient dimension.")

        # Individual normalization improves conditioning without changing the span.
        norms = np.sqrt(np.sum(known * gram[:, None] * known, axis=0))
        if np.any(norms <= np.finfo(np.float64).eps):
            raise ValueError("A known semantic direction has zero Hilbert norm.")
        known = known / norms[None, :]

        sqrt_g = np.sqrt(gram)
        whitened_known = sqrt_g[:, None] * known
        left_vectors, singular_values, _ = np.linalg.svd(
            whitened_known, full_matrices=True
        )
        tolerance = (
            max(whitened_known.shape)
            * np.finfo(np.float64).eps
            * max(float(singular_values[0]), 1.0)
        )
        rank = int(np.sum(singular_values > tolerance))
        if rank != count:
            raise ValueError(
                f"Known semantic directions are rank deficient: rank={rank}, count={count}."
            )
        whitened_complement = left_vectors[:, rank:]
        complement = whitened_complement / sqrt_g[:, None]

        self.dimension = dimension
        self.gram_diag = gram
        self.known = known
        self.q = complement
        self.known_rank = rank

        known_error = np.max(np.abs(known.T @ (gram[:, None] * complement)))
        gram_error = np.max(
            np.abs(complement.T @ (gram[:, None] * complement) - np.eye(dimension - rank))
        )
        self.diagnostics = GeometryDiagnostics(
            ambient_dimension=dimension,
            known_rank=rank,
            complement_dimension=dimension - rank,
            max_known_complement_inner=float(known_error),
            max_complement_gram_error=float(gram_error),
        )

    @property
    def complement_dimension(self) -> int:
        return self.q.shape[1]

    def direction(self, reduced: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced, dtype=np.float64).reshape(-1)
        if reduced.shape != (self.complement_dimension,):
            raise ValueError("Unexpected reduced-coordinate shape.")
        return self.q @ reduced

    def normalize_reduced(self, reduced: np.ndarray) -> np.ndarray:
        reduced = np.asarray(reduced, dtype=np.float64).reshape(-1)
        norm = float(np.linalg.norm(reduced))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("Cannot normalize a zero reduced vector.")
        return reduced / norm

    def constraint_errors(self, direction: np.ndarray) -> dict[str, float]:
        direction = np.asarray(direction, dtype=np.float64).reshape(-1)
        orthogonality = self.known.T @ (self.gram_diag * direction)
        norm_squared = hilbert_inner(direction, direction, self.gram_diag)
        return {
            "orthogonality_l2": float(np.linalg.norm(orthogonality)),
            "orthogonality_max_abs": float(np.max(np.abs(orthogonality))),
            "hilbert_norm_squared": norm_squared,
            "hilbert_norm_error": abs(norm_squared - 1.0),
        }

    def project_and_normalize(self, vector: np.ndarray) -> np.ndarray:
        """Project an ambient function onto the complement and H-normalize it."""
        vector = np.asarray(vector, dtype=np.float64).reshape(-1)
        coordinates = self.q.T @ (self.gram_diag * vector)
        coordinates = self.normalize_reduced(coordinates)
        return self.q @ coordinates

    def torch_q(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.q, dtype=dtype, device=device)

    def torch_gram(self, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return torch.as_tensor(self.gram_diag, dtype=dtype, device=device)


def sphere_tangent_projection(point: torch.Tensor, ambient_gradient: torch.Tensor) -> torch.Tensor:
    """Project a Euclidean gradient onto T_point S^(d-1)."""
    return ambient_gradient - point * torch.dot(point, ambient_gradient)


def sphere_retraction(point: torch.Tensor, tangent_step: torch.Tensor) -> torch.Tensor:
    """Normalization retraction on the Euclidean unit sphere."""
    candidate = point + tangent_step
    norm = torch.linalg.vector_norm(candidate)
    if float(norm.detach().cpu()) <= torch.finfo(candidate.dtype).eps:
        raise ValueError("Retraction received a zero candidate.")
    return candidate / norm

