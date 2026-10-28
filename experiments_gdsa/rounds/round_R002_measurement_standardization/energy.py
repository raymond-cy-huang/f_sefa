"""Numerical path-energy definitions for GDSA Round R002."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import legendre


def build_legendre_transform(dimension: int) -> np.ndarray:
    nodes, weights = legendre.leggauss(dimension)
    orders = np.arange(dimension, dtype=np.float64)
    norms = 2.0 / (2.0 * orders + 1.0)
    basis = legendre.legvander(nodes, dimension - 1)
    return np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]


def metric_scales(transform: np.ndarray, point: np.ndarray, alpha: float) -> np.ndarray:
    coordinates = transform @ point
    return np.ones_like(coordinates) if alpha == 0.0 else np.cosh(alpha * coordinates) ** 2


def metric_quadratic(
    transform: np.ndarray, scales: np.ndarray, vector: np.ndarray
) -> float:
    coordinates = transform @ vector
    return float(np.dot(scales * coordinates, coordinates))


def inverse_metric_apply(
    transform: np.ndarray, scales: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return transform.T @ ((transform @ vector) / scales)


def matched_progress_direction(
    transform: np.ndarray, scales: np.ndarray, target: np.ndarray
) -> np.ndarray:
    inverse_target = inverse_metric_apply(transform, scales, target)
    progress = float(np.dot(target, inverse_target))
    if progress <= np.finfo(np.float64).eps:
        raise ValueError("Target direction has zero inverse-metric norm")
    return inverse_target / progress


@dataclass(frozen=True)
class PathResult:
    points: tuple[np.ndarray, ...]
    directions: tuple[np.ndarray, ...]
    legacy_left_energy: float
    midpoint_energy: float
    target_progress: float


def build_path(
    point: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    alpha: float,
    target_tau: float,
    num_steps: int,
    method: str,
) -> PathResult:
    if num_steps <= 0 or target_tau <= 0:
        raise ValueError("num_steps and target_tau must be positive")
    if method not in {"B1_euclidean", "B2_fixed_metric", "B3_dynamic_metric"}:
        raise ValueError(f"Unknown method: {method}")

    delta_t = target_tau / num_steps
    fixed_scales = metric_scales(transform, point, alpha)
    fixed_direction = matched_progress_direction(transform, fixed_scales, target)
    euclidean_direction = target / float(np.dot(target, target))
    current = point.copy()
    points = [current.copy()]
    directions = []
    legacy_energy = 0.0
    midpoint_energy = 0.0

    for _ in range(num_steps):
        left_scales = metric_scales(transform, current, alpha)
        if method == "B1_euclidean":
            direction = euclidean_direction
        elif method == "B2_fixed_metric":
            direction = fixed_direction
        else:
            direction = matched_progress_direction(transform, left_scales, target)

        delta_w = delta_t * direction
        next_point = current + delta_w
        midpoint = 0.5 * (current + next_point)
        midpoint_scales = metric_scales(transform, midpoint, alpha)

        legacy_energy += metric_quadratic(transform, left_scales, delta_w)
        midpoint_energy += 0.5 * metric_quadratic(
            transform, midpoint_scales, delta_w
        ) / delta_t

        directions.append(direction.copy())
        current = next_point
        points.append(current.copy())

    return PathResult(
        points=tuple(points),
        directions=tuple(directions),
        legacy_left_energy=legacy_energy,
        midpoint_energy=midpoint_energy,
        target_progress=float(np.dot(target, current - point)),
    )
