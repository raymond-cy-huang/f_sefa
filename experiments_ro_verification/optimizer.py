"""Riemannian gradient ascent on the reduced-coordinate unit sphere."""

from __future__ import annotations

import time

import torch

from geometry import sphere_retraction, sphere_tangent_projection


def optimize_sphere(
    objective,
    anchors,
    initial: torch.Tensor,
    geometry,
    learning_rate: float,
    min_learning_rate: float,
    armijo_constant: float,
    line_search_steps: int,
    max_iterations: int,
    gradient_tolerance: float,
):
    """Maximize an image objective with projected gradients and retraction."""
    point = initial.detach().to(objective.device, dtype=torch.float32)
    point = point / torch.linalg.vector_norm(point)
    history = []
    start_time = time.perf_counter()

    for iteration in range(max_iterations + 1):
        metrics, ambient_gradient = objective.value_and_gradient(point, anchors)
        riemannian_gradient = sphere_tangent_projection(point, ambient_gradient)
        gradient_norm = float(torch.linalg.vector_norm(riemannian_gradient).cpu())
        tangent_error = abs(float(torch.dot(point, riemannian_gradient).cpu()))
        direction_np = (objective.q @ point).detach().cpu().double().numpy()
        constraints = geometry.constraint_errors(direction_np)
        row = {
            "iteration": iteration,
            **metrics,
            "euclidean_gradient_norm": float(
                torch.linalg.vector_norm(ambient_gradient).cpu()
            ),
            "riemannian_gradient_norm": gradient_norm,
            "tangent_error": tangent_error,
            **constraints,
            "accepted_step_size": 0.0,
            "line_search_evaluations": 0,
        }
        history.append(row)
        if iteration == max_iterations or gradient_norm <= gradient_tolerance:
            break

        step = float(learning_rate)
        accepted = None
        evaluations = 0
        grad_norm_squared = gradient_norm**2
        while step >= min_learning_rate and evaluations < line_search_steps:
            candidate = sphere_retraction(point, step * riemannian_gradient)
            candidate_metrics, _ = objective.evaluate(candidate, anchors)
            evaluations += 1
            sufficient_increase = (
                metrics["objective"] + armijo_constant * step * grad_norm_squared
            )
            if candidate_metrics["objective"] >= sufficient_increase:
                accepted = candidate.detach()
                break
            step *= 0.5

        history[-1]["accepted_step_size"] = step if accepted is not None else 0.0
        history[-1]["line_search_evaluations"] = evaluations
        if accepted is None:
            history[-1]["stop_reason"] = "line_search_failed"
            break
        point = accepted

    runtime = time.perf_counter() - start_time
    return point.detach(), history, runtime

