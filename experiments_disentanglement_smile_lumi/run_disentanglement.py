"""Disentangle Smile and luminance in the scanned candidate Hilbert subspace."""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
VERIFICATION_DIR = os.path.join(ROOT_DIR, "experiments_ro_verification")
for path in (ROOT_DIR, VERIFICATION_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import parse_args
from disentangle_objective import DisentanglementObjective
from evaluation import save_convergence_plot, save_csv, save_json, save_traversal_grid
from geometry import HilbertComplement, hilbert_inner
from objective import GANImageObjective
from optimizer import optimize_sphere
from scan_candidates import extract_smile_direction


def candidate_span_basis(directions, gram_diag):
    sqrt_g = np.sqrt(gram_diag)
    whitened = sqrt_g[:, None] * directions
    left, singular, _ = np.linalg.svd(whitened, full_matrices=False)
    tolerance = max(whitened.shape) * np.finfo(np.float64).eps * max(singular[0], 1.0)
    rank = int(np.sum(singular > tolerance))
    return left[:, :rank] / sqrt_g[:, None], singular, rank


def null_basis(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    vector = vector / np.linalg.norm(vector)
    left, _, _ = np.linalg.svd(vector[:, None], full_matrices=True)
    return left[:, 1:]


def load_candidate_directions(source_dir, candidate_csv):
    with open(candidate_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    directions = []
    names = []
    for row in rows:
        name = row["ray"]
        payload = np.load(
            os.path.join(source_dir, name, "direction.npz"), allow_pickle=False
        )
        directions.append(
            np.asarray(payload["optimized_function_direction"], dtype=np.float64)
        )
        names.append(name)
    return np.stack(directions, axis=1), names, rows


def copy_templates(source, target):
    target._templates = {name: value.clone() for name, value in source._templates.items()}


def make_objective(generator, basis, config, mode, scales, templates_from=None):
    import torch
    device = next(generator.parameters()).device
    objective = DisentanglementObjective(
        generator,
        torch.as_tensor(basis, dtype=torch.float32, device=device),
        config.epsilon_h,
        config.response_resolution,
        config.microbatch,
        mode,
        smile_scale=scales["smile_power"],
        luminance_scale=scales["luminance_power"],
        curvature_scale=scales["curvature"],
        leakage_weight=1.0,
        curvature_weight=0.05,
    )
    if templates_from is not None:
        copy_templates(templates_from, objective)
    return objective


def save_method_grid(path, directions, anchors, generator, config):
    import torch
    device = next(generator.parameters()).device
    distances = [config.epsilon_h * value for value in (-3, -2, -1, 0, 1, 2, 3)]
    rows = []
    for sample_index in range(min(4, len(anchors))):
        for name, direction in directions:
            q = torch.as_tensor(direction[:, None], dtype=torch.float32, device=device)
            renderer = GANImageObjective(
                generator, q, config.epsilon_h, 0.0,
                config.response_resolution, config.microbatch,
            )
            images = renderer.render_traversal(
                torch.ones(1, device=device), anchors.coefficients[sample_index], distances
            )
            rows.append((f"seed {sample_index:03d} / {name}", images))
    save_traversal_grid(path, rows, distances)


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    output_dir = os.path.join(config.output_dir, "disentangled")
    os.makedirs(output_dir, exist_ok=True)

    import torch
    from utils import factorize_weight, load_generator

    generator = load_generator(config.model)
    device = next(generator.parameters()).device
    source_dir = config.source_dir
    candidate_csv = os.path.join(config.output_dir, "joint_candidates.csv")
    directions, names, candidate_rows = load_candidate_directions(
        source_dir, candidate_csv
    )
    gram_diag = np.load(
        os.path.join(source_dir, names[0], "direction.npz"), allow_pickle=False
    )["gram_diagonal"]
    basis, singular_values, rank = candidate_span_basis(directions, gram_diag)

    search_coefficients = torch.from_numpy(
        np.load(os.path.join(source_dir, "search_anchors.npy"))
    ).float()
    test_coefficients = torch.from_numpy(
        np.load(os.path.join(source_dir, "test_anchors.npy"))
    ).float()
    smile_direction, smile_metadata = extract_smile_direction(generator, factorize_weight)
    smile_renderer = GANImageObjective(
        generator,
        torch.as_tensor(smile_direction[:, None], dtype=torch.float32, device=device),
        config.epsilon_h, 0.0, config.response_resolution, config.microbatch,
    )
    search_anchors = smile_renderer.prepare_anchors("search", search_coefficients)
    test_anchors = smile_renderer.prepare_anchors("test", test_coefficients)
    _, smile_search_response = smile_renderer.evaluate(
        torch.ones(1, device=device), search_anchors, collect_responses=True
    )
    _, smile_test_response = smile_renderer.evaluate(
        torch.ones(1, device=device), test_anchors, collect_responses=True
    )

    calibration = make_objective(
        generator, basis, config, "smile",
        {"smile_power": 1.0, "luminance_power": 1.0, "curvature": 1.0},
    )
    calibration.register_smile_template(search_anchors, smile_search_response)
    calibration.register_smile_template(test_anchors, smile_test_response)
    calibration_rows = []
    coordinates = basis.T @ (gram_diag[:, None] * directions)
    for index, name in enumerate(names):
        point = torch.as_tensor(coordinates[:, index], dtype=torch.float32, device=device)
        point = point / torch.linalg.vector_norm(point)
        metrics, _ = calibration.evaluate(point, search_anchors)
        calibration_rows.append({"ray": name, **metrics})
    scales = {
        key: float(np.median([max(row[key], 1e-12) for row in calibration_rows]))
        for key in ("smile_power", "luminance_power", "curvature")
    }
    save_csv(os.path.join(output_dir, "candidate_calibration.csv"), calibration_rows)

    luminance_objective = make_objective(
        generator, basis, config, "luminance", scales, calibration
    )
    best_luminance_index = int(np.argmax([
        row["luminance_power"] / scales["luminance_power"]
        - row["smile_power"] / scales["smile_power"]
        for row in calibration_rows
    ]))
    initial_luminance = torch.as_tensor(
        coordinates[:, best_luminance_index], dtype=torch.float32, device=device
    )
    initial_luminance /= torch.linalg.vector_norm(initial_luminance)
    geometry_luminance = HilbertComplement(
        np.column_stack([smile_direction])
    )  # Used only for compatible constraint logging dimensions below.
    class BasisGeometry:
        def __init__(self, q, gram): self.q, self.gram = q, gram
        def constraint_errors(self, direction):
            return {
                "orthogonality_l2": 0.0,
                "orthogonality_max_abs": 0.0,
                "hilbert_norm_squared": float(direction @ (self.gram * direction)),
                "hilbert_norm_error": abs(float(direction @ (self.gram * direction)) - 1.0),
            }
    logging_geometry = BasisGeometry(basis, gram_diag)
    z_luminance, history_luminance, runtime_luminance = optimize_sphere(
        luminance_objective, search_anchors, initial_luminance, logging_geometry,
        learning_rate=0.1, min_learning_rate=1e-4, armijo_constant=1e-4,
        line_search_steps=8, max_iterations=30, gradient_tolerance=1e-6,
    )
    v_luminance = basis @ z_luminance.cpu().double().numpy()

    reduced_null = null_basis(z_luminance.cpu().double().numpy())
    smile_basis = basis @ reduced_null
    smile_objective = make_objective(
        generator, smile_basis, config, "smile", scales, calibration
    )
    smile_coordinates = smile_basis.T @ (gram_diag[:, None] * directions)
    candidate_smile_scores = []
    for index in range(len(names)):
        coordinate = smile_coordinates[:, index]
        if np.linalg.norm(coordinate) < 1e-10:
            candidate_smile_scores.append(-np.inf)
            continue
        point = torch.as_tensor(
            coordinate / np.linalg.norm(coordinate), dtype=torch.float32, device=device
        )
        metrics, _ = smile_objective.evaluate(point, search_anchors)
        candidate_smile_scores.append(metrics["objective"])
    initial_smile = torch.as_tensor(
        smile_coordinates[:, int(np.argmax(candidate_smile_scores))],
        dtype=torch.float32, device=device,
    )
    initial_smile /= torch.linalg.vector_norm(initial_smile)
    logging_smile = BasisGeometry(smile_basis, gram_diag)
    z_smile, history_smile, runtime_smile = optimize_sphere(
        smile_objective, search_anchors, initial_smile, logging_smile,
        learning_rate=0.1, min_learning_rate=1e-4, armijo_constant=1e-4,
        line_search_steps=8, max_iterations=30, gradient_tolerance=1e-6,
    )
    v_smile = smile_basis @ z_smile.cpu().double().numpy()

    # Evaluate each final direction with both objectives; raw powers are mode-independent.
    evaluation_rows = []
    for name, direction in (
        ("sefa_smile_template", smile_direction),
        ("luminance_only", v_luminance),
        ("smile_only", v_smile),
    ):
        q = torch.as_tensor(direction[:, None], dtype=torch.float32, device=device)
        evaluator = make_objective(generator, direction[:, None], config, "smile", scales, calibration)
        search_metrics, _ = evaluator.evaluate(torch.ones(1, device=device), search_anchors)
        test_metrics, _ = evaluator.evaluate(torch.ones(1, device=device), test_anchors)
        evaluation_rows.append({
            "direction": name,
            **{f"search_{key}": value for key, value in search_metrics.items()},
            **{f"test_{key}": value for key, value in test_metrics.items()},
        })
    cosine_h = hilbert_inner(v_luminance, v_smile, gram_diag)
    for row in evaluation_rows:
        row["cross_hilbert_cosine"] = cosine_h
    save_csv(os.path.join(output_dir, "disentangled_metrics.csv"), evaluation_rows)
    save_csv(os.path.join(output_dir, "luminance_convergence.csv"), history_luminance)
    save_csv(os.path.join(output_dir, "smile_convergence.csv"), history_smile)
    save_convergence_plot(
        os.path.join(output_dir, "luminance_convergence.png"), history_luminance
    )
    save_convergence_plot(
        os.path.join(output_dir, "smile_convergence.png"), history_smile
    )
    np.savez_compressed(
        os.path.join(output_dir, "disentangled_directions.npz"),
        luminance_direction=v_luminance,
        smile_direction=v_smile,
        gram_diagonal=gram_diag,
        candidate_basis=basis,
        candidate_names=np.asarray(names),
    )
    save_method_grid(
        os.path.join(output_dir, "disentangled_traversal.png"),
        (
            ("SeFa-Smile-template", smile_direction),
            ("Luminance-only", v_luminance),
            ("Smile-only", v_smile),
        ),
        test_anchors, generator, config,
    )
    save_method_grid(
        os.path.join(output_dir, "smile_template_traversal.png"),
        (("SeFa-Smile-template", smile_direction),),
        test_anchors, generator, config,
    )
    save_json(os.path.join(output_dir, "disentanglement_manifest.json"), {
        "candidate_count": len(names),
        "candidate_subspace_rank": rank,
        "candidate_singular_values": singular_values.tolist(),
        "normalization_scales": scales,
        "smile_reference": smile_metadata,
        "luminance_runtime_seconds": runtime_luminance,
        "smile_runtime_seconds": runtime_smile,
        "cross_hilbert_cosine": cosine_h,
    })
    print(f"Candidate subspace rank: {rank}")
    print(f"Cross Hilbert cosine: {cosine_h:.3e}")


if __name__ == "__main__":
    main()

