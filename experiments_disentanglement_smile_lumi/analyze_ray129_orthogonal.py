"""Analyze ray129 against existing joint candidates and rotate its best G-orthogonal pair."""

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

from evaluation import save_csv, save_json, save_traversal_grid
from objective import GANImageObjective
from utils import load_generator


REFERENCE_RAY = "ray129"
ORTHOGONAL_TOLERANCE = 1e-8
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def load_direction(source_dir, ray_name):
    payload = np.load(
        os.path.join(source_dir, ray_name, "direction.npz"), allow_pickle=False
    )
    return (
        np.asarray(payload["optimized_function_direction"], dtype=np.float64),
        np.asarray(payload["gram_diagonal"], dtype=np.float64),
    )


def hilbert_cosine(left, right, gram):
    numerator = float(np.dot(left * gram, right))
    denominator = np.sqrt(np.dot(left * gram, left) * np.dot(right * gram, right))
    return numerator / denominator


def normalize_hilbert(direction, gram):
    return direction / np.sqrt(np.dot(direction * gram, direction))


def evaluate_response(generator, direction, anchors, epsilon_h, resolution, microbatch):
    import torch

    device = next(generator.parameters()).device
    renderer = GANImageObjective(
        generator,
        torch.as_tensor(direction[:, None], dtype=torch.float32, device=device),
        epsilon_h,
        curvature_beta=0.0,
        resolution=resolution,
        microbatch=microbatch,
    )
    anchor_set = renderer.prepare_anchors("ray129_pair", anchors)
    _, flat = renderer.evaluate(
        torch.ones(1, dtype=torch.float32, device=device),
        anchor_set,
        collect_responses=True,
    )
    response = flat.reshape(-1, 3, resolution, resolution).astype(np.float64)
    luma = np.sum(response.mean(axis=(2, 3)) * LUMA_WEIGHTS[None, :], axis=1)
    return anchor_set, response, luma


def render_grid(path, named_directions, anchor_set, generator, epsilon_h, resolution, microbatch):
    import torch

    device = next(generator.parameters()).device
    distances = [epsilon_h * value for value in (-3, -2, -1, 0, 1, 2, 3)]
    rows = []
    for sample_index in range(min(4, len(anchor_set))):
        for name, direction in named_directions:
            renderer = GANImageObjective(
                generator,
                torch.as_tensor(direction[:, None], dtype=torch.float32, device=device),
                epsilon_h,
                curvature_beta=0.0,
                resolution=resolution,
                microbatch=microbatch,
            )
            images = renderer.render_traversal(
                torch.ones(1, dtype=torch.float32, device=device),
                anchor_set.coefficients[sample_index],
                distances,
            )
            rows.append((f"seed {sample_index:03d} / {name}", images))
    save_traversal_grid(path, rows, distances)


def main():
    source_dir = os.path.join(ROOT_DIR, "experiments_ro_complete_orth")
    candidate_csv = os.path.join(
        EXPERIMENT_DIR, "outputs", "joint_candidates.csv"
    )
    output_dir = os.path.join(
        EXPERIMENT_DIR, "outputs", "ray129_orthogonal"
    )
    os.makedirs(output_dir, exist_ok=True)

    with open(candidate_csv, newline="", encoding="utf-8") as handle:
        candidate_rows = list(csv.DictReader(handle))

    reference, gram = load_direction(source_dir, REFERENCE_RAY)
    reference = normalize_hilbert(reference, gram)
    orthogonality_rows = []
    for row in candidate_rows:
        direction, candidate_gram = load_direction(source_dir, row["ray"])
        if not np.allclose(candidate_gram, gram):
            raise ValueError(f"Gram mismatch for {row['ray']}")
        direction = normalize_hilbert(direction, gram)
        cosine = hilbert_cosine(reference, direction, gram)
        orthogonality_rows.append({
            "ray": row["ray"],
            "hilbert_cosine_to_ray129": cosine,
            "abs_hilbert_cosine_to_ray129": abs(cosine),
            "is_exactly_orthogonal": abs(cosine) <= ORTHOGONAL_TOLERANCE,
            "joint_candidate_score": float(row["joint_candidate_score"]),
            "mouth_smile_alignment_mean": float(row["mouth_smile_alignment_mean"]),
            "luminance_slope_mean": float(row["luminance_slope_mean"]),
        })

    exact = [row for row in orthogonality_rows if row["is_exactly_orthogonal"]]
    exact.sort(key=lambda row: row["joint_candidate_score"], reverse=True)
    if not exact:
        raise RuntimeError("No existing joint candidate is G-orthogonal to ray129.")
    partner_name = exact[0]["ray"]
    partner, _ = load_direction(source_dir, partner_name)
    partner = normalize_hilbert(partner, gram)
    if hilbert_cosine(reference, partner, gram) < 0:
        partner = -partner

    generator = load_generator("e4e_ffhq_encode")
    import torch

    anchors = torch.from_numpy(
        np.load(os.path.join(source_dir, "test_anchors.npy"))
    ).float()
    epsilon_h = 0.1
    resolution = 64
    microbatch = 1
    anchor_set, _, reference_luma = evaluate_response(
        generator, reference, anchors, epsilon_h, resolution, microbatch
    )
    _, _, partner_luma = evaluate_response(
        generator, partner, anchors, epsilon_h, resolution, microbatch
    )

    # Minimize held-out luminance derivative in span{ray129, partner}.
    alpha = -float(np.dot(reference_luma, partner_luma)) / max(
        float(np.dot(partner_luma, partner_luma)), 1e-12
    )
    normalization = np.sqrt(1.0 + alpha * alpha)
    luminance_cancelled = normalize_hilbert(
        (reference + alpha * partner) / normalization, gram
    )
    orthogonal_complement = normalize_hilbert(
        (-alpha * reference + partner) / normalization, gram
    )

    _, _, cancelled_luma = evaluate_response(
        generator, luminance_cancelled, anchors, epsilon_h, resolution, microbatch
    )
    _, _, complement_luma = evaluate_response(
        generator, orthogonal_complement, anchors, epsilon_h, resolution, microbatch
    )
    if float(np.mean(complement_luma)) < 0:
        orthogonal_complement = -orthogonal_complement
        complement_luma = -complement_luma

    named = (
        (REFERENCE_RAY, reference),
        (partner_name, partner),
        ("ray129-luminance-cancelled", luminance_cancelled),
        ("G-orthogonal-complement", orthogonal_complement),
    )
    render_grid(
        os.path.join(output_dir, "ray129_orthogonal_rotation.png"),
        named, anchor_set, generator, epsilon_h, resolution, microbatch,
    )
    save_csv(
        os.path.join(output_dir, "candidate_orthogonality.csv"),
        sorted(
            orthogonality_rows,
            key=lambda row: (
                not row["is_exactly_orthogonal"],
                -row["joint_candidate_score"],
            ),
        ),
    )
    metric_rows = []
    for name, direction, luma in (
        (REFERENCE_RAY, reference, reference_luma),
        (partner_name, partner, partner_luma),
        ("ray129-luminance-cancelled", luminance_cancelled, cancelled_luma),
        ("G-orthogonal-complement", orthogonal_complement, complement_luma),
    ):
        metric_rows.append({
            "direction": name,
            "mean_luminance_slope": float(np.mean(luma)),
            "rms_luminance_slope": float(np.sqrt(np.mean(luma ** 2))),
            "hilbert_cosine_to_ray129": hilbert_cosine(reference, direction, gram),
            "hilbert_cosine_to_cancelled_axis": hilbert_cosine(
                luminance_cancelled, direction, gram
            ),
        })
    save_csv(os.path.join(output_dir, "rotation_metrics.csv"), metric_rows)
    np.savez_compressed(
        os.path.join(output_dir, "rotated_directions.npz"),
        ray129=reference,
        partner=partner,
        luminance_cancelled=luminance_cancelled,
        orthogonal_complement=orthogonal_complement,
        gram_diagonal=gram,
    )
    save_json(os.path.join(output_dir, "manifest.json"), {
        "reference_ray": REFERENCE_RAY,
        "orthogonality_tolerance": ORTHOGONAL_TOLERANCE,
        "exactly_orthogonal_candidate_count": len(exact),
        "selected_partner": partner_name,
        "selection_rule": "Highest existing joint-candidate score among rays exactly G-orthogonal to ray129.",
        "rotation_alpha": alpha,
        "final_cross_hilbert_cosine": hilbert_cosine(
            luminance_cancelled, orthogonal_complement, gram
        ),
    })
    print(f"Exact G-orthogonal candidates: {len(exact)}")
    print(f"Selected partner: {partner_name}")
    print(f"Rotation alpha: {alpha:.8f}")
    print(
        "Final cross Hilbert cosine: "
        f"{hilbert_cosine(luminance_cancelled, orthogonal_complement, gram):.3e}"
    )


if __name__ == "__main__":
    main()
