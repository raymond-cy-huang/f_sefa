"""Find rays that jointly express Smile-like and luminance responses."""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
VERIFICATION_DIR = os.path.join(ROOT_DIR, "experiments_ro_verification")
for path in (ROOT_DIR, VERIFICATION_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import parse_args
from evaluation import save_csv, save_json
from geometry import legendre_gram_diagonal
from objective import GANImageObjective


SMILE_RANK = 18
LUMA_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)


def extract_smile_direction(generator, factorize_weight):
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    source_index = int(order[SMILE_RANK])
    direction = boundaries[source_index].reshape(-1)
    gram = legendre_gram_diagonal(len(direction))
    norm = np.sqrt(np.dot(direction * gram, direction))
    return direction / norm, {
        "semantic": "smile",
        "eigen_rank": SMILE_RANK,
        "source_eigen_index": source_index,
        "eigenvalue": float(eigenvalues[source_index]),
    }


def cosine_rows(left, right, eps=1e-12):
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, eps)


def response_features(flat_responses, resolution):
    response = np.asarray(flat_responses, dtype=np.float64).reshape(
        -1, 3, resolution, resolution
    )
    channel_means = response.mean(axis=(2, 3), keepdims=True)
    spatial_centered = response - channel_means
    y0, y1 = int(round(0.52 * resolution)), int(round(0.84 * resolution))
    x0, x1 = int(round(0.24 * resolution)), int(round(0.76 * resolution))
    mouth = spatial_centered[:, :, y0:y1, x0:x1].reshape(len(response), -1)
    centered_full = spatial_centered.reshape(len(response), -1)
    luminance = np.sum(
        response.mean(axis=(2, 3)) * LUMA_WEIGHTS[None, :], axis=1
    )
    mouth_energy = np.mean(mouth**2, axis=1)
    full_energy = np.mean(response**2, axis=(1, 2, 3))
    return {
        "raw": response,
        "centered_full": centered_full,
        "mouth": mouth,
        "luminance_slope": luminance,
        "mouth_energy": mouth_energy,
        "full_energy": full_energy,
    }


def load_statuses(source_dir):
    path = os.path.join(source_dir, "complete_orth_summary.csv")
    with open(path, newline="", encoding="utf-8") as handle:
        return {row["ray"]: row for row in csv.DictReader(handle)}


def save_scatter(path, rows, smile_threshold, luminance_threshold):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.asarray([float(row["abs_luminance_slope_mean"]) for row in rows])
    y = np.asarray([float(row["abs_mouth_smile_alignment_mean"]) for row in rows])
    colors = ["#d62728" if row["is_joint_candidate"] else "#4c78a8" for row in rows]
    figure, axis = plt.subplots(figsize=(8.0, 6.2))
    axis.scatter(x, y, c=colors, s=20, alpha=0.72)
    axis.axvline(luminance_threshold, color="black", linestyle="--", linewidth=1)
    axis.axhline(smile_threshold, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Mean absolute luminance slope")
    axis.set_ylabel("Absolute mean mouth Smile-response alignment")
    axis.set_title("Smile + luminance candidate scan")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)

    import torch
    from utils import load_generator, factorize_weight

    if not torch.cuda.is_available():
        raise RuntimeError("Candidate scanning requires CUDA.")
    started = time.perf_counter()
    generator = load_generator(config.model)
    device = next(generator.parameters()).device
    smile_direction, smile_metadata = extract_smile_direction(
        generator, factorize_weight
    )
    anchors_path = os.path.join(config.source_dir, "test_anchors.npy")
    anchors = torch.from_numpy(np.load(anchors_path)).float()

    smile_q = torch.as_tensor(
        smile_direction[:, None], dtype=torch.float32, device=device
    )
    smile_objective = GANImageObjective(
        generator,
        smile_q,
        config.epsilon_h,
        curvature_beta=0.0,
        resolution=config.response_resolution,
        microbatch=config.microbatch,
    )
    anchor_set = smile_objective.prepare_anchors("held_out", anchors)
    _, smile_responses = smile_objective.evaluate(
        torch.ones(1, dtype=torch.float32, device=device),
        anchor_set,
        collect_responses=True,
    )
    smile_features = response_features(smile_responses, config.response_resolution)
    statuses = load_statuses(config.source_dir)

    rows = []
    per_sample_by_ray = {}
    ray_names = sorted(
        name for name in os.listdir(config.source_dir)
        if name.startswith("ray") and name[3:].isdigit()
    )
    for position, ray_name in enumerate(ray_names, start=1):
        direction_path = os.path.join(config.source_dir, ray_name, "direction.npz")
        if not os.path.exists(direction_path):
            continue
        payload = np.load(direction_path, allow_pickle=False)
        direction = np.asarray(payload["optimized_function_direction"], dtype=np.float64)
        direction_q = torch.as_tensor(
            direction[:, None], dtype=torch.float32, device=device
        )
        ray_objective = GANImageObjective(
            generator,
            direction_q,
            config.epsilon_h,
            curvature_beta=0.0,
            resolution=config.response_resolution,
            microbatch=config.microbatch,
        )
        _, ray_responses = ray_objective.evaluate(
            torch.ones(1, dtype=torch.float32, device=device),
            anchor_set,
            collect_responses=True,
        )
        features = response_features(ray_responses, config.response_resolution)
        mouth_cosines = cosine_rows(features["mouth"], smile_features["mouth"])
        full_cosines = cosine_rows(
            features["centered_full"], smile_features["centered_full"]
        )
        luminance = features["luminance_slope"]
        sample_rows = []
        for sample_index in range(len(luminance)):
            sample_rows.append({
                "ray": ray_name,
                "sample_index": sample_index,
                "mouth_smile_alignment": float(mouth_cosines[sample_index]),
                "full_centered_smile_alignment": float(full_cosines[sample_index]),
                "luminance_slope": float(luminance[sample_index]),
                "mouth_response_energy": float(features["mouth_energy"][sample_index]),
                "full_response_energy": float(features["full_energy"][sample_index]),
            })
        per_sample_by_ray[ray_name] = sample_rows
        status_row = statuses.get(ray_name, {})
        rows.append({
            "ray": ray_name,
            "source_status": status_row.get("status", "unknown"),
            "mouth_smile_alignment_mean": float(np.mean(mouth_cosines)),
            "abs_mouth_smile_alignment_mean": abs(float(np.mean(mouth_cosines))),
            "mouth_smile_alignment_abs_mean": float(np.mean(np.abs(mouth_cosines))),
            "full_centered_smile_alignment_mean": float(np.mean(full_cosines)),
            "abs_full_centered_smile_alignment_mean": abs(float(np.mean(full_cosines))),
            "luminance_slope_mean": float(np.mean(luminance)),
            "abs_luminance_slope_mean": float(np.mean(np.abs(luminance))),
            "mouth_response_energy_mean": float(np.mean(features["mouth_energy"])),
            "full_response_energy_mean": float(np.mean(features["full_energy"])),
        })
        if position % 25 == 0 or position == len(ray_names):
            print(f"Scanned {position}/{len(ray_names)} rays", flush=True)

    smile_values = np.asarray(
        [row["abs_mouth_smile_alignment_mean"] for row in rows], dtype=np.float64
    )
    luminance_values = np.asarray(
        [row["abs_luminance_slope_mean"] for row in rows], dtype=np.float64
    )
    smile_threshold = float(np.quantile(smile_values, config.quantile_threshold))
    luminance_threshold = float(
        np.quantile(luminance_values, config.quantile_threshold)
    )
    candidates = []
    for row in rows:
        row["is_joint_candidate"] = bool(
            row["abs_mouth_smile_alignment_mean"] >= smile_threshold
            and row["abs_luminance_slope_mean"] >= luminance_threshold
        )
        row["joint_candidate_score"] = float(
            (row["abs_mouth_smile_alignment_mean"] / max(smile_threshold, 1e-12))
            * (row["abs_luminance_slope_mean"] / max(luminance_threshold, 1e-12))
        )
        if row["is_joint_candidate"]:
            candidates.append(row)

    candidates.sort(key=lambda row: row["joint_candidate_score"], reverse=True)
    save_csv(os.path.join(config.output_dir, "all_ray_scores.csv"), rows)
    save_csv(os.path.join(config.output_dir, "joint_candidates.csv"), candidates)
    candidate_dir = os.path.join(config.output_dir, "candidates")
    os.makedirs(candidate_dir, exist_ok=True)
    for candidate in candidates:
        ray_name = candidate["ray"]
        save_csv(
            os.path.join(candidate_dir, f"{ray_name}.csv"),
            per_sample_by_ray[ray_name],
        )
    save_scatter(
        os.path.join(config.output_dir, "smile_luminance_scatter.png"),
        rows,
        smile_threshold,
        luminance_threshold,
    )
    save_json(
        os.path.join(config.output_dir, "scan_manifest.json"),
        {
            "config": config.to_dict(),
            "smile_reference": smile_metadata,
            "smile_response_definition": (
                "Cosine to direct SeFa Smile central image response after "
                "per-channel spatial-mean removal, measured in aligned mouth crop."
            ),
            "luminance_response_definition": (
                "Mean Rec.709 luminance derivative of the central image response."
            ),
            "threshold_rule": (
                "A joint candidate must be at or above the configured quantile "
                "for both absolute mean mouth-Smile alignment and mean absolute "
                "luminance slope."
            ),
            "smile_threshold": smile_threshold,
            "luminance_threshold": luminance_threshold,
            "scanned_rays": len(rows),
            "joint_candidates": len(candidates),
            "accepted_source_candidates": sum(
                row["source_status"] == "accepted" for row in candidates
            ),
            "skipped_source_candidates": sum(
                row["source_status"] == "skipped" for row in candidates
            ),
            "runtime_seconds": time.perf_counter() - started,
        },
    )
    print(f"Joint candidates: {len(candidates)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()

