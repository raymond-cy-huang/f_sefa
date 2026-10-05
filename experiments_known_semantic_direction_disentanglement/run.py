"""Disentangle five known SeFa directions using existing complete-orthogonal rays."""

from __future__ import annotations

import csv
import os
import sys
import time

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
VERIFICATION_DIR = os.path.join(ROOT_DIR, "experiments_ro_verification")
for path in (EXPERIMENT_DIR, ROOT_DIR, VERIFICATION_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import Config
from evaluation import save_csv, save_json, save_traversal_grid
from objective import GANImageObjective
from utils import factorize_weight, load_generator


SEMANTIC_RANKS = {
    "smile": 18,
    "gender": 1,
    "age": 3,
    "glasses": 10,
    "pose": 7,
}


def normalize_h(direction, gram):
    norm = np.sqrt(np.dot(direction * gram, direction))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a near-zero Hilbert direction.")
    return direction / norm


def inner_h(left, right, gram):
    return float(np.dot(left * gram, right))


def load_ray(source_dir, ray_name):
    payload = np.load(
        os.path.join(source_dir, ray_name, "direction.npz"), allow_pickle=False
    )
    return (
        np.asarray(payload["optimized_function_direction"], dtype=np.float64),
        np.asarray(payload["gram_diagonal"], dtype=np.float64),
    )


def response(generator, direction, anchor_set, config):
    import torch

    device = next(generator.parameters()).device
    evaluator = GANImageObjective(
        generator,
        torch.as_tensor(direction[:, None], dtype=torch.float32, device=device),
        config.epsilon_h,
        curvature_beta=0.0,
        resolution=config.resolution,
        microbatch=config.microbatch,
    )
    _, flat = evaluator.evaluate(
        torch.ones(1, dtype=torch.float32, device=device),
        anchor_set,
        collect_responses=True,
    )
    return flat.reshape(-1, 3, config.resolution, config.resolution).astype(np.float64)


def centered_flat(images):
    centered = images - images.mean(axis=(2, 3), keepdims=True)
    return centered.reshape(len(centered), -1)


def cosine_rows(left, right):
    numerator = np.sum(left * right, axis=1)
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    return numerator / np.maximum(denominator, 1e-12)


def amplitudes(flat_response, normalized_templates):
    return np.sum(flat_response * normalized_templates, axis=1) / np.sqrt(
        flat_response.shape[1]
    )


def normalized_powers(flat_response, templates, baseline_powers):
    result = {}
    for name, template in templates.items():
        power = float(np.mean(amplitudes(flat_response, template) ** 2))
        result[name] = power / max(baseline_powers[name], 1e-12)
    return result


def save_grid(path, named_directions, anchor_set, generator, config):
    import torch

    device = next(generator.parameters()).device
    distances = [config.epsilon_h * value for value in (-3, -2, -1, 0, 1, 2, 3)]
    rows = []
    for sample_index in range(min(4, len(anchor_set))):
        for name, direction in named_directions:
            renderer = GANImageObjective(
                generator,
                torch.as_tensor(direction[:, None], dtype=torch.float32, device=device),
                config.epsilon_h,
                curvature_beta=0.0,
                resolution=config.resolution,
                microbatch=config.microbatch,
            )
            images = renderer.render_traversal(
                torch.ones(1, dtype=torch.float32, device=device),
                anchor_set.coefficients[sample_index],
                distances,
            )
            rows.append((f"seed {sample_index:03d} / {name}", images))
    save_traversal_grid(path, rows, distances)


def main():
    config = Config()
    os.makedirs(config.output_dir, exist_ok=True)
    started = time.perf_counter()

    import torch

    generator = load_generator(config.model)
    device = next(generator.parameters()).device
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]

    first_ray = sorted(
        name for name in os.listdir(config.source_dir)
        if name.startswith("ray") and name[3:].isdigit()
    )[0]
    _, gram = load_ray(config.source_dir, first_ray)
    semantic_directions = {}
    semantic_metadata = {}
    for name, rank in SEMANTIC_RANKS.items():
        source_index = int(order[rank])
        semantic_directions[name] = normalize_h(
            boundaries[source_index].reshape(-1), gram
        )
        semantic_metadata[name] = {
            "eigen_rank": rank,
            "source_eigen_index": source_index,
            "eigenvalue": float(eigenvalues[source_index]),
        }

    anchors = torch.from_numpy(
        np.load(os.path.join(config.source_dir, "test_anchors.npy"))
    ).float()
    seed_direction = next(iter(semantic_directions.values()))
    seed_evaluator = GANImageObjective(
        generator,
        torch.as_tensor(seed_direction[:, None], dtype=torch.float32, device=device),
        config.epsilon_h,
        curvature_beta=0.0,
        resolution=config.resolution,
        microbatch=config.microbatch,
    )
    anchor_set = seed_evaluator.prepare_anchors("known_semantics", anchors)

    semantic_responses = {
        name: response(generator, direction, anchor_set, config)
        for name, direction in semantic_directions.items()
    }
    templates = {}
    baseline_powers = {}
    for name, images in semantic_responses.items():
        flat = centered_flat(images)
        templates[name] = flat / np.maximum(
            np.linalg.norm(flat, axis=1, keepdims=True), 1e-12
        )
        baseline_powers[name] = float(
            np.mean(amplitudes(flat, templates[name]) ** 2)
        )

    statuses = {}
    with open(
        os.path.join(config.source_dir, "complete_orth_summary.csv"),
        newline="", encoding="utf-8",
    ) as handle:
        statuses = {row["ray"]: row["status"] for row in csv.DictReader(handle)}

    ray_names = sorted(
        name for name in os.listdir(config.source_dir)
        if name.startswith("ray") and name[3:].isdigit()
    )
    scan_rows = []
    ray_cache = {}
    for index, ray_name in enumerate(ray_names, start=1):
        direction, ray_gram = load_ray(config.source_dir, ray_name)
        if not np.allclose(ray_gram, gram):
            raise ValueError(f"Gram mismatch for {ray_name}")
        direction = normalize_h(direction, gram)
        images = response(generator, direction, anchor_set, config)
        flat = centered_flat(images)
        ray_cache[ray_name] = (direction, flat)
        row = {"ray": ray_name, "source_status": statuses.get(ray_name, "unknown")}
        for semantic, template in templates.items():
            values = cosine_rows(flat, template)
            row[f"{semantic}_alignment_mean"] = float(np.mean(values))
            row[f"{semantic}_alignment_abs_mean"] = float(np.mean(np.abs(values)))
            row[f"{semantic}_hilbert_cosine"] = inner_h(
                direction, semantic_directions[semantic], gram
            )
        scan_rows.append(row)
        if index % 25 == 0 or index == len(ray_names):
            print(f"Scanned {index}/{len(ray_names)} rays", flush=True)

    save_csv(os.path.join(config.output_dir, "all_ray_semantic_scores.csv"), scan_rows)
    summary_rows = []
    for target in SEMANTIC_RANKS:
        semantic_dir = os.path.join(config.output_dir, target)
        os.makedirs(semantic_dir, exist_ok=True)
        key = f"{target}_alignment_abs_mean"
        threshold = float(np.quantile(
            np.asarray([row[key] for row in scan_rows]), config.candidate_quantile
        ))
        candidates = [row for row in scan_rows if row[key] >= threshold]
        candidate_results = []
        target_ref = semantic_directions[target]
        target_response = centered_flat(semantic_responses[target])
        for row in candidates:
            raw_direction, raw_flat = ray_cache[row["ray"]]
            coefficient = inner_h(raw_direction, target_ref, gram)
            projected = raw_direction - coefficient * target_ref
            projected_norm = np.sqrt(np.dot(projected * gram, projected))
            if projected_norm <= 1e-8:
                continue
            projected /= projected_norm
            projected_flat = (raw_flat - coefficient * target_response) / projected_norm
            powers = normalized_powers(projected_flat, templates, baseline_powers)
            nuisance = float(np.mean([
                value for name, value in powers.items() if name != target
            ]))
            score = powers[target] / (1.0 + nuisance)
            candidate_results.append({
                **row,
                "projection_coefficient": coefficient,
                "projected_hilbert_norm_before_normalization": projected_norm,
                "projected_target_power": powers[target],
                "projected_mean_nuisance_power": nuisance,
                "projected_selection_score": score,
            })
        candidate_results.sort(
            key=lambda row: row["projected_selection_score"], reverse=True
        )
        save_csv(os.path.join(semantic_dir, "candidates.csv"), candidate_results)
        if not candidate_results:
            raise RuntimeError(f"No usable candidate for {target}")

        selected = candidate_results[0]
        raw_partner, raw_partner_flat = ray_cache[selected["ray"]]
        coefficient = inner_h(raw_partner, target_ref, gram)
        partner = normalize_h(raw_partner - coefficient * target_ref, gram)
        partner_response = response(generator, partner, anchor_set, config)
        partner_flat = centered_flat(partner_response)
        if float(np.mean(cosine_rows(partner_flat, templates[target]))) < 0:
            partner = -partner
            partner_flat = -partner_flat

        angle_rows = []
        angles = np.linspace(-0.5 * np.pi, 0.5 * np.pi, config.angle_samples)
        for theta in angles:
            c, s = float(np.cos(theta)), float(np.sin(theta))
            mixed_flat = c * target_response + s * partner_flat
            powers = normalized_powers(mixed_flat, templates, baseline_powers)
            nuisance = float(np.mean([
                value for name, value in powers.items() if name != target
            ]))
            angle_rows.append({
                "theta_radians": theta,
                "theta_degrees": float(np.degrees(theta)),
                "target_power": powers[target],
                "mean_nuisance_power": nuisance,
                "objective": powers[target] - nuisance,
                **{f"response_{name}": value for name, value in powers.items()},
            })
        feasible = [
            row for row in angle_rows
            if row["target_power"] >= config.minimum_target_retention
        ]
        # Disentanglement is lexicographic: first retain the requested semantic,
        # then minimize leakage into the other four known semantics.  A gain in
        # target strength must not hide an increase in nuisance response.
        best = min(
            feasible or angle_rows,
            key=lambda row: (row["mean_nuisance_power"], -row["target_power"]),
        )
        theta = float(best["theta_radians"])
        c, s = np.cos(theta), np.sin(theta)
        disentangled = normalize_h(c * target_ref + s * partner, gram)
        complement = normalize_h(-s * target_ref + c * partner, gram)
        if inner_h(disentangled, target_ref, gram) < 0:
            disentangled = -disentangled
            complement = -complement

        final_rows = []
        named = (
            (f"SeFa-{target}", target_ref),
            (f"{selected['ray']}-projected", partner),
            (f"{target}-disentangled", disentangled),
            ("G-orthogonal-complement", complement),
        )
        for label, direction in named:
            flat = centered_flat(response(generator, direction, anchor_set, config))
            powers = normalized_powers(flat, templates, baseline_powers)
            final_rows.append({
                "direction": label,
                "target_semantic": target,
                "target_power": powers[target],
                "mean_nuisance_power": float(np.mean([
                    value for name, value in powers.items() if name != target
                ])),
                "hilbert_cosine_to_sefa_target": inner_h(direction, target_ref, gram),
                "hilbert_cosine_to_disentangled": inner_h(direction, disentangled, gram),
                **{f"response_{name}": value for name, value in powers.items()},
            })
        save_csv(os.path.join(semantic_dir, "rotation_curve.csv"), angle_rows)
        save_csv(os.path.join(semantic_dir, "metrics.csv"), final_rows)
        np.savez_compressed(
            os.path.join(semantic_dir, "directions.npz"),
            sefa_target=target_ref,
            selected_projected_partner=partner,
            disentangled=disentangled,
            orthogonal_complement=complement,
            gram_diagonal=gram,
        )
        save_grid(
            os.path.join(semantic_dir, "traversal.png"),
            named, anchor_set, generator, config,
        )
        final_cross = inner_h(disentangled, complement, gram)
        save_json(os.path.join(semantic_dir, "manifest.json"), {
            "semantic": target,
            **semantic_metadata[target],
            "candidate_threshold": threshold,
            "candidate_count": len(candidate_results),
            "selected_candidate": selected["ray"],
            "selected_rotation_degrees": float(np.degrees(theta)),
            "final_cross_hilbert_cosine": final_cross,
            "selection_rule": "Top-quartile image-response candidates; project each into the SeFa target G-orthogonal complement; select best target-to-known-nuisance ratio; rotate on S1.",
        })
        ref_metrics = final_rows[0]
        final_metrics = final_rows[2]
        summary_rows.append({
            "semantic": target,
            "eigen_rank": SEMANTIC_RANKS[target],
            "candidate_count": len(candidate_results),
            "selected_candidate": selected["ray"],
            "rotation_degrees": float(np.degrees(theta)),
            "sefa_target_power": ref_metrics["target_power"],
            "sefa_mean_nuisance_power": ref_metrics["mean_nuisance_power"],
            "final_target_power": final_metrics["target_power"],
            "final_mean_nuisance_power": final_metrics["mean_nuisance_power"],
            "final_cross_hilbert_cosine": final_cross,
        })
        print(f"Completed {target}: partner={selected['ray']}", flush=True)

    save_csv(os.path.join(config.output_dir, "summary.csv"), summary_rows)
    save_json(os.path.join(config.output_dir, "manifest.json"), {
        "experiment": "known-semantic-direction-disentanglement",
        "semantic_ranks": SEMANTIC_RANKS,
        "scanned_rays": len(scan_rows),
        "held_out_anchors": len(anchor_set),
        "candidate_quantile": config.candidate_quantile,
        "minimum_target_retention": config.minimum_target_retention,
        "runtime_seconds": time.perf_counter() - started,
    })
    print(f"Finished in {time.perf_counter() - started:.1f} seconds", flush=True)


if __name__ == "__main__":
    main()

