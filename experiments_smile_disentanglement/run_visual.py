"""Visual-first local RO search around the F-SeFa Smile direction.

This runner intentionally keeps the first experiment small: one locality cap,
eight candidates, a signed Smile response, and image grids as the primary
deliverable.  Every candidate is checkpointed before the next search starts.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP = os.path.dirname(__file__)
sys.path.extend([ROOT, os.path.join(ROOT, "experiments_ro_verification")])

from config import Config
from evaluation import save_csv, save_json, save_traversal_grid
from optimizer import optimize_sphere
from run import Geo, LocalObjective, anchors, dirs, evaluator, get_response, ih, template
from utils import factorize_weight, load_generator


TAU = 0.90
NUM_CANDIDATES = 8
SEARCH_IDENTITIES = 8
VALIDATION_IDENTITIES = 8
VISUAL_IDENTITIES = 8
MAX_ITERATIONS = 4
MULTIPLIERS = (-3, -2, -1, 0, 1, 2, 3)
CONTACT_MULTIPLIERS = (-2, -1, 0, 1, 2)


def _raw_sefa_smile(generator):
    """Return the original Euclidean-unit SeFa Smile eigenvector."""
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    order = np.argsort(np.real(eigenvalues).astype(np.float64))[::-1]
    vector = boundaries[int(order[18])].reshape(-1)
    return vector / max(np.linalg.norm(vector), 1e-30)


def _write_progress(path, rows):
    save_csv(path, rows)


def _save_candidate(path, direction, gram, metrics):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(os.path.splitext(path)[0] + ".npy", np.asarray(direction, dtype=np.float64))
    np.savez_compressed(
        path,
        direction=np.asarray(direction, dtype=np.float64),
        gram_diagonal=np.asarray(gram, dtype=np.float64),
        metrics=np.asarray([metrics], dtype=object),
    )


def _render_rows(renderer, coefficient, rows, distances):
    result = []
    device = renderer.device
    zero_columns = [index for index, value in enumerate(distances) if abs(value) < 1e-12]
    center_image = None
    for label, direction in rows:
        tensor = torch.as_tensor(direction, dtype=torch.float32, device=device)
        images = renderer.render_traversal(tensor, coefficient, distances)
        if zero_columns:
            if center_image is None:
                center_image = images[zero_columns[0]].copy()
            for index in zero_columns:
                images[index] = center_image.copy()
        result.append((label, images))
    return result


def main():
    cfg = Config()
    start = time.perf_counter()
    output = cfg.output_dir
    directions_dir = os.path.join(output, "directions", "visual_tau090")
    figures_dir = os.path.join(output, "figures")
    metrics_dir = os.path.join(output, "metrics")
    for path in (directions_dir, figures_dir, metrics_dir):
        os.makedirs(path, exist_ok=True)

    log_path = os.path.join(output, "visual_run.log")

    def log(message):
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("Loading generator")
    generator = load_generator(cfg.model)
    device = next(generator.parameters()).device
    semantic_directions, direction_meta, gram = dirs(generator)
    gram = np.asarray(gram, dtype=np.float64)
    fsefa_smile = semantic_directions["smile"]
    sefa_smile = _raw_sefa_smile(generator)
    q = torch.diag(torch.as_tensor(1.0 / np.sqrt(gram), dtype=torch.float32, device=device))
    x0 = np.sqrt(gram) * fsefa_smile
    geometry = Geo(q.detach().cpu().double().numpy(), gram)

    np.save(os.path.join(directions_dir, "sefa_smile.npy"), sefa_smile)
    np.save(os.path.join(directions_dir, "fsefa_smile.npy"), fsefa_smile)

    total = SEARCH_IDENTITIES + VALIDATION_IDENTITIES + VISUAL_IDENTITIES
    latent = anchors(generator, total, cfg.seed + 9000)
    search_w = latent[:SEARCH_IDENTITIES]
    validation_w = latent[SEARCH_IDENTITIES : SEARCH_IDENTITIES + VALIDATION_IDENTITIES]
    visual_w = latent[-VISUAL_IDENTITIES:]

    base = evaluator(generator, q, cfg.epsilon_h, cfg.resolution, cfg.microbatch)
    search_anchors = base.prepare_anchors("search", search_w)
    validation_anchors = base.prepare_anchors("validation", validation_w)

    log("Building signed semantic response templates")
    templates = {
        name: template(get_response(generator, direction, search_anchors, cfg))
        for name, direction in semantic_directions.items()
    }
    scales = {}
    for name, direction in semantic_directions.items():
        response = get_response(generator, direction, search_anchors, cfg).reshape(
            SEARCH_IDENTITIES, -1
        )
        signed = np.sum(response * templates[name], axis=1) / math.sqrt(response.shape[1])
        scales[name] = max(float(np.mean(np.abs(signed))), 1e-8)
    scales["luminance"] = 1.0

    # beta=gamma=0 deliberately implements SmileResponse - locality penalty only.
    objective = LocalObjective(
        generator, q, cfg, templates, scales, x0, TAU, beta=0.0, gamma=0.0
    )
    rows = []
    candidates = []
    progress_path = os.path.join(metrics_dir, "visual_candidates.csv")

    for candidate_id in range(NUM_CANDIDATES):
        rng = np.random.default_rng(cfg.seed + 12000 + candidate_id)
        tangent = torch.as_tensor(rng.normal(size=cfg.dimension), dtype=torch.float32, device=device)
        center = torch.as_tensor(x0, dtype=torch.float32, device=device)
        tangent = tangent - center * torch.dot(center, tangent)
        tangent = tangent / torch.linalg.vector_norm(tangent)
        alpha = (0.02, 0.05, 0.10, 0.15)[candidate_id % 4]
        initial = math.cos(alpha) * center + math.sin(alpha) * tangent

        optimized, history, runtime = optimize_sphere(
            objective,
            search_anchors,
            initial,
            geometry,
            cfg.learning_rate,
            cfg.min_learning_rate,
            1e-4,
            cfg.line_search_steps,
            MAX_ITERATIONS,
            cfg.gradient_tolerance,
        )
        # Enforce the spherical-cap constraint exactly.  A soft penalty guides
        # the search, while this retraction keeps the final point feasible.
        cap_target = TAU + 1e-4
        raw_cos = float(torch.dot(optimized, center).detach().cpu())
        projected_to_cap = raw_cos < cap_target
        if projected_to_cap:
            cap_tangent = optimized - torch.dot(optimized, center) * center
            cap_tangent = cap_tangent / torch.linalg.vector_norm(cap_tangent)
            optimized = (
                cap_target * center
                + math.sqrt(1.0 - cap_target**2) * cap_tangent
            )
        direction = (q @ optimized).detach().cpu().double().numpy()
        validation, _ = objective.evaluate(optimized, validation_anchors, False)
        flipped = False
        if validation["smile_mean"] < 0:
            direction = -direction
            flipped = True
        cos_h = ih(direction, fsefa_smile, gram)
        norm_error = abs(ih(direction, direction, gram) - 1.0)
        accepted = cos_h >= TAU and norm_error <= 1e-5
        metrics = {
            "candidate_id": candidate_id + 1,
            "accepted": accepted,
            "flipped_for_positive_smile": flipped,
            "raw_cos_h_before_cap": raw_cos,
            "projected_to_locality_cap": projected_to_cap,
            "tau": TAU,
            "cos_h": cos_h,
            "angle_degrees": float(np.degrees(np.arccos(np.clip(cos_h, -1.0, 1.0)))),
            "hilbert_norm_error": norm_error,
            "smile_response": abs(float(validation["smile_mean"])),
            "search_objective": float(history[-1]["objective"]),
            "iterations": len(history) - 1,
            "runtime_seconds": runtime,
            "status": "accepted" if accepted else "failed_locality_or_norm",
        }
        rows.append(metrics)
        candidates.append((candidate_id + 1, direction, metrics))
        _save_candidate(
            os.path.join(directions_dir, f"ro_candidate_{candidate_id + 1:02d}.npz"),
            direction,
            gram,
            metrics,
        )
        _write_progress(progress_path, rows)
        log(
            f"Candidate {candidate_id + 1:02d}/{NUM_CANDIDATES}: "
            f"{metrics['status']}, cos_H={cos_h:.4f}, "
            f"smile={metrics['smile_response']:.4f}"
        )

    # Rendering uses identity coordinates so directions are interpreted directly in W.
    renderer = evaluator(
        generator,
        torch.eye(cfg.dimension, dtype=torch.float32, device=device),
        cfg.epsilon_h,
        cfg.resolution,
        cfg.microbatch,
    )
    h_distances = [cfg.epsilon_h * value for value in MULTIPLIERS]
    contact_distances = [cfg.epsilon_h * value for value in CONTACT_MULTIPLIERS]
    all_methods = [("SeFa Smile", sefa_smile), ("F-SeFa Smile", fsefa_smile)] + [
        (f"RO Candidate {candidate_id:02d}", direction)
        for candidate_id, direction, _ in candidates
    ]

    log("Rendering per-identity comparison figures")
    for identity, coefficient in enumerate(visual_w):
        rendered = _render_rows(renderer, coefficient, all_methods, h_distances)
        save_traversal_grid(
            os.path.join(figures_dir, f"id_{identity:03d}_grid.png"),
            rendered,
            [float(value) for value in MULTIPLIERS],
            thumb=128,
        )

    contact_methods = [("F-SeFa Smile", fsefa_smile)] + all_methods[2:]
    contact = _render_rows(renderer, visual_w[0], contact_methods, contact_distances)
    save_traversal_grid(
        os.path.join(figures_dir, "candidate_contact_sheet.png"),
        contact,
        [float(value) for value in CONTACT_MULTIPLIERS],
        thumb=144,
    )

    overview_rows = []
    overview_distances = [-2.0 * cfg.epsilon_h, 0.0, 2.0 * cfg.epsilon_h]
    accepted_candidates = [item for item in candidates if item[2]["accepted"]]
    representatives = (accepted_candidates or candidates)[:3]
    for identity, coefficient in enumerate(visual_w):
        overview_methods = [(f"id {identity:03d} / F-SeFa", fsefa_smile)] + [
            (f"id {identity:03d} / RO {item[0]:02d}", item[1])
            for item in representatives
        ]
        overview_rows.extend(
            _render_rows(renderer, coefficient, overview_methods, overview_distances)
        )
    save_traversal_grid(
        os.path.join(figures_dir, "overview.png"),
        overview_rows,
        [-2.0, 0.0, 2.0],
        thumb=112,
    )

    elapsed = time.perf_counter() - start
    manifest = {
        "experiment": "visual-first-local-smile-disentanglement",
        "tau": TAU,
        "num_candidates": NUM_CANDIDATES,
        "search_identities": SEARCH_IDENTITIES,
        "validation_identities": VALIDATION_IDENTITIES,
        "visual_identities": VISUAL_IDENTITIES,
        "max_iterations": MAX_ITERATIONS,
        "epsilon_h": cfg.epsilon_h,
        "accepted_candidates": [row["candidate_id"] for row in rows if row["accepted"]],
        "failed_candidates": [row["candidate_id"] for row in rows if not row["accepted"]],
        "direction_metadata": direction_meta,
        "runtime_seconds": elapsed,
    }
    save_json(os.path.join(output, "visual_manifest.json"), manifest)
    with open(os.path.join(output, "SUMMARY.md"), "w", encoding="utf-8") as handle:
        handle.write(
            "# Smile Disentanglement — Visual-first Run\n\n"
            f"- Locality cap: `cos_H >= {TAU}`\n"
            f"- Candidates: `{NUM_CANDIDATES}` (accepted: `{len(manifest['accepted_candidates'])}`)\n"
            f"- Optimization: signed Smile response minus soft locality penalty; nuisance penalties disabled.\n"
            f"- Search/validation/visual identities: `{SEARCH_IDENTITIES}/{VALIDATION_IDENTITIES}/{VISUAL_IDENTITIES}`\n"
            f"- RO iterations per candidate: `{MAX_ITERATIONS}`\n"
            "- Main visual range: `-3H ... +3H`, where one H step is 0.1 in the Hilbert-normalized rows.\n"
            "- SeFa is retained as a qualitative raw Euclidean reference; its numeric scale is not Hilbert calibrated.\n\n"
            "Primary artifacts are in `outputs/figures/`; all candidate vectors, including failures, are in "
            "`outputs/directions/visual_tau090/`.\n"
        )
    log(f"Completed visual-first run in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
