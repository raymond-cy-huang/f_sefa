"""Paired hard/soft semantic-exclusion exploration on the Hilbert sphere."""
from __future__ import annotations

import math
import os
import shutil
import sys
import time

import numpy as np
import torch

EXP = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(EXP, ".."))
VERIFICATION = os.path.join(ROOT, "experiments_ro_verification")
for path in (ROOT, VERIFICATION):
    if path not in sys.path:
        sys.path.append(path)

from config import SEMANTIC_RANKS
from evaluation import save_csv, save_json, save_traversal_grid
from geometry import HilbertComplement, hilbert_inner, legendre_gram_diagonal
from objective import GANImageObjective
from optimizer import optimize_sphere
from utils import factorize_weight, load_generator


NUM_SEEDS = 8
NUM_IDENTITIES = 4
NUM_SEARCH = 8
MAX_ITERATIONS = 6
RESOLUTION = 64
EPSILON_H = 0.1
CURVATURE_BETA = 0.1
LEARNING_RATE = 0.25
MIN_LEARNING_RATE = 1e-10
LINE_SEARCH_STEPS = 20
GRADIENT_TOLERANCE = 1e-6
MICROBATCH = 1
RANDOM_SEED = 20261006
SEMANTICS = ("gender", "age", "pose", "glasses", "smile")
REGIMES = (("hard", None), ("lambda_10", 10.0), ("lambda_1", 1.0),
           ("lambda_0p1", 0.1), ("lambda_0", 0.0))
TRAVERSAL = (-3, -2, -1, 0, 1, 2, 3)


class FullSphereGeometry:
    def __init__(self, gram):
        self.gram_diag = np.asarray(gram, dtype=np.float64)

    def constraint_errors(self, direction):
        value = hilbert_inner(direction, direction, self.gram_diag)
        return {
            "orthogonality_l2": 0.0,
            "orthogonality_max_abs": 0.0,
            "hilbert_norm_squared": value,
            "hilbert_norm_error": abs(value - 1.0),
        }


class SoftOverlapObjective(GANImageObjective):
    """The verified image objective minus lambda times known overlap."""

    def __init__(self, *args, overlap_matrix, overlap_lambda, **kwargs):
        super().__init__(*args, **kwargs)
        self.overlap_matrix = torch.as_tensor(
            overlap_matrix, dtype=torch.float32, device=self.device
        )
        self.overlap_lambda = float(overlap_lambda)

    def _overlap(self, point):
        coordinates = self.overlap_matrix @ point
        return torch.sum(coordinates.square()), coordinates

    def value_and_gradient(self, point, anchors):
        metrics, base_gradient = super().value_and_gradient(point, anchors)
        point = point.detach().to(self.device, dtype=torch.float32)
        overlap, coordinates = self._overlap(point)
        overlap_gradient = 2.0 * self.overlap_matrix.T @ coordinates
        total = float(overlap.detach().cpu())
        metrics["image_response"] = metrics["objective"]
        metrics["total_known_overlap"] = total
        metrics["objective"] = metrics["image_response"] - self.overlap_lambda * total
        return metrics, base_gradient - self.overlap_lambda * overlap_gradient

    @torch.no_grad()
    def evaluate(self, point, anchors, collect_responses=False):
        metrics, responses = super().evaluate(point, anchors, collect_responses)
        point = point.detach().to(self.device, dtype=torch.float32)
        overlap, _ = self._overlap(point)
        total = float(overlap.cpu())
        metrics["image_response"] = metrics["objective"]
        metrics["total_known_overlap"] = total
        metrics["objective"] = metrics["image_response"] - self.overlap_lambda * total
        return metrics, responses


def sample_w(generator, count):
    device = next(generator.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(RANDOM_SEED + 101)
    z = torch.randn(count, generator.z_space_dim, generator=rng, device=device)
    with torch.no_grad():
        w = generator.truncation(generator.mapping(z)["w"], trunc_psi=0.7, trunc_layers=8)
    if w.ndim == 3:
        w = w[:, 0, :]
    return w.detach().cpu().float()


def known_directions(generator, gram):
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    columns, metadata = [], []
    for name in SEMANTICS:
        index = int(order[SEMANTIC_RANKS[name]])
        direction = boundaries[index].reshape(-1)
        direction /= math.sqrt(max(hilbert_inner(direction, direction, gram), 1e-30))
        columns.append(direction)
        metadata.append({
            "name": name,
            "rank": int(SEMANTIC_RANKS[name]),
            "source_index": index,
            "eigenvalue": float(eigenvalues[index]),
        })
    return np.column_stack(columns), metadata


def overlaps(direction, known, gram):
    values = known.T @ (gram * np.asarray(direction, dtype=np.float64))
    result = {f"overlap_{name}": float(value) for name, value in zip(SEMANTICS, values)}
    result["total_known_overlap"] = float(np.sum(values ** 2))
    return result


def render_rows(renderer, coefficient, methods, distances):
    result, center = [], None
    zero = [index for index, value in enumerate(distances) if abs(value) < 1e-12]
    for label, direction in methods:
        point = torch.as_tensor(direction, dtype=torch.float32, device=renderer.device)
        images = renderer.render_traversal(point, coefficient, distances)
        if zero:
            if center is None:
                center = images[zero[0]].copy()
            for index in zero:
                images[index] = center.copy()
        result.append((label, images))
    return result


def save_plots(rows, figures_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"hard": "black", "lambda_10": "#3454D1", "lambda_1": "#2A9D8F",
              "lambda_0p1": "#F4A261", "lambda_0": "#E63946"}
    labels = {"hard": "Hard", "lambda_10": "10", "lambda_1": "1",
              "lambda_0p1": "0.1", "lambda_0": "0"}
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for regime, _ in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        axis.scatter([row["total_known_overlap"] for row in subset],
                     [row["image_response"] for row in subset],
                     label=labels[regime], color=colors[regime], s=48, alpha=.85)
    axis.set_xlabel("Total known-semantic overlap $O_{known}(v)$")
    axis.set_ylabel("Image response $R_{image}(v)$")
    axis.set_title("Known-semantic overlap vs. image response")
    axis.grid(alpha=.25); axis.legend(title="Exclusion")
    figure.tight_layout(); figure.savefig(os.path.join(figures_dir, "overlap_vs_response.png"), dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    for x, (regime, _) in enumerate(REGIMES):
        values = [row["total_known_overlap"] for row in rows if row["regime"] == regime]
        jitter = np.linspace(-.08, .08, len(values))
        axis.scatter(x + jitter, values, color=colors[regime], s=44, alpha=.85)
        axis.plot([x - .18, x + .18], [np.mean(values)] * 2, color="black", linewidth=2)
    axis.set_xticks(range(len(REGIMES)), ["Hard", "10", "1", "0.1", "0"])
    axis.set_xlabel("$\\lambda_{overlap}$ (Hard is exact constraint)")
    axis.set_ylabel("Total known-semantic overlap $O_{known}(v)$")
    axis.set_title("Exclusion pressure controls semantic overlap")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout(); figure.savefig(os.path.join(figures_dir, "lambda_vs_overlap.png"), dpi=180)
    plt.close(figure)


def main():
    output = os.path.join(EXP, "outputs")
    figures = os.path.join(output, "figures")
    directions = os.path.join(output, "directions")
    metrics_dir = os.path.join(output, "metrics")
    for path in (figures, directions, metrics_dir):
        os.makedirs(path, exist_ok=True)
    log_path = os.path.join(output, "run.log")
    started = time.perf_counter()

    def log(message):
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    log("Loading generator")
    generator = load_generator("e4e_ffhq_encode")
    device = next(generator.parameters()).device
    dimension = int(generator.w_space_dim)
    gram = legendre_gram_diagonal(dimension)
    known, semantic_metadata = known_directions(generator, gram)
    hard_geometry = HilbertComplement(known, gram)
    full_geometry = FullSphereGeometry(gram)
    hard_q = hard_geometry.torch_q(device)
    full_q = torch.diag(torch.as_tensor(1.0 / np.sqrt(gram), dtype=torch.float32, device=device))
    overlap_matrix = known.T @ (gram[:, None] * full_q.detach().cpu().double().numpy())

    latent = sample_w(generator, NUM_SEARCH + NUM_IDENTITIES)
    anchor_builder = GANImageObjective(generator, full_q, EPSILON_H, CURVATURE_BETA, RESOLUTION, MICROBATCH)
    search_anchors = anchor_builder.prepare_anchors("search", latent[:NUM_SEARCH])
    identities = latent[NUM_SEARCH:]
    np.save(os.path.join(output, "search_anchors.npy"), search_anchors.coefficients.numpy())
    np.save(os.path.join(output, "visual_identities.npy"), identities.numpy())

    rng = np.random.default_rng(RANDOM_SEED + 3000)
    shared_initial_directions = []
    for _ in range(NUM_SEEDS):
        shared_initial_directions.append(hard_geometry.project_and_normalize(rng.normal(size=dimension)))
    np.save(os.path.join(output, "shared_initial_directions.npy"), np.stack(shared_initial_directions))

    rows, optimized = [], {}
    progress = os.path.join(metrics_dir, "all_candidates.csv")
    for seed, initial_direction in enumerate(shared_initial_directions):
        for regime, penalty in REGIMES:
            if regime == "hard":
                q, geometry = hard_q, hard_geometry
                initial = hard_q.T @ (
                    torch.as_tensor(gram, dtype=torch.float32, device=device)
                    * torch.as_tensor(initial_direction, dtype=torch.float32, device=device)
                )
                objective = GANImageObjective(
                    generator, q, EPSILON_H, CURVATURE_BETA, RESOLUTION, MICROBATCH
                )
            else:
                q, geometry = full_q, full_geometry
                initial = torch.as_tensor(
                    np.sqrt(gram) * initial_direction, dtype=torch.float32, device=device
                )
                objective = SoftOverlapObjective(
                    generator, q, EPSILON_H, CURVATURE_BETA, RESOLUTION, MICROBATCH,
                    overlap_matrix=overlap_matrix, overlap_lambda=penalty,
                )
            point, history, runtime = optimize_sphere(
                objective, search_anchors, initial, geometry, LEARNING_RATE,
                MIN_LEARNING_RATE, 1e-4, LINE_SEARCH_STEPS, MAX_ITERATIONS,
                GRADIENT_TOLERANCE,
            )
            direction = (q @ point).detach().cpu().double().numpy()
            base_eval = GANImageObjective(
                generator,
                torch.as_tensor(direction[:, None], dtype=torch.float32, device=device),
                EPSILON_H, CURVATURE_BETA, RESOLUTION, MICROBATCH,
            )
            unit = torch.ones(1, dtype=torch.float32, device=device)
            image_metrics, _ = base_eval.evaluate(unit, search_anchors, False)
            overlap_metrics = overlaps(direction, known, gram)
            total_overlap = overlap_metrics["total_known_overlap"]
            effective_lambda = 0.0 if penalty is None else penalty
            row = {
                "seed": seed,
                "regime": regime,
                "lambda_overlap": "HARD" if penalty is None else penalty,
                "objective": float(image_metrics["objective"] - effective_lambda * total_overlap),
                "image_response": float(image_metrics["objective"]),
                "effect": float(image_metrics["effect"]),
                "curvature": float(image_metrics["curvature"]),
                **overlap_metrics,
                "hilbert_norm_error": abs(hilbert_inner(direction, direction, gram) - 1.0),
                "iterations": len(history) - 1,
                "last_accepted_step": next(
                    (float(item["accepted_step_size"]) for item in reversed(history)
                     if float(item.get("accepted_step_size", 0.0)) > 0.0),
                    0.0,
                ),
                "runtime_seconds": runtime,
            }
            rows.append(row); optimized[(seed, regime)] = direction
            base_name = f"seed_{seed:03d}_{regime}"
            np.save(os.path.join(directions, base_name + ".npy"), direction)
            np.savez_compressed(os.path.join(directions, base_name + ".npz"),
                                direction=direction, gram_diagonal=gram,
                                initial_direction=initial_direction)
            save_csv(progress, rows)
            last_step = row["last_accepted_step"]
            stop_reason = history[-1].get("stop_reason", "max_iterations_or_tolerance")
            log(f"seed {seed:03d} {regime}: iterations={row['iterations']}, "
                f"last_step={last_step:.3e}, stop={stop_reason}, "
                f"overlap={total_overlap:.6e}, response={row['image_response']:.6e}")

    renderer = GANImageObjective(
        generator, torch.eye(dimension, dtype=torch.float32, device=device),
        EPSILON_H, CURVATURE_BETA, RESOLUTION, MICROBATCH,
    )
    traversal_distances = [EPSILON_H * value for value in TRAVERSAL]
    contact_distances = [-2 * EPSILON_H, 0.0, 2 * EPSILON_H]
    display_labels = {"hard": "Hard Orthogonal", "lambda_10": "lambda = 10",
                      "lambda_1": "lambda = 1", "lambda_0p1": "lambda = 0.1",
                      "lambda_0": "lambda = 0"}
    log("Rendering image grids")
    for seed in range(NUM_SEEDS):
        methods = [(display_labels[regime], optimized[(seed, regime)]) for regime, _ in REGIMES]
        for identity, coefficient in enumerate(identities):
            rendered = render_rows(renderer, coefficient, methods, traversal_distances)
            path = os.path.join(figures, f"seed_{seed:03d}_id_{identity:03d}.png")
            save_traversal_grid(path, rendered, [float(x) for x in TRAVERSAL], thumb=128)
            if identity == 0:
                shutil.copyfile(path, os.path.join(figures, f"seed_{seed:03d}_grid.png"))
        contact_rows = []
        for identity, coefficient in enumerate(identities):
            identity_methods = [(f"id {identity:03d} / {label}", direction) for label, direction in methods]
            contact_rows.extend(render_rows(renderer, coefficient, identity_methods, contact_distances))
        save_traversal_grid(
            os.path.join(figures, f"seed_{seed:03d}_contact_sheet.png"),
            contact_rows, [-2.0, 0.0, 2.0], thumb=112,
        )

    save_plots(rows, figures)
    means = {}
    for regime, penalty in REGIMES:
        subset = [row for row in rows if row["regime"] == regime]
        means[regime] = {
            "lambda_overlap": "HARD" if penalty is None else penalty,
            "mean_known_overlap": float(np.mean([row["total_known_overlap"] for row in subset])),
            "mean_image_response": float(np.mean([row["image_response"] for row in subset])),
        }
    elapsed = time.perf_counter() - started
    save_json(os.path.join(output, "manifest.json"), {
        "experiment": "soft-exclusion-exploration",
        "num_seeds": NUM_SEEDS, "num_identities": NUM_IDENTITIES,
        "num_search": NUM_SEARCH, "max_iterations": MAX_ITERATIONS,
        "resolution": RESOLUTION, "epsilon_h": EPSILON_H,
        "curvature_beta": CURVATURE_BETA, "learning_rate": LEARNING_RATE,
        "regimes": [{"name": name, "lambda": value} for name, value in REGIMES],
        "semantic_metadata": semantic_metadata,
        "hard_geometry": hard_geometry.diagnostics.__dict__,
        "means": means, "runtime_seconds": elapsed,
    })
    with open(os.path.join(output, "SUMMARY.md"), "w", encoding="utf-8") as handle:
        handle.write("# Soft Exclusion Exploration\n\n")
        handle.write(f"- Seeds: `{NUM_SEEDS}` (paired initialization across all regimes)\n")
        handle.write(f"- Visual identities: `{NUM_IDENTITIES}`\n")
        handle.write("- Regimes: exact Hard, lambda = 10, 1, 0.1, 0\n")
        handle.write(f"- Optimization: `{MAX_ITERATIONS}` iterations, `{NUM_SEARCH}` anchors, "
                     f"resolution `{RESOLUTION}`, epsilon_H `{EPSILON_H}`, curvature beta `{CURVATURE_BETA}`\n\n")
        handle.write("## Numerical summary\n\n")
        handle.write("| Regime | Mean known overlap | Mean image response |\n|---|---:|---:|\n")
        for regime, _ in REGIMES:
            item = means[regime]
            handle.write(f"| {regime} | {item['mean_known_overlap']:.6e} | {item['mean_image_response']:.6e} |\n")
        handle.write("\n## Qualitative review\n\n")
        handle.write("Manual categories to record from the contact sheets: known-like, mixed, fine-grained, "
                     "illumination, structural, unclear, and potentially novel. No candidate is automatically "
                     "claimed as a new semantic.\n\n")
        handle.write("## Limitations\n\nThis first run uses 8 optimization anchors at 64 px and six RO updates. "
                     "The grids support qualitative screening, not final semantic or statistical claims. "
                     "Classifier-based labels were intentionally excluded.\n")
    log(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
