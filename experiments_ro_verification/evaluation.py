"""Evaluation and artifact helpers for Random-vs-RO experiments."""

from __future__ import annotations

import csv
import json
import os

import numpy as np


def response_consistency(responses: np.ndarray, eps: float = 1e-12):
    responses = np.asarray(responses, dtype=np.float64)
    if responses.ndim != 2 or len(responses) < 2:
        raise ValueError("responses must have shape [N, features] with N >= 2.")
    norms = np.linalg.norm(responses, axis=1, keepdims=True)
    normalized = responses / np.maximum(norms, eps)
    cosine = np.clip(normalized @ normalized.T, -1.0, 1.0)
    off_diagonal = cosine[~np.eye(len(cosine), dtype=bool)]
    centered = responses - responses.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    variance = singular_values**2
    explained_pc1 = float(variance[0] / max(float(variance.sum()), eps))
    return {
        "mean_pairwise_response_cosine": float(off_diagonal.mean()),
        "median_pairwise_response_cosine": float(np.median(off_diagonal)),
        "min_pairwise_response_cosine": float(off_diagonal.min()),
        "response_norm_mean": float(norms.mean()),
        "response_norm_std": float(norms.std()),
        "response_pc1_explained_variance": explained_pc1,
    }, cosine, singular_values


def save_csv(path: str, rows: list[dict], fieldnames: list[str] | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not rows:
        raise ValueError("Cannot save an empty CSV.")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def save_json(path: str, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)


def save_convergence_plot(path: str, history: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [row["iteration"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.7))
    axes[0].plot(iterations, [row["objective"] for row in history])
    axes[0].set_title("Search objective J")
    axes[1].plot(iterations, [row["effect"] for row in history], label="E")
    axes[1].plot(iterations, [row["curvature"] for row in history], label="C")
    axes[1].set_title("Effect and curvature")
    axes[1].legend()
    axes[2].semilogy(
        iterations,
        np.maximum([row["riemannian_gradient_norm"] for row in history], 1e-16),
    )
    axes[2].set_title("Riemannian gradient norm")
    for axis in axes:
        axis.set_xlabel("Iteration")
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_heatmap(path: str, matrix: np.ndarray, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6.2, 5.4))
    image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    axis.set_title(title)
    axis.set_xlabel("Held-out sample")
    axis.set_ylabel("Held-out sample")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_paired_plot(path: str, rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    random_values = np.array([row["random_test_objective"] for row in rows])
    ro_values = np.array([row["ro_test_objective"] for row in rows])
    figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.0))
    for left, right in zip(random_values, ro_values):
        axes[0].plot([0, 1], [left, right], marker="o", alpha=0.7)
    axes[0].set_xticks([0, 1], ["Random", "RO"])
    axes[0].set_title("Held-out paired objective")
    axes[0].grid(alpha=0.25)
    differences = ro_values - random_values
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].bar(np.arange(len(differences)), differences)
    axes[1].set_title("Paired improvement")
    axes[1].set_xlabel("Restart")
    axes[1].set_ylabel("RO - Random")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_traversal_grid(path, rows, distances, thumb=160):
    """Rows are [(label, [uint8 RGB images...]), ...]."""
    from PIL import Image, ImageDraw

    label_width = 155
    header_height = 32
    canvas = Image.new(
        "RGB",
        (label_width + len(distances) * thumb, header_height + len(rows) * thumb),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, distance in enumerate(distances):
        draw.text(
            (label_width + column * thumb + 5, 8),
            f"d_H={distance:+.3g}",
            fill="black",
        )
    for row_index, (label, images) in enumerate(rows):
        top = header_height + row_index * thumb
        draw.text((6, top + thumb // 2), label, fill="black")
        for column, array in enumerate(images):
            image = Image.fromarray(np.asarray(array, dtype=np.uint8)).resize(
                (thumb, thumb), Image.Resampling.LANCZOS
            )
            canvas.paste(image, (label_width + column * thumb, top))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    canvas.save(path)


def bootstrap_mean_interval(values, seed=0, draws=10000, alpha=0.05):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "ci_low": float(np.quantile(means, alpha / 2.0)),
        "ci_high": float(np.quantile(means, 1.0 - alpha / 2.0)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }

