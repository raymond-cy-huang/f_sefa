"""Compare geometry-driven semantic direction proposal methods."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import pymanopt
import torch


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / "experiments_gdsa" / "03_unknown_semantic_discovery"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT))

from run import (  # noqa: E402
    KNOWN_RANKS,
    KNOWN_SEMANTICS,
    build_legendre_transform,
    chart_forward,
    chart_inverse,
    decode_uint8,
    finite_difference_matrix,
    inverse_chart_tangent,
    inverse_sqrt_spd,
    make_grid,
    optimize_stiefel,
    residualize,
    sample_w,
)
from utils import factorize_weight, load_generator  # noqa: E402


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="e4e_ffhq_encode")
    parser.add_argument("--search-rank-count", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[71, 89, 107])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--response-resolution", type=int, default=64)
    parser.add_argument("--direction-batch", type=int, default=2)
    parser.add_argument("--nuisance-weight", type=float, default=3.0)
    parser.add_argument("--common-sensitivity-weight", type=float, default=0.05)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[-4, -2, 0, 2, 4])
    parser.add_argument("--trunc-psi", type=float, default=0.7)
    parser.add_argument("--trunc-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--output-dir", default=str(HERE / "outputs"))
    return parser.parse_args()


def normalized_gram(matrix):
    gram = matrix.T @ matrix
    return gram * (len(gram) / max(float(np.trace(gram)), 1e-12))


def common_invariant_subspace(metrics, sensitivity, nuisance, count, args):
    dimension = sensitivity.shape[0]
    manifold = pymanopt.manifolds.Stiefel(dimension, count)
    metric_tensor = torch.as_tensor(np.stack(metrics), dtype=torch.float64)
    sensitivity_tensor = torch.as_tensor(sensitivity, dtype=torch.float64)
    nuisance_tensor = torch.as_tensor(nuisance, dtype=torch.float64)

    @pymanopt.function.pytorch(manifold)
    def cost(point):
        action = torch.einsum("ndj,jk->ndk", metric_tensor, point)
        projected = torch.einsum("dk,ndj->nkj", point, action)
        residual = action - torch.einsum("dk,nkj->ndj", point, projected)
        invariant_error = torch.mean(torch.sum(residual * residual, dim=(1, 2)))
        nuisance_cost = torch.trace(point.T @ nuisance_tensor @ point) / count
        sensitivity_gain = torch.trace(point.T @ sensitivity_tensor @ point) / count
        return (
            invariant_error
            + args.nuisance_weight * nuisance_cost
            - args.common_sensitivity_weight * sensitivity_gain
        )

    rng = np.random.default_rng(args.seed + 17)
    initial, _ = np.linalg.qr(rng.normal(size=(dimension, count)))
    optimizer = pymanopt.optimizers.ConjugateGradient(
        max_iterations=300, min_gradient_norm=1e-8, verbosity=1)
    result = optimizer.run(pymanopt.Problem(manifold, cost), initial_point=initial)
    subspace = np.asarray(result.point)
    restricted = subspace.T @ sensitivity @ subspace
    values, vectors = np.linalg.eigh((restricted + restricted.T) / 2.0)
    order = np.argsort(values)[::-1]
    return subspace @ vectors[:, order], values[order]


def method_coefficients(responses, nuisance_responses, args):
    dimension = responses[0].shape[1]
    metrics = [normalized_gram(matrix) for matrix in responses]
    energy = sum(metrics) / len(metrics)
    nuisance = sum(normalized_gram(matrix) for matrix in nuisance_responses) / len(
        nuisance_responses)
    identity = np.eye(dimension)

    nuisance_whitener = inverse_sqrt_spd(
        identity + args.nuisance_weight * nuisance + 1e-4 * identity)
    average_operator = nuisance_whitener @ energy @ nuisance_whitener
    average_u, _, _ = optimize_stiefel(
        (average_operator + average_operator.T) / 2,
        args.num_candidates,
        args.seed,
    )
    average = nuisance_whitener @ average_u

    mean_response = sum(responses) / len(responses)
    coherence = mean_response.T @ mean_response
    coherence *= dimension / max(float(np.trace(energy)), 1e-12)
    coherence_whitener = inverse_sqrt_spd(
        energy + args.nuisance_weight * nuisance + 1e-4 * identity)
    coherence_operator = coherence_whitener @ coherence @ coherence_whitener
    coherence_u, _, _ = optimize_stiefel(
        (coherence_operator + coherence_operator.T) / 2,
        args.num_candidates,
        args.seed + 1,
    )
    coherent = coherence_whitener @ coherence_u

    common, _ = common_invariant_subspace(
        metrics, energy, nuisance, args.num_candidates, args)
    return {
        "average_metric": average,
        "response_coherence": coherent,
        "common_invariant": common,
    }, metrics


def candidate_hilbert_directions(coefficients, search_h, train_points, transform, alpha):
    result = []
    for index in range(coefficients.shape[1]):
        q = search_h @ coefficients[:, index]
        norms = [
            np.linalg.norm(inverse_chart_tangent(point, q, transform, alpha))
            for point in train_points
        ]
        result.append(q / max(float(np.mean(norms)), 1e-12))
    return np.column_stack(result)


def main():
    args = arguments()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    generator = load_generator(args.model).eval()
    transform = build_legendre_transform(int(generator.w_space_dim))
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    ranked = np.real(boundaries)[np.argsort(np.real(eigenvalues))[::-1]]
    known_w, _ = np.linalg.qr(ranked[list(KNOWN_RANKS)].T)
    search_ranks = [
        rank for rank in range(args.search_rank_count) if rank not in KNOWN_RANKS
    ]
    search_w, _ = np.linalg.qr(ranked[search_ranks].T)
    train_points = sample_w(
        generator, args.train_seeds, args.trunc_psi, args.trunc_layers)
    mean_point = np.mean(train_points, axis=0)
    scales = np.cosh(args.alpha * (transform @ mean_point))
    search_h, _ = np.linalg.qr(scales[:, None] * transform @ search_w)

    responses = []
    nuisance_responses = []
    for sample_id, point in enumerate(train_points):
        local_search = np.column_stack([
            inverse_chart_tangent(point, search_h[:, j], transform, args.alpha)
            for j in range(search_h.shape[1])
        ])
        structural = finite_difference_matrix(
            generator, point, local_search, args.epsilon,
            args.response_resolution, args.direction_batch, "face_edges")
        known = finite_difference_matrix(
            generator, point, known_w, args.epsilon,
            args.response_resolution, args.direction_batch, "face_edges")
        responses.append(residualize(structural, known))
        nuisance_responses.append(finite_difference_matrix(
            generator, point, local_search, args.epsilon,
            args.response_resolution, args.direction_batch, "lowfreq_rgb"))
        print("anchor {} ready".format(sample_id), flush=True)

    methods, metrics = method_coefficients(responses, nuisance_responses, args)
    heldout_points = sample_w(
        generator, args.heldout_seeds, args.trunc_psi, args.trunc_layers)
    rows = []
    saved = {}
    for method, coefficients in methods.items():
        candidate_q = candidate_hilbert_directions(
            coefficients, search_h, train_points, transform, args.alpha)
        saved[method] = candidate_q
        mean_response = sum(responses) / len(responses) @ coefficients
        normalized_mean = mean_response / np.maximum(
            np.linalg.norm(mean_response, axis=0, keepdims=True), 1e-12)
        duplicate = np.abs(normalized_mean.T @ normalized_mean)
        np.fill_diagonal(duplicate, 0.0)
        heldout_by_candidate = [[] for _ in range(args.num_candidates)]
        for point in heldout_points:
            tangents = np.column_stack([
                inverse_chart_tangent(
                    point, candidate_q[:, candidate], transform, args.alpha)
                for candidate in range(args.num_candidates)
            ])
            tangents /= np.maximum(
                np.linalg.norm(tangents, axis=0, keepdims=True), 1e-12)
            candidate_response = finite_difference_matrix(
                generator, point, tangents, args.epsilon,
                args.response_resolution, args.direction_batch, "face_edges")
            known_response = finite_difference_matrix(
                generator, point, known_w, args.epsilon,
                args.response_resolution, args.direction_batch, "face_edges")
            candidate_response /= np.maximum(
                np.linalg.norm(candidate_response, axis=0, keepdims=True), 1e-12)
            known_response /= np.maximum(
                np.linalg.norm(known_response, axis=0, keepdims=True), 1e-12)
            leakage = np.abs(candidate_response.T @ known_response)
            for candidate in range(args.num_candidates):
                heldout_by_candidate[candidate].append(
                    float(np.max(leakage[candidate])))
        for candidate in range(args.num_candidates):
            c = coefficients[:, candidate]
            energies = np.asarray([np.linalg.norm(a @ c) ** 2 for a in responses])
            eigen_errors = []
            c_unit = c / max(np.linalg.norm(c), 1e-12)
            for metric in metrics:
                action = metric @ c_unit
                residual = action - c_unit * float(c_unit @ action)
                eigen_errors.append(np.linalg.norm(residual) / max(np.linalg.norm(action), 1e-12))
            heldout_maxima = heldout_by_candidate[candidate]
            rows.append({
                "method": method,
                "candidate": candidate,
                "cross_anchor_energy_cv": float(np.std(energies) / max(np.mean(energies), 1e-12)),
                "common_eigen_residual": float(np.mean(eigen_errors)),
                "max_same_method_response_cosine": float(np.max(duplicate[candidate])),
                "heldout_known_response_cosine_mean": float(np.mean(heldout_maxima)),
                "heldout_known_response_cosine_max": float(np.max(heldout_maxima)),
            })

        method_dir = output / method
        method_dir.mkdir(exist_ok=True)
        for seed, point in zip(args.heldout_seeds, heldout_points):
            grid_rows = []
            h0 = chart_forward(point, transform, args.alpha)
            for candidate in range(args.num_candidates):
                q = candidate_q[:, candidate]
                q /= max(np.linalg.norm(
                    inverse_chart_tangent(point, q, transform, args.alpha)), 1e-12)
                images = [
                    decode_uint8(generator, chart_inverse(
                        h0 + value * q, transform, args.alpha))
                    for value in args.lambdas
                ]
                grid_rows.append(("candidate_{:02d}".format(candidate), images))
            make_grid(
                grid_rows, args.lambdas,
                method_dir / "heldout_seed_{:03d}.png".format(seed),
                "{} / held-out seed {}".format(method, seed))

    with (output / "candidate_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / "directions.npz", **saved)
    (output / "config.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    print("Saved geometry method search to {}".format(output))


if __name__ == "__main__":
    main()
