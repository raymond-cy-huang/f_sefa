"""Sequentially optimize rays and exclude each accepted ray from later searches."""

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

from config import SEMANTIC_RANKS, parse_args
from evaluation import (
    response_consistency,
    save_convergence_plot,
    save_csv,
    save_heatmap,
    save_json,
    save_traversal_grid,
)
from geometry import HilbertComplement, hilbert_inner
from objective import GANImageObjective
from optimizer import optimize_sphere


def set_seeds(seed, torch):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_w(generator, count, config, torch):
    device = next(generator.parameters()).device
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(config.seed + 101)
    z = torch.randn(
        count, generator.z_space_dim, generator=random_generator, device=device
    )
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w, trunc_psi=config.trunc_psi, trunc_layers=config.trunc_layers
        )
    if w.ndim == 3:
        w = w[:, 0, :]
    expected = (count, config.latent_dimension)
    if tuple(w.shape) != expected:
        raise ValueError(f"Expected W shape {expected}, got {tuple(w.shape)}.")
    return w.detach().cpu().float()


def extract_semantics(generator, factorize_weight, names):
    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    columns = []
    metadata = []
    for name in names:
        rank = SEMANTIC_RANKS[name]
        source_index = int(order[rank])
        columns.append(boundaries[source_index].reshape(-1))
        metadata.append(
            {
                "name": name,
                "eigen_rank": int(rank),
                "source_eigen_index": source_index,
                "eigenvalue": float(eigenvalues[source_index]),
            }
        )
    return np.stack(columns, axis=1), metadata, [int(value) for value in layers]


def prefixed(prefix, values):
    return {f"{prefix}_{key}": float(value) for key, value in values.items()}


def write_direction_csv(path, random_reduced, optimized_reduced, geometry):
    random_direction = geometry.direction(random_reduced)
    optimized_direction = geometry.direction(optimized_reduced)
    rows = []
    for index, (random_value, optimized_value, gram) in enumerate(
        zip(random_direction, optimized_direction, geometry.gram_diag)
    ):
        rows.append(
            {
                "coefficient_index": index,
                "gram_weight": float(gram),
                "random_coefficient": float(random_value),
                "optimized_coefficient": float(optimized_value),
            }
        )
    save_csv(path, rows)


def luminance(image):
    image = np.asarray(image, dtype=np.float64) / 255.0
    return float(
        np.mean(0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2])
    )


def render_ray_outputs(
    ray_dir,
    objective,
    random_point,
    optimized_point,
    test_anchors,
    distances,
    sample_count,
):
    grid_rows = []
    luminance_rows = []
    for sample_index in range(min(sample_count, len(test_anchors))):
        coefficient = test_anchors.coefficients[sample_index]
        for method, point in (("random", random_point), ("ro", optimized_point)):
            images = objective.render_traversal(point, coefficient, distances)
            grid_rows.append((f"seed {sample_index:03d} / {method.upper()}", images))
            values = []
            for distance, image in zip(distances, images):
                value = luminance(image)
                values.append(value)
                luminance_rows.append(
                    {
                        "sample_index": sample_index,
                        "method": method,
                        "hilbert_distance": float(distance),
                        "mean_srgb_luminance": value,
                    }
                )
            values_array = np.asarray(values, dtype=np.float64)
            slope, intercept = np.polyfit(distances, values_array, 1)
            prediction = slope * np.asarray(distances) + intercept
            denominator = np.sum((values_array - np.mean(values_array)) ** 2)
            r_squared = 1.0 - np.sum((values_array - prediction) ** 2) / max(denominator, 1e-12)
            for row in luminance_rows[-len(distances):]:
                row["luminance_slope"] = float(slope)
                row["luminance_r_squared"] = float(r_squared)
                row["strictly_monotonic"] = bool(
                    np.all(np.diff(values) > 0) or np.all(np.diff(values) < 0)
                )
    save_traversal_grid(os.path.join(ray_dir, "traversal.png"), grid_rows, distances)
    save_csv(os.path.join(ray_dir, "traversal_luminance.csv"), luminance_rows)


def write_ray_summary(path, ray_name, metrics, exclusion_rows, complement_dimension):
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(f"# {ray_name}\n\n")
        if metrics["status"] == "accepted":
            handle.write(
                "This ray was accepted only after all known semantics and all "
                "previously accepted rays were removed from its feasible space.\n\n"
            )
        else:
            handle.write(
                "This ray slot did not meet the search-improvement threshold. "
                "It was recorded as skipped and was not added to the excluded subspace.\n\n"
            )
        handle.write(f"- Status: `{metrics['status']}`\n")
        handle.write(f"- Search complement dimension: `{complement_dimension}`\n")
        handle.write(
            f"- Held-out objective improvement: `{metrics['delta_test_objective']:.6e}`\n"
        )
        handle.write(
            f"- Held-out effect change: `{metrics['delta_test_effect']:.6e}`\n"
        )
        handle.write(
            f"- Held-out curvature change: `{metrics['delta_test_curvature']:.6e}`\n"
        )
        handle.write(
            "- Final Hilbert norm error: "
            f"`{metrics['final_hilbert_norm_error']:.3e}`\n"
        )
        handle.write(
            "- Maximum absolute Hilbert cosine with excluded directions: "
            f"`{max(row['absolute_hilbert_cosine'] for row in exclusion_rows):.3e}`\n"
        )


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)

    import torch
    from utils import factorize_weight, get_gan_type, load_generator

    if not torch.cuda.is_available():
        raise RuntimeError("The sequential GAN experiment requires CUDA.")
    set_seeds(config.seed, torch)
    start_time = time.perf_counter()
    generator = load_generator(config.model)
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator.")

    known, semantic_metadata, layers = extract_semantics(
        generator, factorize_weight, config.known_semantics
    )
    excluded = known.copy()
    excluded_names = [f"known_{name}" for name in config.known_semantics]
    accepted_directions = []
    accepted_ray_names = []
    summary_rows = []
    start_ray_index = 1
    accepted_path = os.path.join(config.output_dir, "accepted_rays.npz")
    summary_path = os.path.join(config.output_dir, "complete_orth_summary.csv")
    if config.resume and os.path.exists(accepted_path):
        checkpoint = np.load(accepted_path, allow_pickle=False)
        checkpoint_directions = np.asarray(checkpoint["directions"], dtype=np.float64)
        checkpoint_names = [str(value) for value in checkpoint["names"].tolist()]
        if checkpoint_directions.ndim != 2 or checkpoint_directions.shape[0] != config.latent_dimension:
            raise ValueError("accepted_rays.npz has an incompatible direction matrix.")
        accepted_directions = [
            checkpoint_directions[:, index].copy()
            for index in range(checkpoint_directions.shape[1])
        ]
        accepted_ray_names = checkpoint_names
        if accepted_directions:
            excluded = np.column_stack([excluded, *accepted_directions])
            excluded_names.extend(accepted_ray_names)
    if config.resume and os.path.exists(summary_path):
        with open(summary_path, newline="", encoding="utf-8") as handle:
            summary_rows = list(csv.DictReader(handle))
        for row in summary_rows:
            row.setdefault("status", "accepted")
        if summary_rows:
            start_ray_index = max(int(row["ray_index"]) for row in summary_rows) + 1
    gram_diag = None
    device = next(generator.parameters()).device

    # Anchor generation is shared by every ray, making comparisons paired.
    all_w = sample_w(generator, config.num_search + config.num_test, config, torch)
    initial_geometry = HilbertComplement(excluded)
    anchor_objective = GANImageObjective(
        generator,
        initial_geometry.torch_q(device),
        config.epsilon_h,
        config.curvature_beta,
        config.objective_resolution,
        config.microbatch,
    )
    search_anchors = anchor_objective.prepare_anchors("search", all_w[: config.num_search])
    test_anchors = anchor_objective.prepare_anchors("test", all_w[config.num_search :])
    np.save(os.path.join(config.output_dir, "search_anchors.npy"), search_anchors.coefficients.numpy())
    np.save(os.path.join(config.output_dir, "test_anchors.npy"), test_anchors.coefficients.numpy())

    save_json(
        os.path.join(config.output_dir, "experiment_manifest.json"),
        {
            "experiment": "sequential-complete-hilbert-orthogonal-search",
            "rule": "After ray k is accepted, append it to the excluded set before ray k+1.",
            "manifold_at_ray_k": "{v: [U,V_<k]^T G v=0, v^T G v=1}",
            "config": config.to_dict(),
            "known_semantics": semantic_metadata,
            "factorized_layers": layers,
        },
    )

    rng = np.random.default_rng(config.seed + 3000 + 1009 * start_ray_index)
    accepted_test_responses = []
    for direction in accepted_directions:
        direction_q = torch.as_tensor(
            direction[:, None], dtype=torch.float32, device=device
        )
        replay_objective = GANImageObjective(
            generator,
            direction_q,
            config.epsilon_h,
            config.curvature_beta,
            config.objective_resolution,
            config.microbatch,
        )
        _, replay_responses = replay_objective.evaluate(
            torch.ones(1, dtype=torch.float32, device=device),
            test_anchors,
            collect_responses=True,
        )
        accepted_test_responses.append(replay_responses.astype(np.float64, copy=False))
    distances = [config.epsilon_h * value for value in config.traversal_multipliers]

    for ray_index in range(start_ray_index, config.num_rays + 1):
        ray_name = f"ray{ray_index:03d}"
        ray_dir = os.path.join(config.output_dir, ray_name)
        os.makedirs(ray_dir, exist_ok=True)

        geometry = HilbertComplement(excluded)
        gram_diag = geometry.gram_diag
        objective = GANImageObjective(
            generator,
            geometry.torch_q(device),
            config.epsilon_h,
            config.curvature_beta,
            config.objective_resolution,
            config.microbatch,
        )
        attempt_rows = []
        accepted_attempt = False
        acceptance_mode = "riemannian_optimization"
        if geometry.complement_dimension == 1:
            # The final one-dimensional complement intersects the unit sphere
            # in two antipodal points. Its tangent space is zero-dimensional,
            # so no non-trivial Riemannian update exists.
            acceptance_mode = "unique_residual_direction"
            attempt = 1
            initial_np = np.ones(1, dtype=np.float64)
            initial = torch.ones(1, dtype=torch.float32, device=device)
            optimized = initial.clone()
            optimized_np = initial_np.copy()
            random_search, _ = objective.evaluate(initial, search_anchors)
            random_test, random_responses = objective.evaluate(
                initial, test_anchors, collect_responses=True
            )
            random_consistency, random_cosine, random_spectrum = response_consistency(
                random_responses
            )
            ro_search = dict(random_search)
            ro_test = dict(random_test)
            ro_responses = random_responses.copy()
            ro_consistency = dict(random_consistency)
            ro_cosine = random_cosine.copy()
            ro_spectrum = random_spectrum.copy()
            accepted_attempt = True
            runtime = 0.0
            residual_direction = geometry.direction(optimized_np)
            residual_constraints = geometry.constraint_errors(residual_direction)
            history = [{
                "iteration": 0,
                **ro_search,
                "euclidean_gradient_norm": 0.0,
                "riemannian_gradient_norm": 0.0,
                "tangent_error": 0.0,
                **residual_constraints,
                "accepted_step_size": 0.0,
                "line_search_evaluations": 0,
                "stop_reason": "zero_dimensional_tangent_space",
            }]
            attempt_rows.append({
                "attempt": 1,
                "accepted": True,
                "acceptance_mode": acceptance_mode,
                "accepted_steps": 0,
                "iterations": 0,
                "random_search_objective": random_search["objective"],
                "ro_search_objective": ro_search["objective"],
                "search_objective_improvement": 0.0,
                "runtime_seconds": 0.0,
            })
        else:
            for attempt in range(1, config.max_attempts_per_ray + 1):
                initial_np = geometry.normalize_reduced(
                    rng.normal(size=geometry.complement_dimension)
                )
                initial = torch.as_tensor(initial_np, dtype=torch.float32, device=device)
                random_search, _ = objective.evaluate(initial, search_anchors)
                random_test, random_responses = objective.evaluate(
                    initial, test_anchors, collect_responses=True
                )
                random_consistency, random_cosine, random_spectrum = response_consistency(
                    random_responses
                )

                optimized, history, runtime = optimize_sphere(
                    objective=objective,
                    anchors=search_anchors,
                    initial=initial,
                    geometry=geometry,
                    learning_rate=config.learning_rate,
                    min_learning_rate=config.min_learning_rate,
                    armijo_constant=config.armijo_constant,
                    line_search_steps=config.line_search_steps,
                    max_iterations=config.max_iterations,
                    gradient_tolerance=config.gradient_tolerance,
                )
                optimized_np = optimized.cpu().double().numpy()
                ro_search, _ = objective.evaluate(optimized, search_anchors)
                search_improvement = ro_search["objective"] - random_search["objective"]
                accepted_steps = sum(
                    float(row.get("accepted_step_size", 0.0)) > 0.0 for row in history
                )
                accepted_attempt = (
                    accepted_steps > 0
                    and search_improvement >= config.minimum_search_improvement
                )
                attempt_rows.append(
                    {
                        "attempt": attempt,
                        "accepted": accepted_attempt,
                        "acceptance_mode": acceptance_mode,
                        "accepted_steps": accepted_steps,
                        "iterations": int(history[-1]["iteration"]),
                        "random_search_objective": random_search["objective"],
                        "ro_search_objective": ro_search["objective"],
                        "search_objective_improvement": search_improvement,
                        "runtime_seconds": runtime,
                    }
                )
                if accepted_attempt:
                    break

        save_csv(os.path.join(ray_dir, "attempts.csv"), attempt_rows)
        status = "accepted" if accepted_attempt else "skipped"

        accepted = geometry.direction(optimized_np)
        ro_test, ro_responses = objective.evaluate(
            optimized, test_anchors, collect_responses=True
        )
        ro_consistency, ro_cosine, ro_spectrum = response_consistency(ro_responses)
        constraints = geometry.constraint_errors(accepted)

        exclusion_rows = []
        for name, direction in zip(excluded_names, excluded.T):
            direction_norm = np.sqrt(hilbert_inner(direction, direction, gram_diag))
            cosine = hilbert_inner(accepted, direction, gram_diag) / direction_norm
            exclusion_rows.append(
                {
                    "excluded_direction": name,
                    "signed_hilbert_cosine": float(cosine),
                    "absolute_hilbert_cosine": abs(float(cosine)),
                }
            )

        metrics = {
            "ray": ray_name,
            "ray_index": ray_index,
            "status": status,
            "excluded_direction_count": excluded.shape[1],
            "complement_dimension": geometry.complement_dimension,
            "optimization_runtime_seconds": runtime,
            "accepted_attempt": attempt,
            "acceptance_mode": acceptance_mode,
            "iterations": int(history[-1]["iteration"]),
            **prefixed("random_search", random_search),
            **prefixed("random_test", random_test),
            **prefixed("random_test", random_consistency),
            **prefixed("ro_search", ro_search),
            **prefixed("ro_test", ro_test),
            **prefixed("ro_test", ro_consistency),
            "delta_search_objective": ro_search["objective"] - random_search["objective"],
            "delta_test_objective": ro_test["objective"] - random_test["objective"],
            "delta_test_effect": ro_test["effect"] - random_test["effect"],
            "delta_test_curvature": ro_test["curvature"] - random_test["curvature"],
            "final_orthogonality_l2": constraints["orthogonality_l2"],
            "final_hilbert_norm_error": constraints["hilbert_norm_error"],
            "max_abs_excluded_hilbert_cosine": max(
                row["absolute_hilbert_cosine"] for row in exclusion_rows
            ),
        }
        summary_rows.append(metrics)

        save_csv(os.path.join(ray_dir, "metrics.csv"), [metrics])
        save_csv(os.path.join(ray_dir, "convergence.csv"), history)
        save_csv(os.path.join(ray_dir, "exclusions.csv"), exclusion_rows)
        save_csv(
            os.path.join(ray_dir, "response_cosine_random.csv"),
            [
                {"sample_index": row, **{f"sample_{col:03d}": float(value) for col, value in enumerate(values)}}
                for row, values in enumerate(random_cosine)
            ],
        )
        save_csv(
            os.path.join(ray_dir, "response_cosine_ro.csv"),
            [
                {"sample_index": row, **{f"sample_{col:03d}": float(value) for col, value in enumerate(values)}}
                for row, values in enumerate(ro_cosine)
            ],
        )
        save_csv(
            os.path.join(ray_dir, "response_spectrum.csv"),
            [
                {
                    "component": index,
                    "random_singular_value": float(random_spectrum[index]) if index < len(random_spectrum) else "",
                    "ro_singular_value": float(ro_spectrum[index]) if index < len(ro_spectrum) else "",
                }
                for index in range(max(len(random_spectrum), len(ro_spectrum)))
            ],
        )
        write_direction_csv(
            os.path.join(ray_dir, "direction_coefficients.csv"),
            initial_np,
            optimized_np,
            geometry,
        )
        np.savez_compressed(
            os.path.join(ray_dir, "direction.npz"),
            random_reduced=initial_np,
            optimized_reduced=optimized_np,
            random_function_direction=geometry.direction(initial_np),
            optimized_function_direction=accepted,
            gram_diagonal=gram_diag,
        )
        save_convergence_plot(os.path.join(ray_dir, "convergence.png"), history)
        save_heatmap(
            os.path.join(ray_dir, "response_cosine_random.png"),
            random_cosine,
            f"{ray_name} Random response cosine",
        )
        save_heatmap(
            os.path.join(ray_dir, "response_cosine_ro.png"),
            ro_cosine,
            f"{ray_name} RO response cosine",
        )
        render_ray_outputs(
            ray_dir,
            objective,
            initial,
            optimized,
            test_anchors,
            distances,
            config.grid_test_samples,
        )
        write_ray_summary(
            os.path.join(ray_dir, "SUMMARY.md"),
            ray_name,
            metrics,
            exclusion_rows,
            geometry.complement_dimension,
        )

        if not accepted_attempt:
            save_csv(summary_path, summary_rows)
            print(
                f"Skipped {ray_name}; search objective did not reach the "
                f"minimum improvement after {config.max_attempts_per_ray} attempts."
            )
            continue

        # The accepted ray is removed from every subsequent feasible space.
        accepted_directions.append(accepted)
        accepted_test_responses.append(ro_responses.astype(np.float64, copy=False))
        accepted_ray_names.append(ray_name)
        excluded = np.column_stack([excluded, accepted])
        excluded_names.append(ray_name)
        save_csv(summary_path, summary_rows)
        np.savez_compressed(
            accepted_path,
            directions=np.stack(accepted_directions, axis=1),
            gram_diagonal=gram_diag,
            names=np.asarray(accepted_ray_names),
        )
        print(
            f"Accepted {ray_name}; removed it from future searches. "
            f"test delta J={metrics['delta_test_objective']:.6e}, "
            f"max excluded cosine={metrics['max_abs_excluded_hilbert_cosine']:.3e}"
        )

    pairwise_rows = []
    for row_index, left in enumerate(accepted_directions):
        row = {"ray": accepted_ray_names[row_index]}
        for col_index, right in enumerate(accepted_directions):
            row[accepted_ray_names[col_index]] = hilbert_inner(left, right, gram_diag)
        pairwise_rows.append(row)
    save_csv(
        os.path.join(config.output_dir, "pairwise_hilbert_cosine.csv"), pairwise_rows
    )
    response_rows = []
    for row_index, left in enumerate(accepted_test_responses):
        row = {"ray": accepted_ray_names[row_index]}
        left_flat = left.reshape(-1)
        for col_index, right in enumerate(accepted_test_responses):
            right_flat = right.reshape(-1)
            denominator = max(
                float(np.linalg.norm(left_flat) * np.linalg.norm(right_flat)),
                1e-12,
            )
            row[accepted_ray_names[col_index]] = float(
                np.dot(left_flat, right_flat) / denominator
            )
        response_rows.append(row)
    save_csv(
        os.path.join(config.output_dir, "pairwise_image_response_cosine.csv"),
        response_rows,
    )
    save_json(
        os.path.join(config.output_dir, "completion.json"),
        {
            "attempted_ray_slots": config.num_rays,
            "accepted_rays": len(accepted_directions),
            "skipped_rays": sum(row.get("status") == "skipped" for row in summary_rows),
            "elapsed_seconds": time.perf_counter() - start_time,
            "final_excluded_dimension": excluded.shape[1],
            "remaining_complement_dimension": config.latent_dimension - excluded.shape[1],
        },
    )


if __name__ == "__main__":
    main()

