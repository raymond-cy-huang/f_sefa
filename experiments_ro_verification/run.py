"""Run paired Random-vs-RO verification in Legendre function space."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
for path in (ROOT_DIR, EXPERIMENT_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from config import SEMANTIC_RANKS, parse_args
from evaluation import (
    bootstrap_mean_interval,
    response_consistency,
    save_convergence_plot,
    save_csv,
    save_heatmap,
    save_json,
    save_paired_plot,
    save_traversal_grid,
)
from geometry import HilbertComplement
from objective import GANImageObjective
from optimizer import optimize_sphere


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def set_seeds(seed: int, torch) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_w(generator, count, config, torch):
    device = next(generator.parameters()).device
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(config.seed + 101)
    z = torch.randn(
        count,
        generator.z_space_dim,
        generator=random_generator,
        device=device,
    )
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w,
            trunc_psi=config.trunc_psi,
            trunc_layers=config.trunc_layers,
        )
    if w.ndim == 3:
        w = w[:, 0, :]
    expected = (count, config.latent_dimension)
    if tuple(w.shape) != expected:
        raise ValueError(f"Expected sampled W shape {expected}, got {tuple(w.shape)}.")
    return w.detach().cpu().float()


def extract_semantics(generator, factorize_weight, names):
    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    directions = []
    metadata = []
    for name in names:
        rank = SEMANTIC_RANKS[name]
        source_index = int(order[rank])
        directions.append(boundaries[source_index].reshape(-1))
        metadata.append(
            {
                "name": name,
                "eigen_rank": int(rank),
                "source_eigen_index": source_index,
                "eigenvalue": float(eigenvalues[source_index]),
            }
        )
    return np.stack(directions, axis=1), metadata, [int(value) for value in layers]


def metric_prefix(prefix, metrics):
    return {f"{prefix}_{key}": float(value) for key, value in metrics.items()}


def save_direction(path, initial, optimized, geometry):
    initial_np = initial.detach().cpu().double().numpy()
    optimized_np = optimized.detach().cpu().double().numpy()
    np.savez_compressed(
        path,
        initial_reduced=initial_np,
        optimized_reduced=optimized_np,
        initial_function_direction=geometry.direction(initial_np),
        optimized_function_direction=geometry.direction(optimized_np),
        gram_diagonal=geometry.gram_diag,
    )


def generate_run_grid(
    path,
    objective,
    random_point,
    optimized_point,
    test_anchors,
    distances,
    sample_count,
):
    rows = []
    for sample_index in range(min(sample_count, len(test_anchors))):
        coefficient = test_anchors.coefficients[sample_index]
        random_images = objective.render_traversal(
            random_point, coefficient, distances
        )
        optimized_images = objective.render_traversal(
            optimized_point, coefficient, distances
        )
        rows.append((f"seed {sample_index:03d} / Random", random_images))
        rows.append((f"seed {sample_index:03d} / RO", optimized_images))
    save_traversal_grid(path, rows, distances)


def aggregate_results(rows, seed):
    quantities = (
        "test_objective",
        "test_effect",
        "test_curvature",
        "test_mean_pairwise_response_cosine",
        "test_response_pc1_explained_variance",
    )
    result = {}
    for quantity in quantities:
        differences = np.array(
            [row[f"ro_{quantity}"] - row[f"random_{quantity}"] for row in rows],
            dtype=np.float64,
        )
        result[f"paired_delta_{quantity}"] = bootstrap_mean_interval(
            differences, seed=seed
        )
    try:
        from scipy.stats import wilcoxon

        objective_differences = np.array(
            [row["delta_test_objective"] for row in rows], dtype=np.float64
        )
        if len(objective_differences) > 1 and np.any(objective_differences != 0):
            statistic, pvalue = wilcoxon(
                objective_differences,
                alternative="greater",
                zero_method="wilcox",
            )
            result["wilcoxon_test_objective_greater"] = {
                "statistic": float(statistic),
                "pvalue": float(pvalue),
            }
    except (ImportError, ValueError):
        pass
    return result


def write_summary(path, config, geometry, rows, aggregate, runtime):
    delta = aggregate["paired_delta_test_objective"]
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# Function-Space Riemannian Verification\n\n")
        handle.write(
            "This experiment compares each random feasible function direction "
            "with Riemannian optimization initialized from that same direction.\n\n"
        )
        handle.write("## Scope\n\n")
        handle.write(
            "RQ1/RQ2 test structured GAN variation and held-out generalization. "
            "They do not by themselves establish semantic discovery; controlled "
            "semantic evidence requires the separate leave-one-out experiment.\n\n"
        )
        handle.write("## Geometry\n\n")
        handle.write(f"- Ambient function dimension: `{geometry.dimension}`\n")
        handle.write(f"- Known semantic rank: `{geometry.known_rank}`\n")
        handle.write(
            f"- Feasible sphere dimension: `{geometry.complement_dimension - 1}`\n"
        )
        handle.write(
            "- Maximum known/complement Hilbert inner product: "
            f"`{geometry.diagnostics.max_known_complement_inner:.3e}`\n"
        )
        handle.write(
            "- Maximum complement Gram error: "
            f"`{geometry.diagnostics.max_complement_gram_error:.3e}`\n\n"
        )
        handle.write("## Paired held-out result\n\n")
        handle.write(f"- Restarts: `{len(rows)}`\n")
        handle.write(
            "- Mean RO - Random test objective: "
            f"`{delta['mean']:.6e}`\n"
        )
        handle.write(
            "- Bootstrap 95% interval: "
            f"`[{delta['ci_low']:.6e}, {delta['ci_high']:.6e}]`\n"
        )
        handle.write(
            "- Fraction of restarts with positive held-out improvement: "
            f"`{delta['positive_fraction']:.3f}`\n"
        )
        handle.write(f"- Total runtime (seconds): `{runtime:.2f}`\n\n")
        handle.write("## Configuration\n\n")
        handle.write("```json\n")
        handle.write(json.dumps(config.to_dict(), indent=2))
        handle.write("\n```\n")


def main():
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)
    runs_dir = os.path.join(config.output_dir, "runs")
    grids_dir = os.path.join(config.output_dir, "grids", "random_vs_ro")
    os.makedirs(runs_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    import torch
    from utils import factorize_weight, get_gan_type, load_generator

    if not torch.cuda.is_available():
        raise RuntimeError("The full GAN experiment requires a CUDA GPU.")
    set_seeds(config.seed, torch)
    overall_start = time.perf_counter()
    generator = load_generator(config.model)
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator.")
    if generator.w_space_dim != config.latent_dimension:
        raise ValueError("Generator W dimension and configured dimension disagree.")

    known, semantic_metadata, factorized_layers = extract_semantics(
        generator, factorize_weight, config.known_semantics
    )
    geometry = HilbertComplement(known)
    device = next(generator.parameters()).device
    objective = GANImageObjective(
        generator=generator,
        complement_q=geometry.torch_q(device),
        epsilon_h=config.epsilon_h,
        curvature_beta=config.curvature_beta,
        resolution=config.objective_resolution,
        microbatch=config.microbatch,
    )

    all_w = sample_w(generator, config.num_search + config.num_test, config, torch)
    search_anchors = objective.prepare_anchors("search", all_w[: config.num_search])
    test_anchors = objective.prepare_anchors("test", all_w[config.num_search :])
    np.save(os.path.join(config.output_dir, "search_anchors.npy"), search_anchors.coefficients.numpy())
    np.save(os.path.join(config.output_dir, "test_anchors.npy"), test_anchors.coefficients.numpy())

    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", config.model + ".pth")
    manifest = {
        "experiment": "function-space-riemannian-random-vs-ro",
        "coordinate_representation": "raw Legendre coefficients",
        "phi": "Phi(w)(x) = sum_k w_k P_k(x)",
        "psi": "exact coefficient-coordinate inverse (identity in stored coordinates)",
        "hilbert_metric": "G_kk = 2/(2k+1)",
        "manifold": "{v: U^T G v = 0, v^T G v = 1}",
        "reduced_parameterization": "v=Qz, Q^TGQ=I, ||z||_2=1",
        "objective": "J=E-beta*C with central image derivatives",
        "config": config.to_dict(),
        "semantics": semantic_metadata,
        "factorized_layers": factorized_layers,
        "geometry": geometry.diagnostics.__dict__,
        "checkpoint": {
            "path": checkpoint_path,
            "sha256": sha256(checkpoint_path) if os.path.exists(checkpoint_path) else None,
        },
    }
    save_json(os.path.join(config.output_dir, "experiment_manifest.json"), manifest)

    rng = np.random.default_rng(config.seed + 1000)
    paired_rows = []
    distances = [config.epsilon_h * value for value in config.traversal_multipliers]
    for restart in range(config.num_restarts):
        run_dir = os.path.join(runs_dir, f"run_{restart:02d}")
        os.makedirs(run_dir, exist_ok=True)
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

        optimized, history, optimization_runtime = optimize_sphere(
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
        ro_search, _ = objective.evaluate(optimized, search_anchors)
        ro_test, ro_responses = objective.evaluate(
            optimized, test_anchors, collect_responses=True
        )
        ro_consistency, ro_cosine, ro_spectrum = response_consistency(ro_responses)

        initial_direction = geometry.direction(initial_np)
        optimized_np = optimized.cpu().double().numpy()
        optimized_direction = geometry.direction(optimized_np)
        initial_constraints = geometry.constraint_errors(initial_direction)
        final_constraints = geometry.constraint_errors(optimized_direction)

        row = {
            "restart": restart,
            "optimization_runtime_seconds": optimization_runtime,
            "iterations": int(history[-1]["iteration"]),
            **metric_prefix("random_search", random_search),
            **metric_prefix("random_test", random_test),
            **metric_prefix("random_test", random_consistency),
            **metric_prefix("ro_search", ro_search),
            **metric_prefix("ro_test", ro_test),
            **metric_prefix("ro_test", ro_consistency),
            "delta_search_objective": ro_search["objective"] - random_search["objective"],
            "delta_test_objective": ro_test["objective"] - random_test["objective"],
            "delta_test_effect": ro_test["effect"] - random_test["effect"],
            "delta_test_curvature": ro_test["curvature"] - random_test["curvature"],
            "initial_orthogonality_l2": initial_constraints["orthogonality_l2"],
            "initial_hilbert_norm_error": initial_constraints["hilbert_norm_error"],
            "final_orthogonality_l2": final_constraints["orthogonality_l2"],
            "final_hilbert_norm_error": final_constraints["hilbert_norm_error"],
        }
        paired_rows.append(row)
        save_csv(os.path.join(run_dir, "convergence.csv"), history)
        save_convergence_plot(os.path.join(run_dir, "convergence.png"), history)
        save_direction(
            os.path.join(run_dir, "direction.npz"), initial, optimized, geometry
        )
        save_json(os.path.join(run_dir, "metrics.json"), row)

        if restart == 0:
            save_heatmap(
                os.path.join(run_dir, "response_cosine_random.png"),
                random_cosine,
                "Random held-out image-response cosine",
            )
            save_heatmap(
                os.path.join(run_dir, "response_cosine_ro.png"),
                ro_cosine,
                "RO held-out image-response cosine",
            )
            np.savez_compressed(
                os.path.join(run_dir, "response_spectrum.npz"),
                random=random_spectrum,
                ro=ro_spectrum,
            )
        if restart < config.grid_restarts:
            generate_run_grid(
                os.path.join(grids_dir, f"run_{restart:02d}.png"),
                objective,
                initial,
                optimized,
                test_anchors,
                distances,
                config.grid_test_samples,
            )
        print(
            f"Restart {restart + 1}/{config.num_restarts}: "
            f"test delta J={row['delta_test_objective']:.6e}, "
            f"iterations={row['iterations']}"
        )

    paired_path = os.path.join(config.output_dir, "paired_metrics.csv")
    save_csv(paired_path, paired_rows)
    aggregate = aggregate_results(paired_rows, config.seed + 2000)
    aggregate["total_runtime_seconds"] = time.perf_counter() - overall_start
    save_json(os.path.join(config.output_dir, "aggregate_metrics.json"), aggregate)
    save_paired_plot(os.path.join(config.output_dir, "paired_comparison.png"), paired_rows)
    write_summary(
        os.path.join(config.output_dir, "SUMMARY.md"),
        config,
        geometry,
        paired_rows,
        aggregate,
        aggregate["total_runtime_seconds"],
    )
    print(f"Saved paired metrics: {paired_path}")
    print(f"Saved summary: {os.path.join(config.output_dir, 'SUMMARY.md')}")


if __name__ == "__main__":
    main()

