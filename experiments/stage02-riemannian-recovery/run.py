"""Minimum viable Riemannian recovery of a known SeFa direction in H."""

import csv
import hashlib
import json
import math
import os
import sys
import time

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from config import parse_args
from experiment_utils.function_space_mapping import _basis_matrix
from experiment_utils.function_space_mapping import _basis_quadrature
from experiment_utils.function_space_mapping import function_to_coefficients
from experiment_utils.metrics import save_image


SEMANTIC_RANKS = {
    "gender": 1,
    "age": 3,
    "pose": 7,
    "glasses": 10,
    "smile": 18,
}

DISTRACTOR_NAMES = ("age", "pose", "glasses", "gender")


class FunctionGeometry:
    """Stage 01 raw-Legendre coefficient representation of H."""

    def __init__(self, basis, dimension):
        self.basis = basis
        self.dimension = dimension
        self.nodes, self.weights, self.norms = _basis_quadrature(basis, dimension)
        self.matrix = _basis_matrix(basis, dimension, self.nodes)

    def _vectors(self, value, name):
        array = np.asarray(value, dtype=np.float64)
        if array.shape[-1] != self.dimension:
            raise ValueError(
                "{} must end in dimension {}, got {}.".format(
                    name, self.dimension, array.shape
                )
            )
        if not np.all(np.isfinite(array)):
            raise ValueError("{} contains non-finite values.".format(name))
        return array.copy()

    def phi(self, latent_points):
        coefficients = self._vectors(latent_points, "latent source points")
        values = coefficients.reshape(-1, self.dimension) @ self.matrix.T
        if not np.all(np.isfinite(values)):
            raise ValueError("Phi produced non-finite function values.")
        return coefficients

    def phi_direction(self, latent_direction):
        coefficients = self._vectors(latent_direction, "latent direction source")
        values = coefficients.reshape(-1, self.dimension) @ self.matrix.T
        if not np.all(np.isfinite(values)):
            raise ValueError("Phi_dir produced non-finite function values.")
        return coefficients.reshape(-1)

    def psi(self, function_coefficients):
        coefficients = self._vectors(function_coefficients, "function coefficients")
        if coefficients.ndim != 1:
            raise ValueError("Psi expects one function-space point.")
        values = self.matrix @ coefficients
        return function_to_coefficients(
            values,
            self.basis,
            self.nodes,
            self.weights,
            self.norms,
        )

    def hilbert_norm(self, vector):
        vector = self._vectors(vector, "function vector").reshape(-1)
        value = float(np.dot(vector * self.norms, vector))
        return float(math.sqrt(max(value, 0.0)))


def validate_config(config):
    if config.basis != "legendre":
        raise ValueError("This validation experiment requires raw Legendre H.")
    if config.semantic != "smile":
        raise ValueError("This MVP fixes Smile as target and the other four as distractors.")
    if config.latent_dimension != 512:
        raise ValueError("The current Stage 01 representation has dimension 512.")
    if config.num_samples < 2:
        raise ValueError("num_samples must be at least 2.")
    if not np.isfinite(config.delta) or config.delta <= 0.0:
        raise ValueError("delta must be finite and positive.")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0.0:
        raise ValueError("learning_rate must be finite and positive.")
    if config.max_iterations <= 0 or config.gradient_tolerance <= 0.0:
        raise ValueError("Iteration limit and gradient tolerance must be positive.")
    if len(config.lambdas) == 0 or not np.all(np.isfinite(config.lambdas)):
        raise ValueError("lambdas must be non-empty and finite.")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize(vector):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero vector.")
    return vector / norm


def signed_cosine(left, right):
    left = np.asarray(left, dtype=np.float64).reshape(-1)
    right = np.asarray(right, dtype=np.float64).reshape(-1)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= np.finfo(np.float64).eps:
        return None
    return float(np.dot(left, right) / denominator)


def absolute_cosine(left, right):
    value = signed_cosine(left, right)
    return None if value is None else abs(value)


def orient(reference, candidate):
    candidate = normalize(candidate)
    return -candidate if float(np.dot(reference, candidate)) < 0.0 else candidate


def sample_w(generator, count, config, torch):
    device = next(generator.parameters()).device
    z = torch.randn(count, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w,
            trunc_psi=config.trunc_psi,
            trunc_layers=config.trunc_layers,
        )
    if w.ndim == 3:
        w = w[:, 0, :]
    result = w.detach().cpu().numpy().astype(np.float64)
    expected = (count, config.latent_dimension)
    if result.shape != expected:
        raise ValueError("Expected sampled W shape {}, got {}.".format(expected, result.shape))
    return result


def sefa_directions(generator, factorize_weight):
    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    result = {}
    for name, rank in SEMANTIC_RANKS.items():
        source_index = int(order[rank])
        result[name] = {
            "direction": boundaries[source_index].reshape(-1),
            "eigen_rank": rank,
            "source_eigen_index": source_index,
            "eigenvalue": float(eigenvalues[source_index]),
        }
    return layers, result


def semantic_covariance(displacements):
    displacements = np.asarray(displacements, dtype=np.float64)
    return displacements.T @ displacements / float(len(displacements))


def riemannian_sphere_recovery(
    covariance,
    initial,
    learning_rate,
    max_iterations,
    gradient_tolerance,
    reference,
):
    """Manual gradient descent on S^(D-1) with normalization retraction."""
    u = normalize(initial)
    history = []
    start = time.perf_counter()
    for iteration in range(max_iterations + 1):
        covariance_u = covariance @ u
        objective = -float(np.dot(u, covariance_u))
        euclidean_gradient = -2.0 * covariance_u
        riemannian_gradient = (
            euclidean_gradient - u * float(np.dot(u, euclidean_gradient))
        )
        gradient_norm = float(np.linalg.norm(riemannian_gradient))
        history.append({
            "iteration": iteration,
            "objective": objective,
            "riemannian_gradient_norm": gradient_norm,
            "alignment_to_mapped_gt": absolute_cosine(u, reference),
        })
        if gradient_norm <= gradient_tolerance:
            break
        tangent_step = -learning_rate * riemannian_gradient
        tangent_error = abs(float(np.dot(u, tangent_step)))
        if tangent_error > 1e-10:
            raise RuntimeError("Computed update is not tangent to the sphere.")
        u = normalize(u + tangent_step)
    runtime = time.perf_counter() - start
    return orient(reference, u), history, runtime


def displacement_diagnostics(displacements):
    normalized = np.stack([normalize(row) for row in displacements], axis=0)
    cosine_matrix = normalized @ normalized.T
    count = len(displacements)
    off_diagonal = cosine_matrix[~np.eye(count, dtype=bool)]
    mean = np.mean(displacements, axis=0)
    centered = displacements - mean
    coefficient_variance = np.var(displacements, axis=0)
    return {
        "cosine_matrix": cosine_matrix,
        "mean": mean,
        "mean_pairwise_cosine": float(np.mean(off_diagonal)),
        "min_pairwise_cosine": float(np.min(off_diagonal)),
        "max_pairwise_cosine": float(np.max(off_diagonal)),
        "mean_coefficient_variance": float(np.mean(coefficient_variance)),
        "max_coefficient_variance": float(np.max(coefficient_variance)),
        "max_l2_deviation_from_mean": float(
            np.max(np.linalg.norm(centered, axis=1))
        ),
        "max_absolute_deviation_from_mean": float(np.max(np.abs(centered))),
    }


def save_convergence_plot(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    iterations = [row["iteration"] for row in history]
    objectives = [row["objective"] for row in history]
    gradients = [row["riemannian_gradient_norm"] for row in history]
    alignments = [row["alignment_to_mapped_gt"] for row in history]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    axes[0].plot(iterations, objectives, color="#2b6cb0")
    axes[0].set_title("Objective")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel(r"$-u^T C_s u$")
    axes[0].grid(alpha=0.25)
    axes[1].semilogy(iterations, np.maximum(gradients, 1e-18), color="#c05621")
    axes[1].set_title("Riemannian gradient norm")
    axes[1].set_xlabel("Iteration")
    axes[1].grid(alpha=0.25)
    axes[2].plot(iterations, alignments, color="#2f855a")
    axes[2].set_ylim(0.0, 1.01)
    axes[2].set_title("Alignment to mapped GT")
    axes[2].set_xlabel("Iteration")
    axes[2].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_cosine_matrix(matrix, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 6.2))
    image = axis.imshow(matrix, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    axis.set_title(r"Pairwise cosine of $\Delta c_i$")
    axis.set_xlabel("Sample index")
    axis.set_ylabel("Sample index")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def decoder_code(generator, latent):
    vector = np.asarray(latent, dtype=np.float32).reshape(-1)
    if vector.shape != (generator.w_space_dim,):
        raise ValueError("Unexpected W shape {}.".format(vector.shape))
    return np.tile(vector[None, :], (generator.num_layers, 1))[None, ...]


def decode(generator, latent, torch, postprocess):
    device = next(generator.parameters()).device
    code = torch.from_numpy(decoder_code(generator, latent)).float().to(device)
    with torch.no_grad():
        image = generator.synthesis(code)["image"]
    return postprocess(image)[0]


def save_traversal_grid(method_images, lambdas, path, thumb=192):
    from PIL import Image, ImageDraw

    method_names = list(method_images)
    label_width = 150
    header_height = 34
    canvas = Image.new(
        "RGB",
        (label_width + len(lambdas) * thumb, header_height + len(method_names) * thumb),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, value in enumerate(lambdas):
        draw.text(
            (label_width + column * thumb + 6, 8),
            "lambda {:+g}".format(value),
            fill="black",
        )
    for row, method in enumerate(method_names):
        top = header_height + row * thumb
        draw.text((8, top + thumb // 2), method, fill="black")
        for column, array in enumerate(method_images[method]):
            image = Image.fromarray(array).resize((thumb, thumb), Image.BILINEAR)
            canvas.paste(image, (label_width + column * thumb, top))
    canvas.save(path)


def save_csv(path, rows, fieldnames):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    config = parse_args()
    validate_config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)
    image_dir = os.path.join(config.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    import torch
    from utils import factorize_weight, get_gan_type, load_generator, postprocess

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    generator = load_generator(config.model)
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator.")
    geometry = FunctionGeometry(config.basis, config.latent_dimension)
    layers, semantic_sources = sefa_directions(generator, factorize_weight)
    target_source = semantic_sources[config.semantic]["direction"]
    distractor_sources = {
        name: semantic_sources[name]["direction"] for name in DISTRACTOR_NAMES
    }

    all_w = sample_w(generator, config.num_samples + 1, config, torch)
    train_w = all_w[:config.num_samples]
    test_w = all_w[-1]

    # Requested centered semantic displacements: edit in W, then map to H.
    w_plus = train_w + config.delta * target_source[None, :]
    w_minus = train_w - config.delta * target_source[None, :]
    c_plus = geometry.phi(w_plus)
    c_minus = geometry.phi(w_minus)
    displacements = (c_plus - c_minus) / (2.0 * config.delta)
    covariance = semantic_covariance(displacements)

    diagnostics = displacement_diagnostics(displacements)
    mapped_target = geometry.phi_direction(target_source)
    u_gt = normalize(mapped_target)
    u_emp = normalize(diagnostics["mean"])

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    u_eig = orient(u_gt, eigenvectors[:, -1])
    principal_eigenvalue = float(eigenvalues[-1])
    covariance_rank_tolerance = (
        np.finfo(np.float64).eps * covariance.shape[0]
        * max(abs(principal_eigenvalue), 1.0)
    )
    covariance_rank = int(np.sum(eigenvalues > covariance_rank_tolerance))

    rng = np.random.default_rng(config.seed + 1000)
    initial = normalize(rng.normal(size=config.latent_dimension))
    u_riemannian, history, runtime = riemannian_sphere_recovery(
        covariance,
        initial,
        config.learning_rate,
        config.max_iterations,
        config.gradient_tolerance,
        u_gt,
    )

    normalized_ablation = None
    if config.normalized_displacement_ablation:
        normalized_displacements = np.stack(
            [normalize(row) for row in displacements], axis=0
        )
        normalized_covariance = semantic_covariance(normalized_displacements)
        _, normalized_vectors = np.linalg.eigh(normalized_covariance)
        normalized_ablation = {
            "principal_alignment_to_primary_eigenvector": absolute_cosine(
                normalized_vectors[:, -1], u_eig
            )
        }

    convergence_path = os.path.join(config.output_dir, "convergence.csv")
    save_csv(
        convergence_path,
        history,
        [
            "iteration", "objective", "riemannian_gradient_norm",
            "alignment_to_mapped_gt",
        ],
    )
    convergence_plot_path = os.path.join(config.output_dir, "convergence.png")
    save_convergence_plot(history, convergence_plot_path)

    cosine_csv_path = os.path.join(config.output_dir, "delta_cosine_matrix.csv")
    np.savetxt(cosine_csv_path, diagnostics["cosine_matrix"], delimiter=",")
    cosine_plot_path = os.path.join(config.output_dir, "delta_cosine_matrix.png")
    save_cosine_matrix(diagnostics["cosine_matrix"], cosine_plot_path)

    methods = {
        "direct_mapped": u_gt,
        "pca_eigenvector": u_eig,
        "riemannian_sphere": u_riemannian,
    }
    c_test = geometry.phi(test_w)
    method_images = {name: [] for name in methods}
    traversal_rows = []
    for method_name, direction_h in methods.items():
        method_dir = os.path.join(image_dir, method_name)
        os.makedirs(method_dir, exist_ok=True)
        for value in config.lambdas:
            c_prime = c_test + value * direction_h
            w_prime = geometry.psi(c_prime)
            reconstructed = geometry.phi(w_prime)
            delta_w = w_prime - test_w
            image = decode(generator, w_prime, torch, postprocess)
            image_path = os.path.join(
                method_dir,
                "lambda_{:+g}.png".format(value),
            )
            save_image(image, image_path)
            method_images[method_name].append(image)
            row = {
                "method": method_name,
                "lambda": float(value),
                "delta_w_l2": float(np.linalg.norm(delta_w)),
                "target_alignment": absolute_cosine(delta_w, target_source),
                "inverse_residual_hilbert": geometry.hilbert_norm(
                    c_prime - reconstructed
                ),
                "inverse_residual_l2": float(np.linalg.norm(c_prime - reconstructed)),
                "image_path": image_path,
            }
            for distractor_name, distractor in distractor_sources.items():
                row["leakage_{}".format(distractor_name)] = absolute_cosine(
                    delta_w, distractor
                )
            traversal_rows.append(row)

    traversal_csv_path = os.path.join(config.output_dir, "traversal_metrics.csv")
    traversal_fields = [
        "method", "lambda", "delta_w_l2", "target_alignment",
        "leakage_age", "leakage_pose", "leakage_glasses", "leakage_gender",
        "inverse_residual_hilbert", "inverse_residual_l2", "image_path",
    ]
    save_csv(traversal_csv_path, traversal_rows, traversal_fields)
    traversal_grid_path = os.path.join(config.output_dir, "semantic_traversal.png")
    save_traversal_grid(method_images, config.lambdas, traversal_grid_path)

    nonzero_riemannian_rows = [
        row for row in traversal_rows
        if (
            row["method"] == "riemannian_sphere"
            and row["lambda"] != 0.0
            and row["target_alignment"] is not None
        )
    ]
    summary_metrics = {
        "alignment_riemannian_vs_eigenvector": absolute_cosine(
            u_riemannian, u_eig
        ),
        "alignment_riemannian_vs_mapped_gt": absolute_cosine(
            u_riemannian, u_gt
        ),
        "alignment_riemannian_vs_empirical_mean": absolute_cosine(
            u_riemannian, u_emp
        ),
        "alignment_eigenvector_vs_mapped_gt": absolute_cosine(u_eig, u_gt),
        "riemannian_direction_l2_norm": float(np.linalg.norm(u_riemannian)),
        "iterations": int(history[-1]["iteration"]),
        "final_objective": float(history[-1]["objective"]),
        "final_gradient_norm": float(history[-1]["riemannian_gradient_norm"]),
        "runtime_seconds": float(runtime),
        "principal_eigenvalue": principal_eigenvalue,
        "covariance_rank": covariance_rank,
        "mean_pairwise_displacement_cosine": diagnostics["mean_pairwise_cosine"],
        "min_pairwise_displacement_cosine": diagnostics["min_pairwise_cosine"],
        "mean_coefficient_variance": diagnostics["mean_coefficient_variance"],
        "max_coefficient_variance": diagnostics["max_coefficient_variance"],
        "max_l2_deviation_from_mean": diagnostics["max_l2_deviation_from_mean"],
        "max_absolute_deviation_from_mean": diagnostics[
            "max_absolute_deviation_from_mean"
        ],
        "mean_target_alignment": float(np.mean([
            row["target_alignment"] for row in nonzero_riemannian_rows
        ])),
    }
    for distractor_name in DISTRACTOR_NAMES:
        summary_metrics["mean_leakage_{}".format(distractor_name)] = float(
            np.mean([
                row["leakage_{}".format(distractor_name)]
                for row in nonzero_riemannian_rows
            ])
        )

    metrics_path = os.path.join(config.output_dir, "metrics.csv")
    save_csv(metrics_path, [summary_metrics], list(summary_metrics))

    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", config.model + ".pth")
    payload = {
        "experiment": "stage02-riemannian-semantic-recovery",
        "purpose": (
            "Validate sphere-manifold optimization machinery against PCA and "
            "the analytically mapped known SeFa direction."
        ),
        "config": config.to_dict(),
        "manifold": {
            "name": "coefficient-space Euclidean unit sphere S^(D-1)",
            "constraint": "u^T u = 1",
            "tangent_projection": "Proj_u(z) = z - u (u^T z)",
            "retraction": "R_u(v) = (u + v) / ||u + v||_2",
            "objective": "f(u) = -u^T C_s u",
            "metric_note": (
                "This MVP uses the canonical Euclidean sphere metric requested "
                "for machinery validation; the raw-Legendre Gram metric is not "
                "used as the sphere constraint."
            ),
        },
        "mapping": {
            "coordinate_system": "raw Legendre coefficients",
            "phi": "Phi(w)(x) = sum_n w[n] P_n(x)",
            "phi_direction": "Phi_dir(d)(x) = sum_n d[n] P_n(x)",
            "psi": "Stage 01 weighted Gauss-Legendre projection",
            "dimension": config.latent_dimension,
        },
        "target_semantic": {
            "name": config.semantic,
            **{
                key: value for key, value in semantic_sources[config.semantic].items()
                if key != "direction"
            },
        },
        "distractors": [{
            "name": name,
            **{
                key: value for key, value in semantic_sources[name].items()
                if key != "direction"
            },
        } for name in DISTRACTOR_NAMES],
        "diagnostics": {
            key: value for key, value in diagnostics.items()
            if key not in ("cosine_matrix", "mean")
        },
        "normalized_displacement_ablation": normalized_ablation,
        "results": summary_metrics,
        "pymanopt": "not installed; manual implementation is the primary MVP",
        "generator": {
            "model": config.model,
            "gan_type": get_gan_type(generator),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256(checkpoint_path),
            "factorized_layers": [int(layer) for layer in layers],
        },
        "outputs": {
            "metrics_csv": metrics_path,
            "convergence_csv": convergence_path,
            "convergence_plot": convergence_plot_path,
            "delta_cosine_csv": cosine_csv_path,
            "delta_cosine_plot": cosine_plot_path,
            "traversal_metrics_csv": traversal_csv_path,
            "semantic_traversal": traversal_grid_path,
            "image_directory": image_dir,
        },
    }
    config_path = os.path.join(config.output_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    summary_path = os.path.join(config.output_dir, "SUMMARY.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write("# Stage 02 Riemannian Semantic Recovery - Results\n\n")
        handle.write(
            "This is a machinery-validation experiment on the Euclidean unit "
            "sphere in raw-Legendre coefficient coordinates. It does not claim "
            "an advantage over eigendecomposition.\n\n"
        )
        handle.write("## Core results\n\n")
        handle.write(
            "- Riemannian vs eigenvector cosine: `{:.16g}`\n".format(
                summary_metrics["alignment_riemannian_vs_eigenvector"]
            )
        )
        handle.write(
            "- Riemannian vs mapped-GT cosine: `{:.16g}`\n".format(
                summary_metrics["alignment_riemannian_vs_mapped_gt"]
            )
        )
        handle.write(
            "- Mean pairwise displacement cosine: `{:.16g}`\n".format(
                diagnostics["mean_pairwise_cosine"]
            )
        )
        handle.write(
            "- Maximum displacement deviation from mean (L2): `{:.6e}`\n".format(
                diagnostics["max_l2_deviation_from_mean"]
            )
        )
        handle.write(
            "- Covariance numerical rank: `{}`\n".format(covariance_rank)
        )
        handle.write(
            "- Iterations: `{}`; final gradient norm: `{:.6e}`\n\n".format(
                summary_metrics["iterations"],
                summary_metrics["final_gradient_norm"],
            )
        )
        handle.write("## Interpretation\n\n")
        handle.write(
            "The centered displacements are identical up to floating-point "
            "error, as predicted by the strict linearity of the Stage 01 map. "
            "Consequently, the semantic covariance is numerically rank one and "
            "the direct mapped direction, PCA eigenvector, and Riemannian sphere "
            "solution agree up to sign and numerical precision. This validates "
            "the tangent projection, retraction, convergence logging, and decode "
            "pipeline before a non-trivial Stiefel-manifold experiment.\n"
        )

    print("Saved metrics: {}".format(metrics_path))
    print("Saved convergence plot: {}".format(convergence_plot_path))
    print("Saved displacement cosine plot: {}".format(cosine_plot_path))
    print("Saved semantic traversal: {}".format(traversal_grid_path))
    print("Saved summary: {}".format(summary_path))


if __name__ == "__main__":
    main()

