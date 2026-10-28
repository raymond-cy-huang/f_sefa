"""Discover anonymous residual semantic candidates with Hilbert+sinh+RO+J_G."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pymanopt
import torch
import torch.nn.functional as functional
from numpy.polynomial import legendre
from PIL import Image, ImageDraw


EXPERIMENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = EXPERIMENT_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils import factorize_weight, get_gan_type, load_generator  # noqa: E402


KNOWN_SEMANTICS = (
    ("gender", 1),
    ("age", 3),
    ("pose", 7),
    ("glasses", 10),
    ("smile", 18),
)
KNOWN_RANKS = tuple(rank for _, rank in KNOWN_SEMANTICS)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="e4e_ffhq_encode")
    parser.add_argument("--search-rank-count", type=int, default=128)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=[11, 29, 47])
    parser.add_argument("--heldout-seeds", type=int, nargs="+", default=[71, 89, 107])
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--response-resolution", type=int, default=32)
    parser.add_argument(
        "--feature-mode",
        choices=("rgb_centered", "face_edges"),
        default="face_edges",
    )
    parser.add_argument(
        "--nuisance-weight",
        type=float,
        default=0.0,
        help=(
            "Penalty on coherent low-frequency RGB/illumination response. "
            "The nuisance Gram matrix is trace-matched to structural energy."
        ),
    )
    parser.add_argument("--direction-batch", type=int, default=2)
    parser.add_argument("--lambdas", type=float, nargs="+", default=[-4, -2, 0, 2, 4])
    parser.add_argument("--trunc-psi", type=float, default=0.7)
    parser.add_argument("--trunc-layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "outputs"))
    values = parser.parse_args()
    if values.search_rank_count <= max(KNOWN_RANKS):
        parser.error("search-rank-count must include every known exclusion rank")
    if values.num_candidates <= 0:
        parser.error("num-candidates must be positive")
    if values.alpha < 0 or values.epsilon <= 0 or values.nuisance_weight < 0:
        parser.error("alpha must be non-negative and epsilon positive")
    if not values.train_seeds or not values.heldout_seeds:
        parser.error("train and heldout seeds must be non-empty")
    if 0.0 not in values.lambdas:
        parser.error("lambdas must include zero")
    return values


def build_legendre_transform(dimension: int) -> np.ndarray:
    nodes, weights = legendre.leggauss(dimension)
    orders = np.arange(dimension, dtype=np.float64)
    norms = 2.0 / (2.0 * orders + 1.0)
    basis = legendre.legvander(nodes, dimension - 1)
    return np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]


def chart_forward(w: np.ndarray, transform: np.ndarray, alpha: float) -> np.ndarray:
    coordinate = transform @ w
    if alpha == 0.0:
        return coordinate
    return np.sinh(alpha * coordinate) / alpha


def chart_inverse(h: np.ndarray, transform: np.ndarray, alpha: float) -> np.ndarray:
    coordinate = h if alpha == 0.0 else np.arcsinh(alpha * h) / alpha
    return transform.T @ coordinate


def inverse_chart_tangent(
    w: np.ndarray, q: np.ndarray, transform: np.ndarray, alpha: float
) -> np.ndarray:
    coordinate = transform @ w
    scales = np.ones_like(coordinate) if alpha == 0.0 else np.cosh(alpha * coordinate)
    return transform.T @ (q / scales)


def sample_w(generator, seeds, trunc_psi, trunc_layers) -> np.ndarray:
    device = next(generator.parameters()).device
    points = []
    for seed in seeds:
        rng = torch.Generator(device=device).manual_seed(int(seed))
        z = torch.randn(1, generator.z_space_dim, generator=rng, device=device)
        with torch.no_grad():
            w = generator.mapping(z)["w"]
            w = generator.truncation(
                w, trunc_psi=trunc_psi, trunc_layers=trunc_layers)
        if w.ndim == 3:
            w = w[:, 0, :]
        points.append(w[0].detach().cpu().numpy().astype(np.float64))
    return np.stack(points)


def synthesize(generator, points: np.ndarray, resolution: int) -> torch.Tensor:
    device = next(generator.parameters()).device
    tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
    wp = tensor[:, None, :].expand(-1, generator.num_layers, -1)
    images = generator.synthesis(wp)["image"]
    if images.shape[-2:] != (resolution, resolution):
        images = functional.interpolate(
            images,
            size=(resolution, resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
    return images


def response_features(images: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "rgb_centered":
        centered = images - images.mean(dim=(-2, -1), keepdim=True)
        return centered.flatten(1) / np.sqrt(centered[0].numel())
    if mode == "lowfreq_rgb":
        # Explicit nuisance representation: coarse chromatic and illumination
        # fields. Unlike simply omitting colour from the objective, response in
        # this channel is actively penalized by the generalized denominator.
        low = functional.adaptive_avg_pool2d(images, (8, 8))
        return low.flatten(1) / np.sqrt(low[0].numel())
    if mode != "face_edges":
        raise ValueError("Unsupported feature mode: {}".format(mode))

    gray = (
        0.2989 * images[:, 0:1]
        + 0.5870 * images[:, 1:2]
        + 0.1140 * images[:, 2:3]
    )
    height, width = gray.shape[-2:]
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=gray.device, dtype=gray.dtype),
        torch.linspace(-1.0, 1.0, width, device=gray.device, dtype=gray.dtype),
        indexing="ij",
    )
    # Fixed, label-free face-centered window. It suppresses background response
    # without using a segmentation model or any semantic attribute labels.
    mask = torch.exp(-0.5 * ((xx / 0.62) ** 2 + (yy / 0.78) ** 2))
    mask = mask[None, None]
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=gray.device,
        dtype=gray.dtype,
    )[None, None] / 8.0
    sobel_y = sobel_x.transpose(-1, -2)
    gradient_x = functional.conv2d(gray, sobel_x, padding=1)
    gradient_y = functional.conv2d(gray, sobel_y, padding=1)
    high_pass = gray - functional.avg_pool2d(gray, 5, stride=1, padding=2)
    features = torch.cat(
        [mask * gradient_x, mask * gradient_y, mask * high_pass], dim=1)
    return features.flatten(1) / np.sqrt(features[0].numel())


@torch.no_grad()
def finite_difference_matrix(
    generator,
    point: np.ndarray,
    directions: np.ndarray,
    epsilon: float,
    resolution: int,
    direction_batch: int,
    feature_mode: str,
) -> np.ndarray:
    columns = []
    for start in range(0, directions.shape[1], direction_batch):
        block = directions[:, start:start + direction_batch].T
        plus = point[None, :] + epsilon * block
        minus = point[None, :] - epsilon * block
        paired = np.concatenate([plus, minus], axis=0)
        features = response_features(
            synthesize(generator, paired, resolution), feature_mode)
        count = len(block)
        derivative = (features[:count] - features[count:]) / (2.0 * epsilon)
        columns.append(derivative.cpu().numpy().T.astype(np.float64))
    return np.concatenate(columns, axis=1)


def residualize(matrix: np.ndarray, known: np.ndarray) -> np.ndarray:
    gram = known.T @ known
    ridge = max(float(np.trace(gram)) / max(len(gram), 1), 1e-12) * 1e-8
    coefficients = np.linalg.solve(
        gram + ridge * np.eye(gram.shape[0]), known.T @ matrix)
    return matrix - known @ coefficients


def inverse_sqrt_spd(matrix: np.ndarray) -> np.ndarray:
    values, vectors = np.linalg.eigh((matrix + matrix.T) / 2.0)
    floor = max(float(values[-1]) * 1e-10, 1e-12)
    values = np.maximum(values, floor)
    return (vectors / np.sqrt(values)[None, :]) @ vectors.T


def optimize_stiefel(operator: np.ndarray, count: int, seed: int):
    dimension = operator.shape[0]
    manifold = pymanopt.manifolds.Stiefel(dimension, count)
    matrix = torch.as_tensor(operator, dtype=torch.float64)
    weights = torch.diag(torch.linspace(count, 1.0, count, dtype=torch.float64))

    @pymanopt.function.pytorch(manifold)
    def cost(point):
        return -torch.trace(weights @ point.T @ matrix @ point)

    problem = pymanopt.Problem(manifold, cost)
    rng = np.random.default_rng(seed)
    initial, _ = np.linalg.qr(rng.normal(size=(dimension, count)))
    optimizer = pymanopt.optimizers.ConjugateGradient(
        max_iterations=250, min_gradient_norm=1e-8, verbosity=1)
    result = optimizer.run(problem, initial_point=initial)
    point = np.asarray(result.point, dtype=np.float64)
    rayleigh = np.diag(point.T @ operator @ point)
    order = np.argsort(rayleigh)[::-1]
    return point[:, order], rayleigh[order], result


@torch.no_grad()
def decode_uint8(generator, point: np.ndarray, resolution: int = 256) -> np.ndarray:
    image = synthesize(generator, point[None, :], resolution)[0].clamp(-1, 1)
    image = ((image + 1.0) * 127.5).round().byte()
    return image.permute(1, 2, 0).cpu().numpy()


def make_grid(rows, lambdas, path: Path, title: str, thumb: int = 160):
    label_width = 140
    title_height = 32
    header_height = 30
    row_height = thumb + 4
    canvas = Image.new(
        "RGB",
        (label_width + len(lambdas) * thumb,
         title_height + header_height + len(rows) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), title, fill="black")
    for column, value in enumerate(lambdas):
        draw.text(
            (label_width + column * thumb + 5, title_height + 7),
            "lambda {:+g}".format(value), fill="black")
    offset = title_height + header_height
    for row_index, (label, images) in enumerate(rows):
        top = offset + row_index * row_height
        draw.text((8, top + thumb // 2), label, fill="black")
        for column, array in enumerate(images):
            image = Image.fromarray(array).resize((thumb, thumb), Image.BILINEAR)
            canvas.paste(image, (label_width + column * thumb, top))
    canvas.save(path)


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir = output_dir / "images"
    image_dir.mkdir(exist_ok=True)

    generator = load_generator(args.model)
    generator.eval()
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator")
    dimension = int(generator.w_space_dim)
    transform = build_legendre_transform(dimension)

    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    boundaries = np.real(boundaries).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    ranked = boundaries[order]
    known_w = ranked[list(KNOWN_RANKS)].T
    known_w, _ = np.linalg.qr(known_w)
    search_ranks = [
        rank for rank in range(args.search_rank_count) if rank not in KNOWN_RANKS
    ]
    search_w = ranked[search_ranks].T
    search_w, _ = np.linalg.qr(search_w)

    train_points = sample_w(
        generator, args.train_seeds, args.trunc_psi, args.trunc_layers)
    mean_point = train_points.mean(axis=0)
    mean_coordinate = transform @ mean_point
    mean_scales = (
        np.ones_like(mean_coordinate) if args.alpha == 0.0
        else np.cosh(args.alpha * mean_coordinate)
    )
    mean_jacobian = mean_scales[:, None] * transform
    search_h, _ = np.linalg.qr(mean_jacobian @ search_w)

    residual_matrices = []
    nuisance_matrices = []
    local_search_bases = []
    for sample_id, point in enumerate(train_points):
        local_search = np.column_stack([
            inverse_chart_tangent(point, search_h[:, index], transform, args.alpha)
            for index in range(search_h.shape[1])
        ])
        local_search_bases.append(local_search)
        search_response = finite_difference_matrix(
            generator, point, local_search, args.epsilon,
            args.response_resolution, args.direction_batch, args.feature_mode)
        known_response = finite_difference_matrix(
            generator, point, known_w, args.epsilon,
            args.response_resolution, args.direction_batch, args.feature_mode)
        residual = residualize(search_response, known_response)
        residual_matrices.append(residual)
        if args.nuisance_weight > 0.0:
            nuisance_matrices.append(finite_difference_matrix(
                generator, point, local_search, args.epsilon,
                args.response_resolution, args.direction_batch, "lowfreq_rgb"))
        print(
            "sample {} residual_fraction={:.6f}".format(
                sample_id,
                np.linalg.norm(residual) / max(np.linalg.norm(search_response), 1e-12),
            ),
            flush=True,
        )

    energy = sum(matrix.T @ matrix for matrix in residual_matrices) / len(residual_matrices)
    mean_response = sum(residual_matrices) / len(residual_matrices)
    coherence = mean_response.T @ mean_response
    denominator = energy.copy()
    nuisance_trace_ratio = 0.0
    if nuisance_matrices:
        nuisance_energy = sum(
            matrix.T @ matrix for matrix in nuisance_matrices
        ) / len(nuisance_matrices)
        energy_trace = max(float(np.trace(energy)), 1e-12)
        nuisance_trace = max(float(np.trace(nuisance_energy)), 1e-12)
        nuisance_energy *= energy_trace / nuisance_trace
        denominator += args.nuisance_weight * nuisance_energy
        nuisance_trace_ratio = float(np.trace(nuisance_energy) / energy_trace)
    ridge = max(float(np.trace(denominator)) / len(denominator), 1e-12) * 1e-4
    inverse_sqrt = inverse_sqrt_spd(
        denominator + ridge * np.eye(len(denominator)))
    operator = inverse_sqrt @ coherence @ inverse_sqrt
    operator = (operator + operator.T) / 2.0

    optimized, scores, result = optimize_stiefel(
        operator, args.num_candidates, args.seed)
    coefficient_matrix = inverse_sqrt @ optimized

    candidate_q = []
    for index in range(args.num_candidates):
        q = search_h @ coefficient_matrix[:, index]
        tangent_norms = [
            np.linalg.norm(inverse_chart_tangent(point, q, transform, args.alpha))
            for point in train_points
        ]
        q = q / max(float(np.mean(tangent_norms)), 1e-12)
        candidate_q.append(q)
    candidate_q = np.column_stack(candidate_q)

    metric_rows = []
    for index in range(args.num_candidates):
        response_norms = [
            float(np.linalg.norm(matrix @ coefficient_matrix[:, index]))
            for matrix in residual_matrices
        ]
        local_cosines = []
        for point in train_points:
            tangent = inverse_chart_tangent(
                point, candidate_q[:, index], transform, args.alpha)
            tangent /= max(np.linalg.norm(tangent), 1e-12)
            local_cosines.append(float(np.max(np.abs(known_w.T @ tangent))))
        metric_rows.append({
            "candidate": index,
            "coherence_score": float(scores[index]),
            "mean_residual_response_norm": float(np.mean(response_norms)),
            "std_residual_response_norm": float(np.std(response_norms)),
            "max_abs_latent_cosine_with_known_mean": float(np.mean(local_cosines)),
        })

    heldout_points = sample_w(
        generator, args.heldout_seeds, args.trunc_psi, args.trunc_layers)
    heldout_known_response_cosines = [[] for _ in range(args.num_candidates)]
    heldout_known_by_semantic = [
        {name: [] for name, _ in KNOWN_SEMANTICS}
        for _ in range(args.num_candidates)
    ]
    for sample_id, (seed, point) in enumerate(zip(args.heldout_seeds, heldout_points)):
        rows = []
        h0 = chart_forward(point, transform, args.alpha)
        heldout_directions = np.column_stack([
            inverse_chart_tangent(
                point, candidate_q[:, candidate], transform, args.alpha)
            for candidate in range(args.num_candidates)
        ])
        heldout_directions /= np.maximum(
            np.linalg.norm(heldout_directions, axis=0, keepdims=True), 1e-12)
        candidate_response = finite_difference_matrix(
            generator, point, heldout_directions, args.epsilon,
            args.response_resolution, args.direction_batch, args.feature_mode)
        known_response = finite_difference_matrix(
            generator, point, known_w, args.epsilon,
            args.response_resolution, args.direction_batch, args.feature_mode)
        candidate_response /= np.maximum(
            np.linalg.norm(candidate_response, axis=0, keepdims=True), 1e-12)
        known_response /= np.maximum(
            np.linalg.norm(known_response, axis=0, keepdims=True), 1e-12)
        response_cosines = np.abs(candidate_response.T @ known_response)
        for candidate in range(args.num_candidates):
            heldout_known_response_cosines[candidate].append(
                float(np.max(response_cosines[candidate])))
            for semantic_index, (name, _) in enumerate(KNOWN_SEMANTICS):
                heldout_known_by_semantic[candidate][name].append(
                    float(response_cosines[candidate, semantic_index]))
        for candidate in range(args.num_candidates):
            q = candidate_q[:, candidate]
            start_tangent = inverse_chart_tangent(point, q, transform, args.alpha)
            q_scaled = q / max(np.linalg.norm(start_tangent), 1e-12)
            images = []
            for value in args.lambdas:
                edited = chart_inverse(
                    h0 + float(value) * q_scaled, transform, args.alpha)
                image = decode_uint8(generator, edited)
                images.append(image)
                Image.fromarray(image).save(
                    image_dir / (
                        "seed_{:03d}_candidate_{:02d}_lambda_{:+g}.png".format(
                            seed, candidate, value
                        ).replace("+", "p").replace("-", "m")
                    )
                )
            rows.append(("candidate_{:02d}".format(candidate), images))
        make_grid(
            rows, args.lambdas,
            output_dir / "heldout_seed_{:03d}_candidates.png".format(seed),
            "Anonymous residual candidates / held-out seed {}".format(seed),
        )

    for candidate, values in enumerate(heldout_known_response_cosines):
        metric_rows[candidate][
            "heldout_max_abs_response_cosine_with_known_mean"
        ] = float(np.mean(values))
        metric_rows[candidate][
            "heldout_max_abs_response_cosine_with_known_max"
        ] = float(np.max(values))
        for name, _ in KNOWN_SEMANTICS:
            metric_rows[candidate][
                "heldout_abs_response_cosine_{}_mean".format(name)
            ] = float(np.mean(heldout_known_by_semantic[candidate][name]))

    np.savez_compressed(
        output_dir / "discovered_hilbert_directions.npz",
        candidate_q=candidate_q,
        coefficient_matrix=coefficient_matrix,
        search_h=search_h,
        known_w=known_w,
        train_points=train_points,
        heldout_points=heldout_points,
        alpha=np.asarray(args.alpha),
    )

    with (output_dir / "candidate_metrics.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)

    config = {
        "experiment": "03_unknown_semantic_discovery",
        "model": args.model,
        "known_exclusion_ranks": list(KNOWN_RANKS),
        "known_labels_not_used_for_candidate_naming": True,
        "search_rank_count": args.search_rank_count,
        "search_ranks_after_exclusion": search_ranks,
        "num_candidates": args.num_candidates,
        "train_seeds": args.train_seeds,
        "heldout_seeds": args.heldout_seeds,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "response_resolution": args.response_resolution,
        "feature_mode": args.feature_mode,
        "nuisance_weight": args.nuisance_weight,
        "nuisance_feature_mode": (
            "lowfreq_rgb" if args.nuisance_weight > 0.0 else None),
        "nuisance_trace_ratio_after_matching": nuisance_trace_ratio,
        "lambdas": args.lambdas,
        "stiefel_iterations": int(result.iterations),
        "stiefel_stopping_criterion": result.stopping_criterion,
        "factorization_layers": [int(value) for value in layers],
        "interpretation_boundary": (
            "Candidates optimize coherent residual low-resolution RGB response. "
            "They are not semantics until the traversals are interpreted and validated."
        ),
    }
    (output_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print("Saved unknown-semantic candidates to {}".format(output_dir))


if __name__ == "__main__":
    main()
