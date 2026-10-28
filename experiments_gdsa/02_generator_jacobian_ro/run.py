"""Combine generator directional-Jacobian measurements with local GDSA RO."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from importlib.metadata import version
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

from utils import factorize_weight, load_generator  # noqa: E402


SEMANTIC_RANKS = {
    "gender": 1,
    "age": 3,
    "pose": 7,
    "glasses": 10,
    "smile": 18,
}


@dataclass(frozen=True)
class Config:
    model: str = "e4e_ffhq_encode"
    semantic: str = "smile"
    num_seeds: int = 2
    seed: int = 42
    alphas: tuple[float, ...] = (0.0, 0.5, 1.0)
    target_tau: float = 0.5
    num_steps: int = 5
    jacobian_epsilon: float = 0.01
    response_resolution: int = 64
    trunc_psi: float = 0.7
    trunc_layers: int = 8
    gpu_id: str = "0"
    output_dir: str = str(EXPERIMENT_DIR / "outputs")


def parse_args() -> Config:
    defaults = Config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--semantic", choices=SEMANTIC_RANKS, default=defaults.semantic)
    parser.add_argument("--num-seeds", type=int, default=defaults.num_seeds)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--alphas", type=float, nargs="+", default=list(defaults.alphas))
    parser.add_argument("--target-tau", type=float, default=defaults.target_tau)
    parser.add_argument("--num-steps", type=int, default=defaults.num_steps)
    parser.add_argument(
        "--jacobian-epsilon", type=float, default=defaults.jacobian_epsilon
    )
    parser.add_argument(
        "--response-resolution", type=int, default=defaults.response_resolution
    )
    parser.add_argument("--trunc-psi", type=float, default=defaults.trunc_psi)
    parser.add_argument("--trunc-layers", type=int, default=defaults.trunc_layers)
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    if values.num_seeds <= 0 or values.num_steps <= 0:
        parser.error("num-seeds and num-steps must be positive")
    if values.target_tau <= 0 or values.jacobian_epsilon <= 0:
        parser.error("target-tau and jacobian-epsilon must be positive")
    if values.response_resolution <= 0:
        parser.error("response-resolution must be positive")
    if any(alpha < 0 for alpha in values.alphas):
        parser.error("alphas must be non-negative")
    return Config(
        model=values.model,
        semantic=values.semantic,
        num_seeds=values.num_seeds,
        seed=values.seed,
        alphas=tuple(values.alphas),
        target_tau=values.target_tau,
        num_steps=values.num_steps,
        jacobian_epsilon=values.jacobian_epsilon,
        response_resolution=values.response_resolution,
        trunc_psi=values.trunc_psi,
        trunc_layers=values.trunc_layers,
        gpu_id=values.gpu_id,
        output_dir=values.output_dir,
    )


def build_legendre_transform(dimension: int) -> np.ndarray:
    nodes, weights = legendre.leggauss(dimension)
    orders = np.arange(dimension, dtype=np.float64)
    norms = 2.0 / (2.0 * orders + 1.0)
    basis = legendre.legvander(nodes, dimension - 1)
    return np.sqrt(weights)[:, None] * basis / np.sqrt(norms)[None, :]


def metric_scales(transform: np.ndarray, point: np.ndarray, alpha: float) -> np.ndarray:
    coordinates = transform @ point
    if alpha == 0.0:
        return np.ones_like(coordinates)
    return np.cosh(alpha * coordinates) ** 2


def metric_inner(
    transform: np.ndarray,
    scales: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
) -> float:
    left_coordinates = transform @ left
    right_coordinates = transform @ right
    return float(np.dot(scales * left_coordinates, right_coordinates))


def inverse_metric_apply(
    transform: np.ndarray, scales: np.ndarray, vector: np.ndarray
) -> np.ndarray:
    return transform.T @ ((transform @ vector) / scales)


def matched_progress_direction(
    transform: np.ndarray,
    scales: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    inverse_target = inverse_metric_apply(transform, scales, target)
    progress = float(np.dot(target, inverse_target))
    if progress <= np.finfo(np.float64).eps:
        raise ValueError("Target direction has zero inverse-metric norm")
    return inverse_target / progress


def extract_target_direction(generator, semantic: str):
    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    boundaries = np.real(boundaries).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    rank = SEMANTIC_RANKS[semantic]
    source_index = int(order[rank])
    direction = boundaries[source_index].reshape(-1)
    direction /= np.linalg.norm(direction)
    metadata = {
        "semantic": semantic,
        "post_annotated_eigen_rank": rank,
        "source_eigen_index": source_index,
        "eigenvalue": float(eigenvalues[source_index]),
        "layers": [int(value) for value in layers],
        "polarity_validated": False,
    }
    return direction, metadata


def sample_w(generator, count: int, config: Config) -> torch.Tensor:
    device = next(generator.parameters()).device
    random_generator = torch.Generator(device=device)
    random_generator.manual_seed(config.seed)
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
    if tuple(w.shape) != (count, generator.w_space_dim):
        raise ValueError(f"Unexpected W shape: {tuple(w.shape)}")
    return w.detach().cpu().float()


def decode_low(generator, points: torch.Tensor, resolution: int) -> torch.Tensor:
    device = next(generator.parameters()).device
    points = points.to(device=device, dtype=torch.float32)
    wp = points[:, None, :].expand(-1, generator.num_layers, -1)
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


@torch.no_grad()
def directional_jacobian(
    generator,
    point: np.ndarray,
    direction: np.ndarray,
    epsilon: float,
    resolution: int,
) -> tuple[float, np.ndarray]:
    point_tensor = torch.as_tensor(point, dtype=torch.float32).reshape(1, -1)
    direction_tensor = torch.as_tensor(direction, dtype=torch.float32).reshape(1, -1)
    paired = torch.cat(
        [point_tensor + epsilon * direction_tensor, point_tensor - epsilon * direction_tensor],
        dim=0,
    )
    plus, minus = torch.chunk(decode_low(generator, paired, resolution), 2, dim=0)
    response = (plus - minus) / (2.0 * epsilon)
    rms = float(torch.sqrt(torch.mean(response.square())).cpu())
    return rms, response.flatten().cpu().numpy()


@torch.no_grad()
def decode_uint8(generator, point: np.ndarray) -> np.ndarray:
    tensor = torch.as_tensor(point, dtype=torch.float32).reshape(1, -1)
    image = decode_low(generator, tensor, 256)[0].clamp(-1.0, 1.0)
    image = ((image + 1.0) * 127.5).round().byte()
    return image.permute(1, 2, 0).cpu().numpy()


def response_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator <= np.finfo(np.float64).eps:
        return float("nan")
    return float(np.dot(left, right) / denominator)


def pymanopt_direction_check(
    transform: np.ndarray,
    scales: np.ndarray,
    target: np.ndarray,
    analytical: np.ndarray,
    seed: int,
) -> dict:
    np.random.seed(seed)
    inverse_sqrt = transform.T @ ((1.0 / np.sqrt(scales))[:, None] * transform)
    matrix = torch.as_tensor(inverse_sqrt, dtype=torch.float64)
    target_tensor = torch.as_tensor(target, dtype=torch.float64)
    manifold = pymanopt.manifolds.Sphere(len(target))

    @pymanopt.function.pytorch(manifold)
    def cost(point):
        return -(target_tensor @ (matrix @ point))

    problem = pymanopt.Problem(manifold, cost)
    optimizer = pymanopt.optimizers.ConjugateGradient(
        max_iterations=200,
        min_gradient_norm=1e-9,
        verbosity=0,
    )
    result = optimizer.run(problem)
    optimized = inverse_sqrt @ np.asarray(result.point)
    optimized /= np.sqrt(metric_inner(transform, scales, optimized, optimized))

    analytical_unit = analytical / np.sqrt(
        metric_inner(transform, scales, analytical, analytical)
    )
    cosine_value = abs(
        float(
            np.dot(optimized, analytical_unit)
            / (np.linalg.norm(optimized) * np.linalg.norm(analytical_unit))
        )
    )
    return {
        "direction_cosine_absolute": cosine_value,
        "iterations": int(result.iterations),
        "stopping_criterion": result.stopping_criterion,
    }


def build_paths(
    point: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    alpha: float,
    target_tau: float,
    num_steps: int,
):
    delta_tau = target_tau / num_steps
    fixed_scales = metric_scales(transform, point, alpha)
    fixed_direction = matched_progress_direction(transform, fixed_scales, target)
    euclidean_direction = target / float(np.dot(target, target))
    paths = {}
    directions = {}
    energies = {}
    for method in ("B1_euclidean", "B2_fixed_metric", "B3_dynamic_metric"):
        current = point.copy()
        path = [current.copy()]
        used_directions = []
        energy = 0.0
        for _ in range(num_steps):
            current_scales = metric_scales(transform, current, alpha)
            if method == "B1_euclidean":
                direction = euclidean_direction
            elif method == "B2_fixed_metric":
                direction = fixed_direction
            else:
                direction = matched_progress_direction(
                    transform, current_scales, target
                )
            energy += (delta_tau**2) * metric_inner(
                transform, current_scales, direction, direction
            )
            used_directions.append(direction.copy())
            current = current + delta_tau * direction
            path.append(current.copy())
        paths[method] = path
        directions[method] = used_directions
        energies[method] = energy
    return paths, directions, energies


def save_grid(path: Path, rows: list[tuple[str, list[np.ndarray]]]) -> None:
    if not rows:
        return
    thumb = 256
    label_width = 190
    columns = max(len(images) for _, images in rows)
    canvas = Image.new("RGB", (label_width + columns * thumb, len(rows) * thumb), "white")
    draw = ImageDraw.Draw(canvas)
    for row_index, (label, images) in enumerate(rows):
        draw.text((8, row_index * thumb + 10), label, fill="black")
        for column, image in enumerate(images):
            tile = Image.fromarray(image).resize((thumb, thumb), Image.Resampling.LANCZOS)
            canvas.paste(tile, (label_width + column * thumb, row_index * thumb))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    config = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    generator = load_generator(config.model)
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    generator.eval()

    target, semantic_metadata = extract_target_direction(generator, config.semantic)
    dimension = len(target)
    transform = build_legendre_transform(dimension)
    points = sample_w(generator, config.num_seeds, config).numpy().astype(np.float64)

    metric_rows = []
    ro_rows = []
    grid_rows = []
    for seed_index, point in enumerate(points):
        for alpha in config.alphas:
            paths, directions, energies = build_paths(
                point,
                target,
                transform,
                alpha,
                config.target_tau,
                config.num_steps,
            )
            base_rms, base_response = directional_jacobian(
                generator,
                point,
                target,
                config.jacobian_epsilon,
                config.response_resolution,
            )

            if seed_index == 0:
                scales = metric_scales(transform, point, alpha)
                analytical = matched_progress_direction(transform, scales, target)
                check = pymanopt_direction_check(
                    transform,
                    scales,
                    target,
                    analytical,
                    config.seed + int(round(alpha * 1000)),
                )
                ro_rows.append({"seed_index": seed_index, "alpha": alpha, **check})

            for method, path in paths.items():
                start_direction = directions[method][0]
                end_direction = directions[method][-1]
                start_rms, start_response = directional_jacobian(
                    generator,
                    path[0],
                    start_direction,
                    config.jacobian_epsilon,
                    config.response_resolution,
                )
                end_rms, end_response = directional_jacobian(
                    generator,
                    path[-1],
                    end_direction,
                    config.jacobian_epsilon,
                    config.response_resolution,
                )
                displacement = path[-1] - point
                target_progress = float(np.dot(target, displacement))
                final_scales = metric_scales(transform, path[-1], alpha)
                metric_rows.append(
                    {
                        "seed_index": seed_index,
                        "alpha": alpha,
                        "method": method,
                        "target_progress": target_progress,
                        "target_progress_error": abs(target_progress - config.target_tau),
                        "latent_displacement_l2": float(np.linalg.norm(displacement)),
                        "path_dynamic_metric_energy": float(energies[method]),
                        "start_direction_l2": float(np.linalg.norm(start_direction)),
                        "end_direction_l2": float(np.linalg.norm(end_direction)),
                        "direction_change_cosine": float(
                            np.dot(start_direction, end_direction)
                            / (
                                np.linalg.norm(start_direction)
                                * np.linalg.norm(end_direction)
                            )
                        ),
                        "start_metric_norm": float(
                            np.sqrt(
                                metric_inner(
                                    transform,
                                    metric_scales(transform, path[0], alpha),
                                    start_direction,
                                    start_direction,
                                )
                            )
                        ),
                        "end_metric_norm": float(
                            np.sqrt(
                                metric_inner(
                                    transform,
                                    final_scales,
                                    end_direction,
                                    end_direction,
                                )
                            )
                        ),
                        "target_generator_jacobian_rms": base_rms,
                        "start_generator_jacobian_rms": start_rms,
                        "end_generator_jacobian_rms": end_rms,
                        "start_response_cosine_to_target": response_cosine(
                            start_response, base_response
                        ),
                        "end_response_cosine_to_start": response_cosine(
                            end_response, start_response
                        ),
                    }
                )

                if seed_index == 0:
                    images = [decode_uint8(generator, value) for value in path]
                    grid_rows.append((f"a={alpha:g} {method}", images))

    write_csv(output_dir / "path_metrics.csv", metric_rows)
    write_csv(output_dir / "pymanopt_checks.csv", ro_rows)
    save_grid(output_dir / "traversal_seed_000.png", grid_rows)

    maximum_progress_error = max(row["target_progress_error"] for row in metric_rows)
    minimum_ro_cosine = min(row["direction_cosine_absolute"] for row in ro_rows)
    status = (
        "PASS"
        if maximum_progress_error <= 1e-10 and minimum_ro_cosine >= 0.999
        else "FAIL"
    )
    summary = {
        "status": status,
        "config": asdict(config),
        "versions": {
            "numpy": np.__version__,
            "torch": torch.__version__,
            "pymanopt": version("pymanopt"),
        },
        "semantic": semantic_metadata,
        "maximum_target_progress_error": maximum_progress_error,
        "minimum_pymanopt_analytical_direction_cosine": minimum_ro_cosine,
        "runtime_seconds": time.perf_counter() - started,
        "interpretation": (
            "Directional generator response and coordinate-metric path comparison; "
            "not an independent image-semantic validation."
        ),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
