"""Compare SEFA semantic changes for the car generator.

This script is a focused visual scanner for `stylegan_car512`. Each sample
gets one overview grid where rows are semantic IDs and columns are traversal
distances. It is intended for quickly finding car semantics such as orientation,
vertical position, zoom, and shape.

Example:
    python experiments/car_semantic_comparison.py --num_samples 3 \
        --semantic_ids 0 1 2 3 4 5 6 7 8 9 \
        --output_dir results/car_semantic_comparison
"""

import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiment_utils.metrics import image_errors
from experiment_utils.metrics import make_comparison_grid
from experiment_utils.metrics import save_image

torch = None
parse_gan_type = None
factorize_weight = None
load_generator = None
postprocess = None
to_tensor = None


def parse_args():
    parser = argparse.ArgumentParser(description="Compare car SEFA semantic traversals.")
    parser.add_argument("--model", default="stylegan_car512")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--semantic_ids", nargs="+", type=int, default=list(range(10)))
    parser.add_argument("--layer_idx", default="0-1")
    parser.add_argument("--start_distance", type=float, default=-3.0)
    parser.add_argument("--end_distance", type=float, default=3.0)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output_dir", default="results/car_semantic_comparison")
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--latent_path", default=None)
    parser.add_argument("--trunc_psi", type=float, default=0.7)
    parser.add_argument("--trunc_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", default="0")
    return parser.parse_args()


def generator_device(generator):
    try:
        return next(generator.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_random_latents(generator, gan_type, num_samples, args):
    z = torch.randn(num_samples, generator.z_space_dim, device=generator_device(generator))
    if gan_type == "pggan":
        return generator.layer0.pixel_norm(z).detach().cpu().numpy().astype(np.float32)
    if gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
        w = generator.mapping(z)["w"]
        wp = generator.truncation(w, trunc_psi=args.trunc_psi, trunc_layers=args.trunc_layers)
        return wp.detach().cpu().numpy().astype(np.float32)
    raise ValueError(f"Unsupported GAN type `{gan_type}`.")


def load_or_sample_latents(generator, gan_type, args):
    latent_path = args.latent_path
    if latent_path is None:
        candidate = os.path.join(args.latent_dir, f"{args.model}_latents.npy")
        latent_path = candidate if os.path.exists(candidate) else None

    if latent_path:
        latents = np.load(latent_path).astype(np.float32)
        if latents.ndim == 1:
            latents = latents[None, :]
        elif (
            gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]
            and latents.ndim == 2
            and latents.shape == (generator.num_layers, generator.w_space_dim)
        ):
            latents = latents[None, ...]
        return latents[: args.num_samples], latent_path

    return prepare_random_latents(generator, gan_type, args.num_samples, args), "random"


def to_decoder_code(generator, gan_type, latent):
    latent = np.asarray(latent, dtype=np.float32)
    if gan_type == "pggan":
        return latent.reshape(1, -1)
    if gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
        if latent.ndim == 1:
            latent = np.tile(latent[None, :], (generator.num_layers, 1))
        elif latent.ndim == 2 and latent.shape[0] == 1:
            latent = np.tile(latent, (generator.num_layers, 1))
        if latent.ndim != 2 or latent.shape != (generator.num_layers, generator.w_space_dim):
            raise ValueError(
                "StyleGAN latent samples must be [512], [1, 512], or "
                f"[{generator.num_layers}, {generator.w_space_dim}], but got {latent.shape}."
            )
        return latent[None, ...]
    raise ValueError(f"Unsupported GAN type `{gan_type}`.")


def decode_image(generator, gan_type, latent):
    code = to_decoder_code(generator, gan_type, latent)
    with torch.no_grad():
        if gan_type == "pggan":
            image = generator(to_tensor(code))["image"]
        elif gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
            image = generator.synthesis(to_tensor(code))["image"]
        else:
            raise ValueError(f"Unsupported GAN type `{gan_type}`.")
    return postprocess(image)[0]


def normalize(vector, eps=1e-12):
    norm = np.linalg.norm(vector.reshape(-1))
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero semantic direction.")
    return vector / norm


def build_semantic_direction(generator, gan_type, latent_shape, layers, boundary):
    if gan_type == "pggan" or len(latent_shape) == 1:
        return normalize(boundary.reshape(latent_shape))
    if gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
        direction = np.zeros(latent_shape, dtype=np.float64)
        direction[layers, :] = boundary
        return normalize(direction)
    raise ValueError(f"Unsupported GAN type `{gan_type}`.")


def csv_fieldnames():
    return [
        "sample_id",
        "semantic_id",
        "eigen_value",
        "step_id",
        "distance",
        "latent_shape",
        "image_mse_from_original",
        "image_ssim_from_original",
        "image_path",
    ]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)
    if args.step < 2:
        raise ValueError("--step must be at least 2.")

    global torch, parse_gan_type, factorize_weight, load_generator, postprocess, to_tensor
    try:
        import torch as torch_module
        from utils import get_gan_type as parse_gan_type_func
        from utils import factorize_weight as factorize_weight_func
        from utils import load_generator as load_generator_func
        from utils import postprocess as postprocess_func
        from utils import to_tensor as to_tensor_func
    except ImportError as exc:
        raise ImportError("This experiment needs torch and the SEFA runtime dependencies.") from exc

    torch = torch_module
    parse_gan_type = parse_gan_type_func
    factorize_weight = factorize_weight_func
    load_generator = load_generator_func
    postprocess = postprocess_func
    to_tensor = to_tensor_func

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generator = load_generator(args.model)
    gan_type = parse_gan_type(generator)
    latents, latent_source = load_or_sample_latents(generator, gan_type, args)
    layers, boundaries, values = factorize_weight(generator, args.layer_idx)
    order = np.argsort(np.real(values))[::-1]

    max_id = len(order) - 1
    for semantic_id in args.semantic_ids:
        if semantic_id < 0 or semantic_id > max_id:
            raise IndexError(f"semantic_id must be in [0, {max_id}], got {semantic_id}.")

    distances = np.linspace(args.start_distance, args.end_distance, args.step)
    rows = []
    for sample_id, latent in enumerate(latents):
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        original_image = decode_image(generator, gan_type, latent)
        save_image(original_image, os.path.join(sample_dir, "original.png"))

        overview_items = []
        for semantic_id in args.semantic_ids:
            sorted_idx = order[semantic_id]
            boundary = np.real(boundaries[sorted_idx]).astype(np.float64)
            eigen_value = float(np.real(values[sorted_idx]))
            direction = build_semantic_direction(
                generator,
                gan_type,
                latent.shape,
                layers,
                boundary,
            )

            semantic_items = []
            for step_id, distance in enumerate(distances):
                edited = latent.astype(np.float64) + distance * direction
                image = decode_image(generator, gan_type, edited)
                image_path = os.path.join(
                    sample_dir,
                    f"semantic_{semantic_id:03d}_step_{step_id:02d}.png",
                )
                save_image(image, image_path)
                label = f"id{semantic_id} {distance:.1f}"
                semantic_items.append((label, image))
                overview_items.append((label, image))
                metrics = image_errors(original_image, image)
                rows.append({
                    "sample_id": sample_id,
                    "semantic_id": semantic_id,
                    "eigen_value": eigen_value,
                    "step_id": step_id,
                    "distance": float(distance),
                    "latent_shape": json.dumps(list(latent.shape)),
                    "image_mse_from_original": metrics["image_mse"],
                    "image_ssim_from_original": metrics["image_ssim"],
                    "image_path": image_path,
                })

            make_comparison_grid(
                semantic_items,
                os.path.join(sample_dir, f"semantic_{semantic_id:03d}_grid.png"),
                columns=args.step,
            )

        make_comparison_grid(
            overview_items,
            os.path.join(sample_dir, "all_car_semantics_grid.png"),
            columns=args.step,
        )

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model": args.model,
            "gan_type": gan_type,
            "latent_source": latent_source,
            "layer_idx": args.layer_idx,
            "active_layers": layers,
            "semantic_ids": args.semantic_ids,
            "distances": distances.tolist(),
            "results": rows,
        }, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved car semantic grids under: {args.output_dir}")


if __name__ == "__main__":
    main()
