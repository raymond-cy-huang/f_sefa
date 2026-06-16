"""Compare SEFA and explicit-basis semantic paths on StyleGAN car.

This is the car-focused counterpart of `latent_basis_semantic_path.py`.
For each sample it writes one `all_paths_grid.png` where:

    row 1: standard SEFA linear traversal
    row 2: Legendre basis-modulated traversal
    row 3: Chebyshev basis-modulated traversal
    row 4: Hermite basis-modulated traversal
    row 5: Laguerre basis-modulated traversal
    row 6: Fourier basis-modulated traversal

Example:
    python experiments/car_basis_semantic_path.py --num_samples 3 \
        --semantic_id 0 --basis_order 3 --amplitude 8 --step 5 \
        --output_dir results/car_basis_semantic_path
"""

import argparse
import csv
import json
import os
import sys

import numpy as np
from numpy.polynomial import chebyshev
from numpy.polynomial import hermite
from numpy.polynomial import laguerre
from numpy.polynomial import legendre

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
    parser = argparse.ArgumentParser(description="Compare car SEFA traversal with basis paths.")
    parser.add_argument("--model", default="stylegan_car512")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--semantic_id", type=int, default=0)
    parser.add_argument("--layer_idx", default="0-1")
    parser.add_argument(
        "--basis",
        nargs="+",
        default=["legendre", "chebyshev", "hermite", "laguerre", "fourier"],
    )
    parser.add_argument("--basis_order", type=int, default=3)
    parser.add_argument("--amplitude", type=float, default=8.0)
    parser.add_argument("--normalize_basis_coefficients", action="store_true", default=False)
    parser.add_argument("--start_distance", type=float, default=-4.0)
    parser.add_argument("--end_distance", type=float, default=4.0)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--output_dir", default="results/car_basis_semantic_path")
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--latent_path", default=None)
    parser.add_argument("--trunc_psi", type=float, default=0.7)
    parser.add_argument("--trunc_layers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", default="0")
    return parser.parse_args()


def _basis_coeff(order):
    coeff = np.zeros(order + 1, dtype=np.float64)
    coeff[order] = 1.0
    return coeff


def basis_response(name, t, order):
    name = name.lower()
    t = np.asarray(t, dtype=np.float64)
    coeff = _basis_coeff(order)
    if name == "legendre":
        return legendre.legval(t, coeff)
    if name == "chebyshev":
        return chebyshev.chebval(t, coeff)
    if name == "hermite":
        return hermite.hermval(t, coeff)
    if name == "laguerre":
        return laguerre.lagval((t + 1.0) * 2.5, coeff)
    if name == "fourier":
        if order <= 0:
            return np.ones_like(t)
        return np.sin(np.pi * order * t)
    raise ValueError(f"Unsupported basis `{name}`.")


def normalize_to_unit_range(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    min_value = float(np.min(values))
    max_value = float(np.max(values))
    if max_value - min_value < eps:
        raise ValueError("Cannot normalize a constant basis response.")
    return 2.0 * (values - min_value) / (max_value - min_value) - 1.0


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
        raise ValueError("Cannot normalize near-zero semantic direction.")
    return vector / norm


def semantic_direction_for_latent(generator, gan_type, latent_shape, layer_idx, semantic_id):
    layers, boundaries, values = factorize_weight(generator, layer_idx)
    order = np.argsort(np.real(values))[::-1]
    if semantic_id < 0 or semantic_id >= len(order):
        raise IndexError(f"semantic_id must be in [0, {len(order) - 1}].")

    boundary = np.real(boundaries[order[semantic_id]]).astype(np.float64)
    if gan_type == "pggan" or len(latent_shape) == 1:
        direction = boundary.reshape(latent_shape)
    elif gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
        direction = np.zeros(latent_shape, dtype=np.float64)
        direction[layers, :] = boundary
    else:
        raise ValueError(f"Unsupported GAN type `{gan_type}`.")

    return normalize(direction), layers, float(np.real(values[order[semantic_id]]))


def csv_fieldnames():
    return [
        "sample_id",
        "path_type",
        "basis_name",
        "basis_order",
        "semantic_id",
        "step_id",
        "t",
        "coefficient",
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
    direction, layers, eigen_value = semantic_direction_for_latent(
        generator,
        gan_type,
        latents[0].shape,
        args.layer_idx,
        args.semantic_id,
    )

    rows = []
    t_values = np.linspace(-1.0, 1.0, args.step)
    linear_coefficients = np.linspace(args.start_distance, args.end_distance, args.step)

    for sample_id, latent in enumerate(latents):
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_dir, exist_ok=True)
        original_image = decode_image(generator, gan_type, latent)
        save_image(original_image, os.path.join(sample_dir, "original.png"))

        all_path_items = []
        for step_id, coefficient in enumerate(linear_coefficients):
            edited = latent.astype(np.float64) + coefficient * direction
            image = decode_image(generator, gan_type, edited)
            image_path = os.path.join(sample_dir, f"sefa_linear_{step_id:02d}.png")
            save_image(image, image_path)
            all_path_items.append((f"sefa {coefficient:.2f}", image))
            metrics = image_errors(original_image, image)
            rows.append({
                "sample_id": sample_id,
                "path_type": "sefa_linear",
                "basis_name": "",
                "basis_order": "",
                "semantic_id": args.semantic_id,
                "step_id": step_id,
                "t": float(t_values[step_id]),
                "coefficient": float(coefficient),
                "latent_shape": json.dumps(list(latent.shape)),
                "image_mse_from_original": metrics["image_mse"],
                "image_ssim_from_original": metrics["image_ssim"],
                "image_path": image_path,
            })

        for basis_name in args.basis:
            response = basis_response(basis_name, t_values, args.basis_order)
            if args.normalize_basis_coefficients:
                response = normalize_to_unit_range(response)
            coefficients = args.amplitude * response
            for step_id, coefficient in enumerate(coefficients):
                edited = latent.astype(np.float64) + coefficient * direction
                image = decode_image(generator, gan_type, edited)
                image_path = os.path.join(
                    sample_dir,
                    f"basis_{basis_name}_order{args.basis_order}_{step_id:02d}.png",
                )
                save_image(image, image_path)
                all_path_items.append((f"{basis_name} {coefficient:.2f}", image))
                metrics = image_errors(original_image, image)
                rows.append({
                    "sample_id": sample_id,
                    "path_type": "basis_modulated",
                    "basis_name": basis_name,
                    "basis_order": args.basis_order,
                    "semantic_id": args.semantic_id,
                    "step_id": step_id,
                    "t": float(t_values[step_id]),
                    "coefficient": float(coefficient),
                    "latent_shape": json.dumps(list(latent.shape)),
                    "image_mse_from_original": metrics["image_mse"],
                    "image_ssim_from_original": metrics["image_ssim"],
                    "image_path": image_path,
                })

        make_comparison_grid(
            all_path_items,
            os.path.join(sample_dir, "all_paths_grid.png"),
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
            "semantic_id": args.semantic_id,
            "layer_idx": args.layer_idx,
            "active_layers": layers,
            "eigen_value": eigen_value,
            "basis": args.basis,
            "basis_order": args.basis_order,
            "amplitude": args.amplitude,
            "normalize_basis_coefficients": args.normalize_basis_coefficients,
            "results": rows,
        }, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved car basis comparison grids under: {args.output_dir}")


if __name__ == "__main__":
    main()
