"""Latent explicit-basis round-trip feasibility experiment.

This experiment is not semantic editing. It only checks whether the path

    latent code -> explicit orthogonal basis mapping -> inverse mapping -> latent code

can stay numerically and visually close after decoding with the existing SEFA
generator interface. It does not train any model and does not change SEFA's
semantic direction discovery flow.

Example:
    python experiments/latent_basis_roundtrip.py --num_samples 10 --degree 4 \
        --output_dir results/basis_roundtrip
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

from experiment_utils.basis_mapping import available_basis_names
from experiment_utils.basis_mapping import build_basis_matrix
from experiment_utils.basis_mapping import map_latent_slices
from experiment_utils.inverse_mapping import available_inverse_methods
from experiment_utils.inverse_mapping import inverse_map
from experiment_utils.metrics import image_errors
from experiment_utils.metrics import latent_errors
from experiment_utils.metrics import make_comparison_grid
from experiment_utils.metrics import save_image

torch = None
parse_gan_type = None
load_generator = None
postprocess = None
to_tensor = None


def parse_args():
    parser = argparse.ArgumentParser(description="Explicit basis latent round-trip experiment.")
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--output_dir", default="results/basis_roundtrip")
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--latent_path", default=None)
    parser.add_argument("--basis", nargs="+", default=available_basis_names())
    parser.add_argument("--inverse_methods", nargs="+", default=available_inverse_methods())
    parser.add_argument("--trunc_psi", type=float, default=0.7)
    parser.add_argument("--trunc_layers", type=int, default=8)
    parser.add_argument("--ridge_lambda", type=float, default=1e-6)
    parser.add_argument("--cg_tol", type=float, default=1e-10)
    parser.add_argument("--cg_maxiter", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_id", default="0")
    parser.add_argument("--grid_thumb_size", type=int, default=160)
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
    if gan_type in ["stylegan", "stylegan2"]:
        w = generator.mapping(z)["w"]
        wp = generator.truncation(
            w,
            trunc_psi=args.trunc_psi,
            trunc_layers=args.trunc_layers,
        )
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
            gan_type in ["stylegan", "stylegan2"]
            and latents.ndim == 2
            and latents.shape == (generator.num_layers, generator.w_space_dim)
        ):
            latents = latents[None, ...]
        return latents[: args.num_samples], latent_path

    return prepare_random_latents(generator, gan_type, args.num_samples, args), "random"


def to_decoder_code(generator, gan_type, latent):
    """Converts [512], [1, 512], W, or W+ sample latents to decoder input."""

    latent = np.asarray(latent, dtype=np.float32)
    if gan_type == "pggan":
        return latent.reshape(1, -1)
    if gan_type in ["stylegan", "stylegan2"]:
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
        elif gan_type in ["stylegan", "stylegan2"]:
            image = generator.synthesis(to_tensor(code))["image"]
        else:
            raise ValueError(f"Unsupported GAN type `{gan_type}`.")
    return postprocess(image)[0]


def csv_fieldnames():
    return [
        "sample_id",
        "basis_name",
        "inverse_method",
        "latent_shape",
        "high_dim_shape",
        "l1_error",
        "l2_error",
        "relative_l2_error",
        "max_abs_error",
        "image_mse",
        "image_ssim",
    ]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)

    global torch, parse_gan_type, load_generator, postprocess, to_tensor
    try:
        import torch as torch_module
        from models import parse_gan_type as parse_gan_type_func
        from utils import load_generator as load_generator_func
        from utils import postprocess as postprocess_func
        from utils import to_tensor as to_tensor_func
    except ImportError as exc:
        raise ImportError(
            "This experiment needs the SEFA runtime dependencies, including "
            "torch, to decode original/reconstructed images."
        ) from exc

    torch = torch_module
    parse_gan_type = parse_gan_type_func
    load_generator = load_generator_func
    postprocess = postprocess_func
    to_tensor = to_tensor_func

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    generator = load_generator(args.model)
    gan_type = parse_gan_type(generator)
    latents, latent_source = load_or_sample_latents(generator, gan_type, args)
    if len(latents) == 0:
        raise ValueError("No latent codes are available for this experiment.")

    latent_dim = int(latents.shape[-1])
    mappings = {
        name: build_basis_matrix(name, latent_dim=latent_dim, degree=args.degree)
        for name in args.basis
    }

    rows = []
    for sample_id, latent in enumerate(latents):
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        original_image = decode_image(generator, gan_type, latent)
        original_path = os.path.join(sample_dir, "original.png")
        save_image(original_image, original_path)
        grid_items = [("original", original_image)]

        for basis_name, mapping in mappings.items():
            high_dim = map_latent_slices(latent.astype(np.float64), mapping.matrix)
            for inverse_method in args.inverse_methods:
                recon_latent = inverse_map(
                    high_dim,
                    mapping.matrix,
                    inverse_method,
                    ridge_lambda=args.ridge_lambda,
                    cg_tol=args.cg_tol,
                    cg_maxiter=args.cg_maxiter,
                )
                recon_image = decode_image(generator, gan_type, recon_latent)

                safe_basis = basis_name.replace("/", "_")
                safe_inverse = inverse_method.replace("/", "_")
                recon_name = f"recon_{safe_basis}_{safe_inverse}.png"
                recon_path = os.path.join(sample_dir, recon_name)
                save_image(recon_image, recon_path)
                grid_items.append((f"{basis_name}/{inverse_method}", recon_image))

                row = {
                    "sample_id": sample_id,
                    "basis_name": basis_name,
                    "inverse_method": inverse_method,
                    "latent_shape": json.dumps(list(latent.shape)),
                    "high_dim_shape": json.dumps(list(high_dim.shape)),
                }
                row.update(latent_errors(latent, recon_latent))
                row.update(image_errors(original_image, recon_image))
                rows.append(row)

        make_comparison_grid(
            grid_items,
            os.path.join(sample_dir, "comparison_grid.png"),
            thumb_size=args.grid_thumb_size,
            columns=5,
        )

    csv_path = os.path.join(args.output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames())
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(args.output_dir, "results.json")
    payload = {
        "model": args.model,
        "gan_type": gan_type,
        "latent_source": latent_source,
        "num_samples": int(len(latents)),
        "degree": args.degree,
        "basis": args.basis,
        "inverse_methods": args.inverse_methods,
        "results": rows,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved per-sample images under: {args.output_dir}")


if __name__ == "__main__":
    main()
