"""Explicit nonlinear basis pre-image experiment.

This experiment tests the nontrivial numerical inverse setting:

    z -> Phi(z) = orthogonal_basis(R z / scale) -> z_recon

where R is a fixed dense mixing matrix. Unlike the simple coordinate-wise
round-trip, the inverse is not a matrix pseudo-inverse and cannot recover z by
reading one coordinate feature directly. It solves the pre-image problem

    min_z ||Phi(z) - Phi(z0)||_2^2

with deterministic Gauss-Newton / SVD least-squares, without training an
inverse network and without changing the SEFA semantic discovery flow.

Example:
    python experiments/latent_basis_preimage.py --num_samples 3 --degree 4 \
        --output_dir results/basis_preimage
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
from experiment_utils.metrics import latent_errors
from experiment_utils.metrics import make_comparison_grid
from experiment_utils.metrics import save_image
from experiment_utils.preimage_mapping import MixedBasisPreimageMap

torch = None
parse_gan_type = None
load_generator = None
postprocess = None
to_tensor = None


def parse_args():
    parser = argparse.ArgumentParser(description="Explicit basis pre-image inversion experiment.")
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--basis", default="legendre",
                        choices=["legendre", "chebyshev", "hermite", "laguerre", "fourier"])
    parser.add_argument("--degree", type=int, default=4)
    parser.add_argument("--num_projections", type=int, default=None)
    parser.add_argument("--basis_input_scale", type=float, default=4.0)
    parser.add_argument("--mixing_seed", type=int, default=123)
    parser.add_argument("--inverse_iterations", type=int, default=25)
    parser.add_argument("--inverse_tolerance", type=float, default=1e-10)
    parser.add_argument("--step_clip", type=float, default=1.0)
    parser.add_argument("--output_dir", default="results/basis_preimage")
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
        "mean_iterations",
        "max_iterations",
        "mean_feature_residual",
        "max_feature_residual",
    ]


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.makedirs(args.output_dir, exist_ok=True)

    global torch, parse_gan_type, load_generator, postprocess, to_tensor
    try:
        import torch as torch_module
        from utils import get_gan_type as parse_gan_type_func
        from utils import load_generator as load_generator_func
        from utils import postprocess as postprocess_func
        from utils import to_tensor as to_tensor_func
    except ImportError as exc:
        raise ImportError("This experiment needs torch and the SEFA runtime dependencies.") from exc

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
        raise ValueError("No latent codes are available.")

    preimage_map = MixedBasisPreimageMap(
        latent_dim=int(latents.shape[-1]),
        basis_name=args.basis,
        degree=args.degree,
        input_scale=args.basis_input_scale,
        num_projections=args.num_projections,
        seed=args.mixing_seed,
    )

    rows = []
    for sample_id, latent in enumerate(latents):
        sample_dir = os.path.join(args.output_dir, f"sample_{sample_id:03d}")
        os.makedirs(sample_dir, exist_ok=True)

        high_dim = preimage_map.encode(latent)
        recon_latent, infos = preimage_map.inverse(
            high_dim,
            max_iterations=args.inverse_iterations,
            tolerance=args.inverse_tolerance,
            step_clip=args.step_clip,
        )

        original_image = decode_image(generator, gan_type, latent)
        recon_image = decode_image(generator, gan_type, recon_latent)
        save_image(original_image, os.path.join(sample_dir, "original.png"))
        save_image(recon_image, os.path.join(sample_dir, "recon_preimage.png"))
        make_comparison_grid(
            [("original", original_image), ("preimage", recon_image)],
            os.path.join(sample_dir, "comparison_grid.png"),
            columns=2,
        )

        feature_residuals = [info["final_residual_norm"] for info in infos]
        iterations = [info["iterations"] for info in infos]
        row = {
            "sample_id": sample_id,
            "basis_name": args.basis,
            "inverse_method": "gauss_newton_svd_lstsq",
            "latent_shape": json.dumps(list(latent.shape)),
            "high_dim_shape": json.dumps(list(high_dim.shape)),
            "mean_iterations": float(np.mean(iterations)),
            "max_iterations": int(np.max(iterations)),
            "mean_feature_residual": float(np.mean(feature_residuals)),
            "max_feature_residual": float(np.max(feature_residuals)),
        }
        row.update(latent_errors(latent, recon_latent))
        row.update(image_errors(original_image, recon_image))
        rows.append(row)

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
            "basis": args.basis,
            "degree": args.degree,
            "basis_input_scale": args.basis_input_scale,
            "num_projections": int(preimage_map.mixing.shape[0]),
            "mixing_seed": args.mixing_seed,
            "inverse_method": "gauss_newton_svd_lstsq",
            "results": rows,
        }, f, indent=2)

    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved images under: {args.output_dir}")


if __name__ == "__main__":
    main()
