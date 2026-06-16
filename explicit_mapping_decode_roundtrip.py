"""Decode a real latent before and after explicit-mapping analytical inverse.

This script uses an existing saved latent code and the real SeFa decoder:

1. Decode the original latent code into an image.
2. Extract two SeFa semantic directions from the generator weights.
3. Build an explicit Legendre nonlinear trajectory from the latent.
4. Recover alpha values by discrete inner products only.
5. Invert one mapped endpoint back to the original latent and decode it.

The expected result is that the original and recovered images are visually
identical, with near-zero latent and image MSE.
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch

from utils import factorize_weight
from utils import get_gan_type
from utils import load_generator
from utils import postprocess
from utils import to_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--layer_idx", default="0-1")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--alpha1", type=float, default=2.0)
    parser.add_argument("--alpha2", type=float, default=-1.5)
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--output_dir", default="results/explicit_mapping_roundtrip")
    return parser.parse_args()


def normalize(vector, eps=1e-12):
    norm = np.linalg.norm(vector)
    if norm < eps:
        raise ValueError("Cannot normalize a near-zero vector.")
    return vector / norm


def legendre_basis(t):
    if np.any(t < -1.0) or np.any(t > 1.0):
        raise ValueError("Legendre basis expects t in [-1, 1].")
    phi1 = t
    phi2 = 0.5 * (3.0 * t**2 - 1.0)
    return phi1, phi2


def riemann_inner(values_a, values_b, dt):
    return float(dt * np.sum(values_a * values_b))


def sorted_sefa_directions(generator, layer_idx):
    """Returns the top two real, orthonormal 512-D SeFa directions."""

    layers, boundaries, values = factorize_weight(generator, layer_idx)
    order = np.argsort(np.real(values))[::-1]

    n1 = normalize(np.real(boundaries[order[0]]).astype(np.float64))
    raw_n2 = np.real(boundaries[order[1]]).astype(np.float64)
    n2 = normalize(raw_n2 - np.dot(raw_n2, n1) * n1)
    return layers, n1, n2, np.real(values[order[:2]])


def make_wplus_direction(boundary, code_shape, active_layers):
    """Places a 512-D direction into selected W+ layers and normalizes it."""

    direction = np.zeros(code_shape, dtype=np.float64)
    direction[active_layers, :] = boundary
    return normalize(direction)


def decode_image(generator, gan_type, code):
    with torch.no_grad():
        if gan_type == "pggan":
            image = generator(to_tensor(code))["image"]
        elif gan_type in ["stylegan", "stylegan2", "psp_stylegan2"]:
            image = generator.synthesis(to_tensor(code))["image"]
        else:
            raise ValueError(f"Unsupported GAN type: {gan_type}")
    return postprocess(image)[0]


def analytical_alpha(trajectory, w0, direction, basis, dt):
    """Solves alpha with only projection and discrete integration."""

    displacement = trajectory - w0[None, ...]
    projected_curve = displacement.reshape(len(basis), -1) @ direction.reshape(-1)
    numerator = riemann_inner(projected_curve, basis, dt)
    denominator = riemann_inner(basis, basis, dt)
    return numerator / denominator


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    latent_path = os.path.join(args.latent_dir, f"{args.model}_latents.npy")
    latents = np.load(latent_path)
    if args.index < 0 or args.index >= len(latents):
        raise IndexError(f"--index must be in [0, {len(latents) - 1}]")

    code = latents[args.index : args.index + 1].astype(np.float32)
    w0 = code[0].astype(np.float64)

    generator = load_generator(args.model)
    gan_type = get_gan_type(generator)
    layers, n1, n2, eigen_values = sorted_sefa_directions(generator, args.layer_idx)

    if gan_type not in ["stylegan", "stylegan2", "psp_stylegan2"]:
        raise ValueError("This round-trip script currently targets W+ StyleGAN latents.")

    direction1 = make_wplus_direction(n1, w0.shape, layers)
    direction2 = make_wplus_direction(n2, w0.shape, layers)

    t = np.linspace(-1.0, 1.0, args.num_samples)
    dt = float((t[-1] - t[0]) / (len(t) - 1))
    phi1, phi2 = legendre_basis(t)

    trajectory = (
        w0[None, ...]
        + args.alpha1 * phi1[:, None, None] * direction1[None, ...]
        + args.alpha2 * phi2[:, None, None] * direction2[None, ...]
    )

    estimated_alpha1 = analytical_alpha(trajectory, w0, direction1, phi1, dt)
    estimated_alpha2 = analytical_alpha(trajectory, w0, direction2, phi2, dt)

    mapped_endpoint = trajectory[-1]
    recovered_w0 = (
        mapped_endpoint
        - estimated_alpha1 * phi1[-1] * direction1
        - estimated_alpha2 * phi2[-1] * direction2
    ).astype(np.float32)
    recovered_code = recovered_w0[None, ...]

    original_image = decode_image(generator, gan_type, code)
    recovered_image = decode_image(generator, gan_type, recovered_code)

    original_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_original.png")
    recovered_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_recovered.png")
    Image.fromarray(original_image).save(original_path)
    Image.fromarray(recovered_image).save(recovered_path)

    latent_mse = float(np.mean((code - recovered_code) ** 2))
    image_mse = float(np.mean((original_image.astype(np.float32) - recovered_image.astype(np.float32)) ** 2))
    alpha_mse = float(
        np.mean(
            (
                np.array([args.alpha1, args.alpha2])
                - np.array([estimated_alpha1, estimated_alpha2])
            )
            ** 2
        )
    )

    print("=== Explicit Mapping Decode Round Trip ===")
    print(f"model: {args.model}")
    print(f"latent index: {args.index}")
    print(f"active SeFa layers: {layers}")
    print(f"top eigen values: {eigen_values.tolist()}")
    print(f"original image: {original_path}")
    print(f"recovered image: {recovered_path}")
    print()
    print(f"true alpha1:      {args.alpha1:.12f}")
    print(f"estimated alpha1: {estimated_alpha1:.12f}")
    print(f"true alpha2:      {args.alpha2:.12f}")
    print(f"estimated alpha2: {estimated_alpha2:.12f}")
    print()
    print(f"alpha MSE:  {alpha_mse:.16e}")
    print(f"latent MSE: {latent_mse:.16e}")
    print(f"image MSE:  {image_mse:.16e}")


if __name__ == "__main__":
    main()
