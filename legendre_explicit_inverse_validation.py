"""Complete PyTorch validation for Legendre explicit mapping inverse.

This script verifies the closed-form function-space inverse on a real SeFa
latent path:

    V(t) = w0 + alpha1 * P1(t) * n1
              + alpha2 * P2(t) * n2
              + alpha3 * P3(t) * n3

The inverse estimates alpha_i by Legendre projection:

    alpha_hat_i = integral(<V(t)-w0, n_i> * P_i(t) dt) / ||P_i||^2
    ||P_i||^2 = 2 / (2*i + 1)

No optimizer, neural reconstructor, or backpropagation is used.
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch

from explicit_sefa_mapping import extract_sefa_directions
from models import parse_gan_type
from utils import load_generator
from utils import parse_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--layer_idx", default="0-1")
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--output_dir", default="results/legendre_explicit_inverse")
    parser.add_argument("--num_integral_samples", type=int, default=8192)
    parser.add_argument("--num_visual_samples", type=int, default=7)
    parser.add_argument("--alpha1", type=float, default=2.0)
    parser.add_argument("--alpha2", type=float, default=-1.5)
    parser.add_argument("--alpha3", type=float, default=0.75)
    return parser.parse_args()


def legendre_basis(t):
    """Returns P1, P2, P3 evaluated on t in [-1, 1]."""

    if torch.any(t < -1.0) or torch.any(t > 1.0):
        raise ValueError("Legendre basis expects t in [-1, 1].")
    p1 = t
    p2 = 0.5 * (3.0 * t.square() - 1.0)
    p3 = 0.5 * (5.0 * t.pow(3) - 3.0 * t)
    return torch.stack([p1, p2, p3], dim=0)


def legendre_norm(order, device, dtype):
    """Continuous Legendre norm: integral_{-1}^{1} P_m(t)^2 dt."""

    return torch.tensor(2.0 / (2 * order + 1), device=device, dtype=dtype)


def midpoint_samples(num_samples, device, dtype):
    """Uniform midpoint Riemann samples over [-1, 1]."""

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    dt = 2.0 / num_samples
    t = -1.0 + (torch.arange(num_samples, device=device, dtype=dtype) + 0.5) * dt
    return t, torch.tensor(dt, device=device, dtype=dtype)


def collect_stylegan_affine_weight(generator, layer_idx):
    """Collects real StyleGAN affine weights into W with shape [out, w_dim]."""

    layers = parse_indices(layer_idx, min_val=0, max_val=generator.num_layers - 1)
    weights = []
    for idx in layers:
        layer = generator.synthesis.__getattr__(f"layer{idx}")
        weights.append(layer.style.weight.detach())
    return layers, torch.cat(weights, dim=0)


def make_wplus_directions(directions_512, w0, active_layers):
    """Places 512-D SeFa directions into selected W+ layers."""

    expanded = []
    for direction in directions_512:
        full_direction = torch.zeros_like(w0, dtype=torch.float64)
        full_direction[active_layers, :] = direction.to(dtype=torch.float64)
        full_direction = torch.nn.functional.normalize(full_direction.reshape(-1), p=2, dim=0).view_as(w0)
        expanded.append(full_direction)
    return torch.stack(expanded, dim=0)


def build_legendre_trajectory(w0, directions, alphas, basis):
    """Builds V(t) as a tensor with shape [num_t, *w0.shape]."""

    view_shape = (basis.shape[1],) + (1,) * w0.ndim
    trajectory = w0.unsqueeze(0).expand((basis.shape[1],) + tuple(w0.shape)).clone()
    for idx in range(3):
        trajectory = trajectory + alphas[idx] * basis[idx].view(view_shape) * directions[idx].unsqueeze(0)
    return trajectory


def estimate_alphas_from_legendre_projection(trajectory, w0, directions, basis, dt):
    """Projects V(t)-w0 onto each n_i and P_i, using exact Legendre norms."""

    displacement = trajectory - w0.unsqueeze(0)
    flat_displacement = displacement.reshape(displacement.shape[0], -1)
    flat_directions = directions.reshape(directions.shape[0], -1)

    estimates = []
    for idx, order in enumerate([1, 2, 3]):
        projected_curve = flat_displacement @ flat_directions[idx]
        numerator = torch.sum(projected_curve * basis[idx]) * dt
        denominator = legendre_norm(order, device=trajectory.device, dtype=trajectory.dtype)
        estimates.append(numerator / denominator)
    return torch.stack(estimates)


def decode_stylegan(generator, code):
    with torch.no_grad():
        image = generator.synthesis(code.float())["image"]
    return image


def image_tensor_to_uint8(image):
    """Converts one NCHW or CHW image tensor from [-1, 1] to uint8 HWC."""

    if image.ndim == 4:
        image = image[0]
    image = (image.detach().cpu().float() + 1.0) * 127.5
    image = image.clamp(0, 255).byte()
    return image.permute(1, 2, 0).numpy()


def save_image(image, path):
    Image.fromarray(image_tensor_to_uint8(image)).save(path)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generator = load_generator(args.model)
    gan_type = parse_gan_type(generator)
    if gan_type not in ["stylegan", "stylegan2"]:
        raise ValueError("This validation script currently supports StyleGAN W+ latents.")

    latent_path = os.path.join(args.latent_dir, f"{args.model}_latents.npy")
    latents = np.load(latent_path)
    if args.index < 0 or args.index >= len(latents):
        raise IndexError(f"--index must be in [0, {len(latents) - 1}]")

    device = next(generator.parameters()).device
    w0_batch = torch.from_numpy(latents[args.index : args.index + 1]).to(device=device)
    w0 = w0_batch[0].to(dtype=torch.float64)

    layers, weight = collect_stylegan_affine_weight(generator, args.layer_idx)
    directions_512 = extract_sefa_directions(weight, k=3).to(device=device, dtype=torch.float64)
    directions = make_wplus_directions(directions_512, w0, layers)
    alphas_true = torch.tensor([args.alpha1, args.alpha2, args.alpha3], device=device, dtype=torch.float64)

    t, dt = midpoint_samples(args.num_integral_samples, device=device, dtype=torch.float64)
    basis = legendre_basis(t)
    trajectory = build_legendre_trajectory(w0, directions, alphas_true, basis)
    alphas_hat = estimate_alphas_from_legendre_projection(trajectory, w0, directions, basis, dt)

    # Invert the endpoint V(1) back to w0 with the recovered coefficients.
    endpoint_basis = legendre_basis(torch.ones(1, device=device, dtype=torch.float64))[:, 0]
    mapped_endpoint = (
        w0
        + torch.sum(alphas_true[:, None, None] * endpoint_basis[:, None, None] * directions, dim=0)
    )
    recovered_w0 = (
        mapped_endpoint
        - torch.sum(alphas_hat[:, None, None] * endpoint_basis[:, None, None] * directions, dim=0)
    )

    original_image = decode_stylegan(generator, w0.unsqueeze(0))
    recovered_image = decode_stylegan(generator, recovered_w0.unsqueeze(0))
    mapped_image = decode_stylegan(generator, mapped_endpoint.unsqueeze(0))

    original_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_original.png")
    mapped_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_mapped_endpoint.png")
    recovered_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_recovered.png")
    save_image(original_image, original_path)
    save_image(mapped_image, mapped_path)
    save_image(recovered_image, recovered_path)

    latent_mse = torch.mean((w0 - recovered_w0).square()).item()
    image_mse = torch.mean((original_image - recovered_image).square()).item()

    print("=== Legendre Explicit Mapping Inverse Validation ===")
    print(f"model: {args.model}")
    print(f"latent index: {args.index}")
    print(f"active SeFa layers: {layers}")
    print(f"integral samples: {args.num_integral_samples}")
    print(f"dt: {dt.item():.16e}")
    print(f"original image: {original_path}")
    print(f"mapped endpoint image: {mapped_path}")
    print(f"recovered image: {recovered_path}")
    print()
    print("alpha_true vs alpha_hat")
    print("order | norm 2/(2m+1)      | alpha_true          | alpha_hat           | abs_error")
    for idx, order in enumerate([1, 2, 3]):
        norm = legendre_norm(order, device=device, dtype=torch.float64).item()
        true_value = alphas_true[idx].item()
        estimated_value = alphas_hat[idx].item()
        print(
            f"P{order:<4} | {norm:.16e} | {true_value: .16e} | "
            f"{estimated_value: .16e} | {abs(true_value - estimated_value):.3e}"
        )
    print()
    print(f"latent MSE: {latent_mse:.16e}")
    print(f"image MSE:  {image_mse:.16e}")


if __name__ == "__main__":
    main()
