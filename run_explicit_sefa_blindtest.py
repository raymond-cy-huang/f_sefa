"""Runs real-generator SeFa blind-test grids with explicit nonlinear mapping."""

import argparse
import os

import numpy as np
from PIL import Image
import torch

from explicit_sefa_mapping import extract_sefa_directions
from explicit_sefa_mapping import generate_explicit_trajectory
from explicit_sefa_mapping import generate_linear_trajectory
from models import parse_gan_type
from utils import load_generator
from utils import parse_indices


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--layer_idx", default="0-1")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--latent_dir", default="latent_codes")
    parser.add_argument("--output_dir", default="results/explicit_sefa_blindtest")
    parser.add_argument("--alpha1", type=float, default=2.0)
    parser.add_argument("--alpha2", type=float, default=-1.5)
    return parser.parse_args()


def collect_stylegan_affine_weight(generator, layer_idx):
    """Collects real StyleGAN affine weights into one SeFa matrix W."""

    layers = parse_indices(layer_idx, min_val=0, max_val=generator.num_layers - 1)
    weights = []
    for idx in layers:
        layer = generator.synthesis.__getattr__(f"layer{idx}")
        # style.weight maps latent W to per-layer style modulation. Stacking
        # rows across layers gives W in shape [total_style_channels, w_dim].
        weights.append(layer.style.weight.detach())
    return layers, torch.cat(weights, dim=0)


def stylegan_decode(generator, latents):
    return generator.synthesis(latents)["image"]


def grid_to_uint8(grid):
    """Converts one CHW image grid from [-1, 1] to HWC uint8 RGB."""

    grid = grid.detach().cpu().float()
    grid = (grid + 1.0) * 127.5
    grid = grid.clamp(0, 255).byte()
    return grid.permute(1, 2, 0).numpy()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    generator = load_generator(args.model)
    gan_type = parse_gan_type(generator)
    if gan_type not in ["stylegan", "stylegan2"]:
        raise ValueError("This blind-test script currently supports StyleGAN-style W+ latents.")

    latent_path = os.path.join(args.latent_dir, f"{args.model}_latents.npy")
    latents = np.load(latent_path)
    if args.index < 0 or args.index >= len(latents):
        raise IndexError(f"--index must be in [0, {len(latents) - 1}]")

    w0 = torch.from_numpy(latents[args.index : args.index + 1]).float().cuda()
    layers, weight = collect_stylegan_affine_weight(generator, args.layer_idx)
    directions = extract_sefa_directions(weight, args.k)

    lambdas = torch.linspace(-3.0, 3.0, 7, device=w0.device)
    t = torch.linspace(-1.0, 1.0, 7, device=w0.device)

    print("=== Real SeFa Blind Test ===")
    print(f"model: {args.model}")
    print(f"latent index: {args.index}")
    print(f"active layers: {layers}")
    print(f"weight shape: {tuple(weight.shape)}")
    print(f"directions shape: {tuple(directions.shape)}")

    for sem_id in range(args.k):
        grid = generate_linear_trajectory(
            generator=generator,
            w0=w0,
            direction=directions[sem_id],
            lambdas=lambdas,
            decode_fn=stylegan_decode,
            return_grid=True,
        )
        path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_linear_sem_{sem_id:02d}.png")
        Image.fromarray(grid_to_uint8(grid)).save(path)
        print(f"linear semantic {sem_id}: {path}")

    explicit_grid = generate_explicit_trajectory(
        generator=generator,
        w0=w0,
        direction1=directions[0],
        direction2=directions[1],
        alpha1=args.alpha1,
        alpha2=args.alpha2,
        t=t,
        decode_fn=stylegan_decode,
        return_grid=True,
    )
    explicit_path = os.path.join(args.output_dir, f"{args.model}_{args.index:03d}_explicit_sem_00_01.png")
    Image.fromarray(grid_to_uint8(explicit_grid)).save(explicit_path)
    print(f"explicit nonlinear trajectory: {explicit_path}")


if __name__ == "__main__":
    main()
