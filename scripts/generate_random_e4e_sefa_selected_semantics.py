"""Generate selected SeFa semantic edits from random FFHQ e4e latents."""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SEMANTIC_EIGEN_RANKS = {
    "age": 10,
    "gender": 1,
    "smile": 18,
}

STRENGTHS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_identities", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def save_rgb(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).convert("RGB").save(path, quality=95)


def main():
    args = parse_args()
    if args.device == "cpu":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from utils import factorize_weight, get_torch_device, load_generator, postprocess

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for semantic in SEMANTIC_EIGEN_RANKS:
        semantic_dir = output_dir / semantic
        if args.overwrite and semantic_dir.exists():
            shutil.rmtree(semantic_dir)
        semantic_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("Loading generator: e4e_ffhq_encode", flush=True)
    generator = load_generator("e4e_ffhq_encode")
    device = get_torch_device()

    print(f"Sampling {args.num_identities} random FFHQ identities", flush=True)
    z = torch.randn(args.num_identities, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        base_codes = generator.truncation(w).detach().cpu().numpy().astype(np.float32)

    print("Factorizing all layers", flush=True)
    layers, boundaries, eigen_values = factorize_weight(generator, "all")
    eigen_values = np.real(eigen_values)
    boundaries = np.real(boundaries).astype(np.float32)
    ranked_indices = np.argsort(eigen_values)[::-1]

    for semantic, eigen_rank in SEMANTIC_EIGEN_RANKS.items():
        eigen_index = int(ranked_indices[eigen_rank])
        eigen_value = float(eigen_values[eigen_index])
        direction = boundaries[eigen_index : eigen_index + 1]
        semantic_dir = output_dir / semantic
        print(
            f"{semantic}: eigen_rank={eigen_rank:03d} source_eigen_index={eigen_index} "
            f"eigenvalue={eigen_value:.6f}",
            flush=True,
        )

        edited_codes = []
        output_paths = []
        for identity_idx, code in enumerate(base_codes):
            for strength_idx, strength in enumerate(STRENGTHS):
                edited = code.copy()
                edited[layers, :] += direction * strength
                edited_codes.append(edited)
                output_paths.append(semantic_dir / f"{identity_idx:03d}_{strength_idx:03d}.jpg")

        for start in range(0, len(edited_codes), args.batch_size):
            end = min(start + args.batch_size, len(edited_codes))
            with torch.no_grad():
                batch = torch.from_numpy(np.stack(edited_codes[start:end], axis=0)).float().to(device)
                images = postprocess(generator.synthesis(batch)["image"])
            for image, output_path in zip(images, output_paths[start:end]):
                save_rgb(image, output_path)
            print(f"{semantic}: wrote {end}/{len(output_paths)} images", flush=True)
        print(f"{semantic}: done -> {semantic_dir}", flush=True)

    print(f"Done -> {output_dir}", flush=True)


if __name__ == "__main__":
    main()
