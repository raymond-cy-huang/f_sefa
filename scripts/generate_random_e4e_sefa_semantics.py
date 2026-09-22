"""Generate random FFHQ SeFa semantic edits with the e4e decoder."""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


SEMANTICS = {
    "group_0-1": "0-1",
    "group2-5": "2-5",
    "group-6-12": "6-12",
}

STRENGTHS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_identities", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def dominant_direction(generator, layer_range):
    from utils import factorize_weight

    layers, boundaries, eigen_values = factorize_weight(generator, layer_range)
    eigen_values = np.real(eigen_values)
    boundaries = np.real(boundaries).astype(np.float32)
    eigen_index = int(np.argmax(eigen_values))
    return layers, boundaries[eigen_index : eigen_index + 1], float(eigen_values[eigen_index]), eigen_index


def save_rgb(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).convert("RGB").save(path, quality=95)


def main():
    args = parse_args()
    if args.device == "cpu":
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    from utils import get_torch_device, load_generator, postprocess

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for semantic in SEMANTICS:
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

    directions = {}
    for semantic, layer_range in SEMANTICS.items():
        print(f"Factorizing {semantic} layers {layer_range}", flush=True)
        layers, boundary, eigen_value, eigen_index = dominant_direction(generator, layer_range)
        directions[semantic] = (layers, boundary, eigen_value, eigen_index)
        print(
            f"{semantic}: layers={layer_range} selected_eigen_index={eigen_index} "
            f"selected_eigenvalue={eigen_value:.6f}",
            flush=True,
        )

    for semantic, (layers, boundary, _eigen_value, _eigen_index) in directions.items():
        semantic_dir = output_dir / semantic
        edited_codes = []
        output_paths = []
        for identity_idx, code in enumerate(base_codes):
            for strength_idx, strength in enumerate(STRENGTHS):
                edited = code.copy()
                edited[layers, :] += boundary * strength
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


if __name__ == "__main__":
    main()
