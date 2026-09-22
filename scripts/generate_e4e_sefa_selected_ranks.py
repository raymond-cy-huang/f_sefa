"""Generate selected ranked SeFa directions with the e4e FFHQ adapter."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
STRENGTHS = [-6.0, -3.0, 0.0, 3.0, 6.0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--ranks", nargs="+", type=int, default=[3, 7])
    parser.add_argument(
        "--names", nargs="+", default=None,
        help="Optional output folder names matching --ranks (for example age pose).",
    )
    parser.add_argument("--num_identities", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()
    if args.names is not None and len(args.names) != len(args.ranks):
        parser.error("--names must contain exactly one name for each --ranks value")

    from utils import factorize_weight, get_torch_device, load_generator, postprocess

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print("Loading generator: e4e_ffhq_encode", flush=True)
    generator = load_generator("e4e_ffhq_encode")
    device = get_torch_device()
    print(f"device: {device}", flush=True)

    z = torch.randn(args.num_identities, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        base_codes = generator.truncation(w).detach().cpu().numpy().astype(np.float32)

    print("Factorizing all layers", flush=True)
    layers, boundaries, eigen_values = factorize_weight(generator, "all")
    eigen_values = np.real(eigen_values)
    boundaries = np.real(boundaries).astype(np.float32)
    ranked_indices = np.argsort(eigen_values)[::-1]

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    for rank_idx, rank in enumerate(args.ranks):
        if rank < 0 or rank >= len(ranked_indices):
            raise ValueError(f"rank {rank} outside 0..{len(ranked_indices)-1}")
        source_index = int(ranked_indices[rank])
        direction = boundaries[source_index : source_index + 1]
        folder_name = args.names[rank_idx] if args.names is not None else f"eigen_{rank:03d}"
        out_dir = output_root / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        print(
            f"eigen_{rank:03d}: source_eigen_index={source_index} "
            f"eigenvalue={float(eigen_values[source_index]):.6f}",
            flush=True,
        )

        edited_codes, paths = [], []
        for identity_idx, code in enumerate(base_codes):
            for strength_idx, strength in enumerate(STRENGTHS):
                edited = code.copy()
                edited[layers, :] += direction * strength
                edited_codes.append(edited)
                paths.append(out_dir / f"{identity_idx:03d}_{strength_idx:03d}.jpg")

        for start in range(0, len(edited_codes), args.batch_size):
            end = min(start + args.batch_size, len(edited_codes))
            with torch.no_grad():
                batch = torch.from_numpy(np.stack(edited_codes[start:end])).float().to(device)
                images = postprocess(generator.synthesis(batch)["image"])
            for image, path in zip(images, paths[start:end]):
                Image.fromarray(image).convert("RGB").save(path, quality=95)
            print(f"eigen_{rank:03d}: wrote {end}/{len(paths)}", flush=True)

    print(f"Done -> {output_root}", flush=True)


if __name__ == "__main__":
    main()
