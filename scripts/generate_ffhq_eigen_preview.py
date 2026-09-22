"""Generate small FFHQ StyleGAN SeFa previews for selected eigen ranks."""
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
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", required=True)
    p.add_argument("--ranks", nargs="+", type=int, default=[3, 7])
    p.add_argument("--num_identities", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    from utils import factorize_weight, load_generator, postprocess, get_torch_device
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    generator = load_generator("stylegan_ffhq_2019")
    device = get_torch_device()
    z = torch.randn(args.num_identities, generator.z_space_dim, device=device)
    # Match the canonical SeFa FFHQ path in sefa.py exactly: map Z to W,
    # then apply the generator's truncation module before editing W+.
    with torch.no_grad():
        codes = generator.mapping(z)["w"]
        codes = generator.truncation(codes, trunc_psi=0.7, trunc_layers=8)
        codes = codes.detach().cpu().numpy().astype(np.float32)
    layers, boundaries, values = factorize_weight(generator, "all")
    layers = np.asarray(layers); boundaries = np.real(boundaries).astype(np.float32)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    for rank in args.ranks:
        if rank < 0 or rank >= len(values): raise ValueError(f"rank {rank} unavailable")
        d = boundaries[rank:rank+1]
        folder = out / f"eigen_{rank:03d}_preview_20"; folder.mkdir(exist_ok=True)
        images = []; paths = []
        for identity, code in enumerate(codes):
            for strength_idx, strength in enumerate(STRENGTHS):
                edited = code.copy(); edited[layers, :] += d * strength
                images.append(edited); paths.append(folder / f"{identity:03d}_{strength_idx:03d}.jpg")
        with torch.no_grad():
            rendered = postprocess(generator.synthesis(
                torch.from_numpy(np.stack(images)).float().to(device),
                randomize_noise=False)["image"])
        for image, path in zip(rendered, paths): Image.fromarray(image).convert("RGB").save(path, quality=95)
        print(f"eigen_{rank:03d}: wrote {len(paths)} -> {folder}", flush=True)

if __name__ == "__main__": main()
