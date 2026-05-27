"""Decode saved SeFa latent codes back into images.

Example:
  python decode_latent.py --model stylegan_animeface512 --index 0
"""

import argparse
import os

import numpy as np
from PIL import Image
import torch

from models import MODEL_ZOO
from models import build_generator
from models import parse_gan_type


CHECKPOINT_DIR = "checkpoints"
LATENT_DIR = "latent_codes"
OUTPUT_DIR = "outputs"


def to_tensor(array):
    return torch.from_numpy(array).float().cuda()


def postprocess(images, min_val=-1.0, max_val=1.0):
    images = images.detach().cpu().numpy()
    images = (images - min_val) * 255 / (max_val - min_val)
    images = np.clip(images + 0.5, 0, 255).astype(np.uint8)
    return images.transpose(0, 2, 3, 1)


def load_decoder(model_name):
    if model_name not in MODEL_ZOO:
        raise KeyError(f"Unknown model name: {model_name}")

    config = MODEL_ZOO[model_name].copy()
    config.pop("url", None)
    generator = build_generator(**config)

    checkpoint_path = os.path.join(CHECKPOINT_DIR, model_name + ".pth")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Missing decoder checkpoint: {checkpoint_path}. "
            "Put the model .pth file under checkpoints/ first."
        )
    if os.path.getsize(checkpoint_path) == 0:
        raise RuntimeError(f"Decoder checkpoint is empty: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "generator_smooth" in checkpoint:
        state_dict = checkpoint["generator_smooth"]
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        state_dict = checkpoint

    generator.load_state_dict(state_dict)
    generator.cuda().eval()
    return generator


def decode(model, code):
    gan_type = parse_gan_type(model)
    with torch.no_grad():
        if gan_type == "pggan":
            image = model(to_tensor(code))["image"]
        elif gan_type in ["stylegan", "stylegan2"]:
            image = model.synthesis(to_tensor(code))["image"]
        else:
            raise ValueError(f"Unsupported GAN type: {gan_type}")
    return postprocess(image)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="stylegan_animeface512")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    latent_path = os.path.join(LATENT_DIR, f"{args.model}_latents.npy")
    latents = np.load(latent_path)
    if args.index < 0 or args.index >= len(latents):
        raise IndexError(f"--index must be in [0, {len(latents) - 1}]")

    code = latents[args.index : args.index + 1]
    decoder = load_decoder(args.model)
    image = decode(decoder, code)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output = args.output or os.path.join(
        OUTPUT_DIR, f"{args.model}_latent_{args.index:03d}.png"
    )
    Image.fromarray(image).save(output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
