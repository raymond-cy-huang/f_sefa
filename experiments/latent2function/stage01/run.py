"""Stage 01: evaluate w -> f(w) -> w' for fixed StyleGAN generators."""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from experiment_utils.function_space_mapping import projected_round_trip
from experiment_utils.metrics import image_errors
from experiment_utils.metrics import latent_errors
from experiment_utils.metrics import save_image

STYLEGAN_TYPES = {"stylegan", "stylegan2", "psp_stylegan2"}


def args():
    p = argparse.ArgumentParser(description="Stage 01 exact function-space round trip.")
    p.add_argument("--models", nargs="+",
                   default=["stylegan_ffhq_2019", "stylegan_cat256"])
    p.add_argument("--num_samples", type=int, default=2)
    p.add_argument("--basis", nargs="+", default=["legendre", "chebyshev"])
    p.add_argument("--function_dim", type=int, default=512)
    p.add_argument("--trunc_psi", type=float, default=0.7)
    p.add_argument("--trunc_layers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--gpu_id", default="0")
    p.add_argument("--output_dir", default="results/latent2function/stage01")
    return p.parse_args()


def sample_w(generator, n, config, torch):
    device = next(generator.parameters()).device
    z = torch.randn(n, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w, trunc_psi=config.trunc_psi, trunc_layers=config.trunc_layers)
    if w.ndim == 3:
        w = w[:, 0, :]
    return w.detach().cpu().numpy().astype(np.float32)


def decoder_code(generator, latent):
    latent = np.asarray(latent, dtype=np.float32)
    if latent.ndim == 1:
        latent = np.tile(latent[None, :], (generator.num_layers, 1))
    elif latent.ndim == 2 and latent.shape[0] == 1:
        latent = np.tile(latent, (generator.num_layers, 1))
    expected = (generator.num_layers, generator.w_space_dim)
    if latent.shape != expected:
        raise ValueError("Expected W/W+ shape {}, got {}.".format(expected, latent.shape))
    return latent[None, ...]


def decode(generator, latent, torch, to_tensor, postprocess):
    with torch.no_grad():
        image = generator.synthesis(
            to_tensor(decoder_code(generator, latent)))["image"]
    return postprocess(image)[0]


def psnr(original, reconstructed):
    mse = np.mean((original.astype(np.float64) -
                   reconstructed.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else float(10 * np.log10(255 ** 2 / mse))


def fieldnames():
    return [
        "model", "sample_id", "basis",
        "l2_error", "cosine_similarity", "image_ssim", "image_psnr",
    ]


def aggregate(rows):
    metrics = fieldnames()[3:]
    result = {}
    for model, basis in sorted({(r["model"], r["basis"]) for r in rows}):
        selected = [r for r in rows
                    if r["model"] == model and r["basis"] == basis]

        def stats(metric):
            values = np.asarray([r[metric] for r in selected], dtype=np.float64)
            if np.all(np.isinf(values)):
                return {"mean": float("inf"), "std": 0.0, "median": float("inf")}
            return {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }

        result["{}/{}".format(model, basis)] = {
            metric: stats(metric) for metric in metrics
        }
    return result


def main():
    config = args()
    if config.num_samples <= 0 or config.function_dim != 512:
        raise ValueError("num_samples must be positive and function_dim must be 512.")
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)

    import torch
    from utils import get_gan_type, load_generator, postprocess, to_tensor

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rows = []
    metadata = {}

    for model_name in config.models:
        generator = load_generator(model_name)
        gan_type = get_gan_type(generator)
        if gan_type not in STYLEGAN_TYPES:
            raise ValueError("Stage 01 expects StyleGAN, got {}.".format(gan_type))
        latents = sample_w(generator, config.num_samples, config, torch)
        model_dir = os.path.join(config.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        metadata[model_name] = {
            "gan_type": gan_type,
            "latent_space": "W",
            "num_samples": int(len(latents)),
            "latent_dimension": int(config.function_dim),
            "latent_source": "seeded random Z mapped to truncated W",
            "mapping_type": "orthogonal coefficient projection",
        }

        for sample_id, latent in enumerate(latents):
            if latent.shape[-1] != config.function_dim:
                raise ValueError("Latent dimension must equal function_dim=512.")
            sample_dir = os.path.join(model_dir, "sample_{:03d}".format(sample_id))
            os.makedirs(sample_dir, exist_ok=True)
            original = decode(generator, latent, torch, to_tensor, postprocess)
            save_image(original, os.path.join(sample_dir, "original.png"))

            for basis in config.basis:
                basis = basis.lower()
                function_values, reconstructed = projected_round_trip(
                    latent.astype(np.float64), basis)
                reconstructed_image = decode(
                    generator, reconstructed, torch, to_tensor, postprocess)
                save_image(
                    reconstructed_image,
                    os.path.join(sample_dir, "reconstructed_{}.png".format(basis)))

                row = {
                    "model": model_name,
                    "sample_id": sample_id,
                    "basis": basis,
                    "latent_shape": json.dumps(list(latent.shape)),
                    "function_shape": json.dumps(list(function_values.shape)),
                }
                row.update(latent_errors(latent, reconstructed))
                row.update(image_errors(original, reconstructed_image))
                row["image_psnr"] = psnr(original, reconstructed_image)
                rows.append(row)

    csv_path = os.path.join(config.output_dir, "results.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames(), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    aggregate_rows = []
    for key, values in aggregate(rows).items():
        model, basis = key.split("/", 1)
        aggregate_rows.append({
            "model": model,
            "basis": basis,
            "num_samples": sum(1 for row in rows if row["model"] == model and row["basis"] == basis),
            "l2_mean": values["l2_error"]["mean"],
            "cosine_mean": values["cosine_similarity"]["mean"],
            "ssim_mean": values["image_ssim"]["mean"],
            "psnr_mean": values["image_psnr"]["mean"],
        })
    average_csv_path = os.path.join(config.output_dir, "results_average.csv")
    with open(average_csv_path, "w", newline="", encoding="utf-8") as handle:
        average_writer = csv.DictWriter(handle, fieldnames=["model", "basis", "num_samples", "l2_mean", "cosine_mean", "ssim_mean", "psnr_mean"])
        average_writer.writeheader()
        average_writer.writerows(aggregate_rows)

    payload = {
        "experiment": "latent2function/stage01",
        "mapping": "w -> f_w(x) -> coefficient projection -> w_prime",
        "models": metadata,
        "basis": [b.lower() for b in config.basis],
        "function_dimension": config.function_dim,
        "basis_order": "0 through 511",
        "inverse": "direct orthogonal projection with basis-specific quadrature",
        "results": rows,
        "aggregate": aggregate(rows),
    }
    json_path = os.path.join(config.output_dir, "results.json")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=True)
    print("Saved CSV: {}".format(csv_path))
    print("Saved average CSV: {}".format(average_csv_path))
    print("Saved JSON: {}".format(json_path))


if __name__ == "__main__":
    main()
