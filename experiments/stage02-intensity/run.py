"""Generate a one-direction SeFa versus F-SeFa intensity sweep."""

import csv
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from analysis import basis_geometry
from analysis import direction_transport
from analysis import gram_validation
from analysis import hilbert_norm
from analysis import image_equivalence_metrics
from analysis import inverse_through_stage01
from analysis import normalize_hilbert
from analysis import orthonormal_coordinate_validation
from config import parse_args
from experiment_utils.metrics import latent_errors
from experiment_utils.metrics import save_image


SEMANTIC_RANKS = {
    "age": 3,
    "gender": 1,
    "smile": 18,
    "glasses": 10,
    "pose": 7,
}

CSV_FIELDS = [
    "semantic", "eigen_rank", "source_eigen_index", "eigenvalue", "lambda",
    "hilbert_direction_norm", "euclidean_direction_norm",
    "sefa_hilbert_displacement", "fsefa_hilbert_displacement",
    "fsefa_normalized_norm_square", "transport_l2", "transport_relative_l2",
    "transport_cosine", "branch_A_latent_l2", "branch_A_relative_l2",
    "branch_A_cosine", "sefa_vs_fsefa_psnr", "sefa_vs_fsefa_ssim",
    "sefa_vs_fsefa_lpips", "sefa_vs_fsefa_mae", "sefa_vs_fsefa_rmse",
]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_metadata():
    def invoke(*arguments):
        try:
            return subprocess.check_output(
                ["git", *arguments], cwd=ROOT_DIR,
                stderr=subprocess.DEVNULL, text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {
        "commit": invoke("rev-parse", "HEAD"),
        "dirty_status": invoke("status", "--short"),
    }


def sample_shared_w(generator, config, torch):
    device = next(generator.parameters()).device
    z = torch.randn(1, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w, trunc_psi=config.trunc_psi,
            trunc_layers=config.trunc_layers)
    if w.ndim == 3:
        w = w[:, 0, :]
    vector = w[0].detach().cpu().numpy().astype(np.float64)
    if vector.shape != (config.latent_dimension,):
        raise ValueError("Expected a 512-dimensional W vector, got {}.".format(
            vector.shape))
    return vector


def decoder_code(generator, latent):
    vector = np.asarray(latent, dtype=np.float32).reshape(-1)
    if vector.shape != (generator.w_space_dim,):
        raise ValueError("Unexpected W shape: {}.".format(vector.shape))
    return np.tile(vector[None, :], (generator.num_layers, 1))[None, ...]


def decode(generator, latent, torch, postprocess):
    device = next(generator.parameters()).device
    code = torch.from_numpy(decoder_code(generator, latent)).float().to(device)
    with torch.no_grad():
        result = generator.synthesis(code)
    return postprocess(result["image"])[0]


def optional_lpips(config, torch):
    if not config.enable_lpips:
        return None, "not requested"
    try:
        import lpips
    except ImportError:
        return None, "requested but optional lpips package is not installed"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device)
    model.eval()
    return model, "enabled"


def lpips_distance(model, first, second, torch):
    if model is None:
        return None
    device = next(model.parameters()).device

    def convert(image):
        tensor = torch.from_numpy(
            image.astype(np.float32).transpose(2, 0, 1))[None, ...]
        return tensor.to(device) / 127.5 - 1.0

    with torch.no_grad():
        return float(model(convert(first), convert(second)).item())


def lambda_label(value):
    return "lambda {:+g}".format(value)


def save_intensity_grid(semantic, rank, lambdas, sefa_images,
                        fsefa_images, path):
    from PIL import Image, ImageDraw

    thumb = 192
    row_label_width = 112
    header_height = 48
    row_height = thumb + 6
    canvas = Image.new(
        "RGB",
        (row_label_width + len(lambdas) * thumb,
         header_height + 2 * row_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "{} / eigen_{:03d}".format(semantic, rank), fill="black")
    for column, value in enumerate(lambdas):
        draw.text(
            (row_label_width + column * thumb + 8, 8),
            lambda_label(value), fill="black")
    for row_index, (label, images) in enumerate((
            ("SeFa", sefa_images), ("F-SeFa", fsefa_images))):
        top = header_height + row_index * row_height
        draw.text((8, top + thumb // 2), label, fill="black")
        for column, image in enumerate(images):
            resized = Image.fromarray(image).resize((thumb, thumb), Image.BILINEAR)
            canvas.paste(resized, (row_label_width + column * thumb, top))
    canvas.save(path)


def main():
    config = parse_args()
    if config.basis != "legendre":
        raise ValueError("This experiment is intentionally raw-Legendre only.")
    if config.latent_dimension != 512 or config.num_samples != 1:
        raise ValueError("This intensity sweep requires dimension 512 and one sample.")
    if len(config.lambdas) == 0 or not np.all(np.isfinite(config.lambdas)):
        raise ValueError("lambdas must contain finite values.")

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    os.makedirs(config.output_dir, exist_ok=True)
    image_dir = os.path.join(config.output_dir, "images")
    os.makedirs(image_dir, exist_ok=True)

    import torch
    from utils import factorize_weight, get_gan_type, load_generator, postprocess

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    generator = load_generator(config.model)
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator.")
    if int(generator.w_space_dim) != config.latent_dimension:
        raise ValueError("Generator W dimension does not match configuration.")
    shared_w = sample_shared_w(generator, config, torch)

    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    ranked_indices = np.argsort(eigenvalues)[::-1]
    rank = SEMANTIC_RANKS[config.semantic]
    source_index = int(ranked_indices[rank])
    direction = boundaries[source_index].reshape(-1)

    geometry = basis_geometry(config.basis, config.latent_dimension)
    norms = geometry["norms"]
    direction_hat, direction_h_norm = normalize_hilbert(direction, norms)
    direction_e_norm = float(np.linalg.norm(direction))
    normalized_square = float(np.dot(direction_hat * norms, direction_hat))
    lpips_model, lpips_status = optional_lpips(config, torch)

    rows = []
    sefa_images = []
    fsefa_images = []
    for value in config.lambdas:
        # Same numeric lambda, two distinct parameterizations.
        sefa_latent = shared_w + value * direction
        fsefa_coefficients = shared_w + value * direction_hat
        _, fsefa_latent = inverse_through_stage01(
            fsefa_coefficients, config.basis)

        # Representation-only Branch A should reproduce direct SeFa.
        _, branch_a_latent = inverse_through_stage01(
            sefa_latent, config.basis)
        branch_a_metrics = latent_errors(sefa_latent, branch_a_latent)
        transport = direction_transport(
            shared_w, direction, value, config.basis)

        sefa_image = decode(generator, sefa_latent, torch, postprocess)
        fsefa_image = decode(generator, fsefa_latent, torch, postprocess)
        suffix = "{:+g}".format(value)
        save_image(sefa_image, os.path.join(
            image_dir, "sefa_{}_lambda_{}.png".format(config.semantic, suffix)))
        save_image(fsefa_image, os.path.join(
            image_dir, "fsefa_{}_lambda_{}.png".format(config.semantic, suffix)))
        sefa_images.append(sefa_image)
        fsefa_images.append(fsefa_image)

        image_metrics = image_equivalence_metrics(sefa_image, fsefa_image)
        rows.append({
            "semantic": config.semantic,
            "eigen_rank": rank,
            "source_eigen_index": source_index,
            "eigenvalue": float(eigenvalues[source_index]),
            "lambda": float(value),
            "hilbert_direction_norm": direction_h_norm,
            "euclidean_direction_norm": direction_e_norm,
            "sefa_hilbert_displacement": abs(value) * direction_h_norm,
            "fsefa_hilbert_displacement": hilbert_norm(
                fsefa_coefficients - shared_w, norms),
            "fsefa_normalized_norm_square": normalized_square,
            "transport_l2": transport["l2_error"],
            "transport_relative_l2": transport["relative_l2_error"],
            "transport_cosine": transport["cosine_similarity"],
            "branch_A_latent_l2": branch_a_metrics["l2_error"],
            "branch_A_relative_l2": branch_a_metrics["relative_l2_error"],
            "branch_A_cosine": branch_a_metrics["cosine_similarity"],
            "sefa_vs_fsefa_psnr": image_metrics["psnr"],
            "sefa_vs_fsefa_ssim": image_metrics["ssim"],
            "sefa_vs_fsefa_lpips": lpips_distance(
                lpips_model, sefa_image, fsefa_image, torch),
            "sefa_vs_fsefa_mae": image_metrics["mae"],
            "sefa_vs_fsefa_rmse": image_metrics["rmse"],
        })

    metrics_path = os.path.join(config.output_dir, "metrics.csv")
    with open(metrics_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    grid_path = os.path.join(
        config.output_dir, "{}_lambda_sweep.png".format(config.semantic))
    save_intensity_grid(
        config.semantic, rank, config.lambdas,
        sefa_images, fsefa_images, grid_path)

    checkpoint_path = os.path.join(
        ROOT_DIR, "checkpoints", config.model + ".pth")
    summary = {
        "experiment": "stage02-intensity/one-direction-lambda-sweep",
        "config": config.to_dict(),
        "semantic": config.semantic,
        "eigen_rank": rank,
        "source_eigen_index": source_index,
        "eigenvalue": float(eigenvalues[source_index]),
        "mapping": {
            "sefa": "w(lambda) = w0 + lambda * d",
            "fsefa": (
                "w(lambda) = Psi(w0 + lambda * d / sqrt(d^T G d))"),
            "same_numeric_lambda": True,
            "interpretation": (
                "SeFa lambda uses the original Euclidean-unit eigenvector "
                "scale. F-SeFa lambda is Hilbert displacement. Equal numeric "
                "lambda values do not imply equal perceptual displacement."),
        },
        "direction": {
            "hilbert_norm": direction_h_norm,
            "euclidean_norm": direction_e_norm,
            "normalized_hilbert_norm_square": normalized_square,
            "orthonormal_coordinate_validation": (
                orthonormal_coordinate_validation(
                    direction, np.roll(direction, 1), norms)),
        },
        "gram_validation": gram_validation(geometry),
        "rows": rows,
        "lpips_status": lpips_status,
        "generator": {
            "model": config.model,
            "gan_type": get_gan_type(generator),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256(checkpoint_path),
            "factorized_layers": [int(item) for item in layers],
        },
        "git": git_metadata(),
        "outputs": {
            "comparison_grid": grid_path,
            "metrics_csv": metrics_path,
            "image_directory": image_dir,
        },
    }
    summary_path = os.path.join(config.output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    print("Saved intensity sweep: {}".format(grid_path))
    print("Saved metrics: {}".format(metrics_path))
    print("Saved summary: {}".format(summary_path))


if __name__ == "__main__":
    main()
