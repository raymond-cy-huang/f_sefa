"""Metrics and image helpers for latent round-trip experiments."""

import numpy as np

try:
    from skimage.metrics import structural_similarity
except ImportError:
    structural_similarity = None


def latent_errors(original, reconstructed, eps=1e-12):
    """Computes vector reconstruction errors."""

    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    diff = original - reconstructed
    l2 = float(np.linalg.norm(diff.reshape(-1), ord=2))
    denom = max(float(np.linalg.norm(original.reshape(-1), ord=2)), eps)
    recon_denom = max(float(np.linalg.norm(reconstructed.reshape(-1), ord=2)), eps)
    cosine = float(np.dot(original.reshape(-1), reconstructed.reshape(-1)) / (denom * recon_denom))
    return {
        "l1_error": float(np.linalg.norm(diff.reshape(-1), ord=1)),
        "l2_error": l2,
        "relative_l2_error": l2 / denom,
        "max_abs_error": float(np.max(np.abs(diff))),
        "cosine_similarity": cosine,
    }


def image_errors(original, reconstructed):
    """Computes image MSE and SSIM for uint8 RGB images."""

    original = np.asarray(original)
    reconstructed = np.asarray(reconstructed)
    mse = float(np.mean((original.astype(np.float64) - reconstructed.astype(np.float64)) ** 2))
    if structural_similarity is None:
        ssim = _global_ssim(original, reconstructed)
    else:
        min_side = min(original.shape[0], original.shape[1])
        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        ssim = structural_similarity(
            original,
            reconstructed,
            channel_axis=2,
            data_range=255,
            win_size=max(3, win_size),
        )
    return {"image_mse": mse, "image_ssim": float(ssim)}


def _global_ssim(original, reconstructed):
    """Fallback SSIM approximation when scikit-image is unavailable."""

    x = original.astype(np.float64)
    y = reconstructed.astype(np.float64)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    scores = []
    for channel in range(x.shape[2]):
        xc = x[..., channel]
        yc = y[..., channel]
        mux = np.mean(xc)
        muy = np.mean(yc)
        varx = np.var(xc)
        vary = np.var(yc)
        cov = np.mean((xc - mux) * (yc - muy))
        scores.append(((2 * mux * muy + c1) * (2 * cov + c2)) /
                      ((mux**2 + muy**2 + c1) * (varx + vary + c2)))
    return float(np.mean(scores))


def save_image(array, path):
    """Saves a uint8 image."""

    from PIL import Image

    Image.fromarray(np.asarray(array, dtype=np.uint8)).save(path)


def make_comparison_grid(items, path, thumb_size=160, columns=5):
    """Saves a labeled comparison grid from [(label, image), ...]."""

    from PIL import Image, ImageDraw

    if not items:
        raise ValueError("items must be non-empty.")
    columns = max(1, columns)
    rows = int(np.ceil(len(items) / columns))
    label_height = 24
    cell_w = thumb_size
    cell_h = thumb_size + label_height
    canvas = Image.new("RGB", (columns * cell_w, rows * cell_h), "white")
    draw = ImageDraw.Draw(canvas)

    for idx, (label, image) in enumerate(items):
        row, col = divmod(idx, columns)
        x = col * cell_w
        y = row * cell_h
        pil_image = Image.fromarray(np.asarray(image, dtype=np.uint8)).resize(
            (thumb_size, thumb_size), Image.BILINEAR
        )
        canvas.paste(pil_image, (x, y + label_height))
        draw.text((x + 4, y + 4), label[:28], fill=(0, 0, 0))

    canvas.save(path)
