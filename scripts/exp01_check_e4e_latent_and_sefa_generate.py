"""Minimal e4e latent loading and SeFa generation check."""

import argparse
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTS = {".npy", ".npz", ".pt", ".pth"}
DEFAULT_STRENGTHS = [-3.0, 0.0, 3.0]
DEFAULT_SEMANTIC_INDICES = [0]
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latent_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_per_group", type=int, default=None)
    parser.add_argument("--sampling_strategy", choices=["first", "even"], default="first")
    parser.add_argument("--semantic_indices", type=int, nargs="+", default=DEFAULT_SEMANTIC_INDICES)
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def discover_latent_files(latent_dir):
    latent_dir = Path(latent_dir)
    return sorted(
        path for path in latent_dir.rglob("*")
        if (
            path.is_file()
            and path.parent.name == "per_image_npy"
            and path.suffix.lower() in SUPPORTED_EXTS
        )
    )


def evenly_sample(paths, count):
    if count >= len(paths):
        return list(paths)
    if count <= 0:
        return []
    indices = np.linspace(0, len(paths) - 1, count)
    return [paths[int(round(index))] for index in indices]


def select_latent_files(latent_files, num_samples, num_per_group=None, sampling_strategy="first"):
    if num_per_group is None:
        return latent_files[:num_samples]
    grouped = {}
    for path in latent_files:
        group = path.parent.parent.name
        grouped.setdefault(group, []).append(path)
    selected = []
    for group in sorted(grouped):
        group_files = grouped[group]
        if sampling_strategy == "even":
            selected.extend(evenly_sample(group_files, num_per_group))
        else:
            selected.extend(group_files[:num_per_group])
    return selected[:num_samples]


def _first_tensor_like(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, dict):
        preferred = ["latent", "latents", "w", "ws", "w_plus", "codes", "code"]
        for key in preferred:
            if key in obj:
                found = _first_tensor_like(obj[key])
                if found is not None:
                    return found
        for value in obj.values():
            found = _first_tensor_like(value)
            if found is not None:
                return found
    if isinstance(obj, (list, tuple)):
        for value in obj:
            found = _first_tensor_like(value)
            if found is not None:
                return found
    return None


def load_latent(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=True)
        if array.dtype == object:
            array = _first_tensor_like(array.item())
    elif path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)
        keys = list(data.keys())
        preferred = [key for key in ["latent", "latents", "w", "ws", "w_plus", "codes", "code"] if key in data]
        key = preferred[0] if preferred else keys[0]
        array = data[key]
    elif path.suffix.lower() in {".pt", ".pth"}:
        obj = torch.load(path, map_location="cpu")
        array = _first_tensor_like(obj)
    else:
        raise ValueError(f"Unsupported latent file extension: {path.suffix}")
    if array is None:
        raise ValueError(f"No tensor-like latent found in {path}")
    return np.asarray(array)


def single_latent(array):
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[1:] == (18, 512):
        return array[:1]
    if array.ndim == 2 and array.shape == (18, 512):
        return array[None, ...]
    if array.ndim == 2 and array.shape[1] == 512:
        return array[:1]
    if array.ndim == 1 and array.shape[0] == 512:
        return array[None, ...]
    if array.ndim > 3 and array.shape[-2:] == (18, 512):
        return array.reshape(-1, 18, 512)[:1]
    raise ValueError(f"Unsupported latent shape {array.shape}")


def detect_latent_type(array):
    shape = tuple(array.shape)
    if shape == (512,) or shape == (1, 512):
        return "W"
    if shape == (18, 512) or shape == (1, 18, 512):
        return "W+"
    if len(shape) == 2 and shape[1] == 512:
        return "W batch"
    if len(shape) == 3 and shape[1:] == (18, 512):
        return "W+ batch"
    return "unknown"


def latent_stats(array):
    numeric = np.asarray(array, dtype=np.float32)
    return {
        "shape": tuple(array.shape),
        "dtype": str(array.dtype),
        "min": float(numeric.min()),
        "max": float(numeric.max()),
        "mean": float(numeric.mean()),
        "std": float(numeric.std()),
        "latent_type": detect_latent_type(array),
    }


def normalize_for_generator(array, num_layers):
    latent_type = detect_latent_type(array)
    code = single_latent(array).astype(np.float32)
    if code.shape == (1, 512):
        code = np.repeat(code[:, None, :], num_layers, axis=1)
        strategy = "W repeated across all W+ layers"
    elif code.shape == (1, 18, 512):
        strategy = "W+ used directly; SeFa edit applied consistently to all selected layers"
    else:
        raise ValueError(f"Cannot normalize latent shape {code.shape}")
    return code, latent_type, strategy


def save_image(array, path):
    Image.fromarray(array).save(path)


def make_contact_sheet(rows, output_path, row_labels=None, col_labels=None):
    if not rows:
        return
    h, w = rows[0][0].shape[:2]
    num_cols = len(rows[0])
    font = ImageFont.load_default(size=56)
    label_w = 440 if row_labels else 0
    header_h = 130 if col_labels else 0
    sheet = Image.new("RGB", (label_w + w * num_cols, header_h + h * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    if col_labels:
        for col_idx, label in enumerate(col_labels):
            bbox = draw.textbbox((0, 0), label, font=font)
            text_w = bbox[2] - bbox[0]
            draw.text(
                (label_w + col_idx * w + (w - text_w) // 2, 34),
                label,
                fill=(0, 0, 0),
                font=font,
            )
    if row_labels:
        for row_idx, label in enumerate(row_labels):
            draw.text((32, header_h + row_idx * h + 42), label, fill=(0, 0, 0), font=font)
    for row_idx, images in enumerate(rows):
        for col_idx, image in enumerate(images):
            sheet.paste(
                Image.fromarray(image).convert("RGB"),
                (label_w + col_idx * w, header_h + row_idx * h),
            )
    sheet.save(output_path)


def strength_filename(strength):
    if strength == 0:
        return "sefa_0.png"
    prefix = "p" if strength > 0 else "m"
    value = abs(strength)
    if float(value).is_integer():
        value_text = str(int(value))
    else:
        value_text = str(value).replace(".", "p")
    return f"sefa_{prefix}{value_text}.png"


def output_identity_from_latent_path(path):
    path = Path(path)
    gender_dir = path.parent.parent.name
    stem = path.stem
    image_id = stem.split("_", 1)[0]
    return gender_dir, image_id


def main():
    args = parse_args()
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    output_dir = Path(args.output_dir)
    if output_dir.exists() and args.overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "exp01_log.txt"

    latent_files = discover_latent_files(args.latent_dir)
    valid_exts = sorted({path.suffix.lower() for path in latent_files})
    selected_files = select_latent_files(
        latent_files,
        args.num_samples,
        args.num_per_group,
        args.sampling_strategy,
    )

    log_file = log_path.open("w", encoding="utf-8", buffering=1)

    def log(message):
        print(message)
        log_file.write(message + "\n")

    log("Experiment: exp01_check_e4e_latent_and_sefa_generate")
    log(f"latent_dir: {args.latent_dir}")
    log(f"output_dir: {output_dir}")
    log(f"requested_device: {args.device}")
    log(f"num_samples: {args.num_samples}")
    log(f"num_per_group: {args.num_per_group}")
    log(f"sampling_strategy: {args.sampling_strategy}")
    log(f"semantic_indices: {args.semantic_indices}")
    log(f"strengths: {args.strengths}")
    log("latent_discovery: recursive search restricted to per_image_npy folders")
    log("output_naming: <output_dir>/<gt_female_or_gt_male>/image_<source_id>/")
    log(f"supported_file_types_detected: {', '.join(valid_exts) if valid_exts else 'none'}")
    log("first_10_latent_files:")
    for path in latent_files[:10]:
        log(f"  {path}")

    if not selected_files:
        log("ERROR: No valid latent files found.")
        log_file.close()
        raise SystemExit(1)

    log("first_valid_latent_stats:")
    loaded = []
    for path in selected_files:
        try:
            array = load_latent(path)
            stats = latent_stats(array)
            code_stats = latent_stats(single_latent(array))
            loaded.append((path, array, stats, code_stats))
            log(
                f"  {path} | shape={stats['shape']} dtype={stats['dtype']} "
                f"min={stats['min']:.6f} max={stats['max']:.6f} "
                f"mean={stats['mean']:.6f} std={stats['std']:.6f} "
                f"type={stats['latent_type']} | selected_code_shape={code_stats['shape']} "
                f"selected_code_type={code_stats['latent_type']}"
            )
        except Exception as exc:
            log(f"  {path} | ERROR loading latent: {exc}")

    if not loaded:
        log("ERROR: No latent files could be loaded.")
        log_file.close()
        raise SystemExit(1)

    from utils import factorize_weight, get_gan_type, load_generator, postprocess

    log("Loading generator: e4e_ffhq_encode")
    generator = load_generator("e4e_ffhq_encode")
    gan_type = get_gan_type(generator)
    device = next(generator.parameters()).device
    log(f"actual_device: {device}")
    log(f"gan_type: {gan_type}")
    log(f"generator_num_layers: {getattr(generator, 'num_layers', 'unknown')}")
    log("generator_latent_expectation: W+ tensor [N, 18, 512] for e4e adapter synthesis()")

    layers, boundaries, values = factorize_weight(generator, "all")
    max_semantic_idx = boundaries.shape[0] - 1
    for semantic_idx in args.semantic_indices:
        if semantic_idx < 0 or semantic_idx > max_semantic_idx:
            raise ValueError(f"semantic index {semantic_idx} outside valid range 0..{max_semantic_idx}")
    log("semantic_direction: SeFa eigenvector indices requested for qualitative inspection")
    for semantic_idx in args.semantic_indices:
        log(f"semantic_{semantic_idx:03d}_eigen_value: {float(values[semantic_idx]):.6f}")
    log(f"sefa_layers: {layers}")

    for sample_idx, (path, array, stats, code_stats) in enumerate(loaded[: args.num_samples]):
        gender_dir, image_id = output_identity_from_latent_path(path)
        sample_dir = output_dir / gender_dir / f"image_{image_id}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        log(f"sample image_{image_id}")
        log(f"  sample_index: {sample_idx:03d}")
        log(f"  output_group: {gender_dir}")
        log(f"  source_image_id: {image_id}")
        log(f"  latent_file_path: {path}")
        log(f"  latent_shape: {stats['shape']}")
        log(f"  detected_latent_type: {stats['latent_type']}")
        log(f"  selected_code_shape: {code_stats['shape']}")
        log(f"  selected_code_type: {code_stats['latent_type']}")
        try:
            code, latent_type, strategy = normalize_for_generator(array, generator.num_layers)
            log(f"  generation_strategy: {strategy}")
            semantic_rows = []
            with torch.no_grad():
                tensor_code = torch.from_numpy(code).float().to(device)
                original = postprocess(generator.synthesis(tensor_code)["image"])[0]
                save_image(original, sample_dir / "original.png")
                log(f"  original_output_image_path: {sample_dir / 'original.png'}")

                for semantic_idx in args.semantic_indices:
                    semantic_dir = sample_dir / f"semantic_{semantic_idx:03d}"
                    semantic_dir.mkdir(parents=True, exist_ok=True)
                    boundary = boundaries[semantic_idx : semantic_idx + 1].astype(np.float32)
                    row_images = [original]
                    edited_codes = []
                    filenames = []
                    for strength in args.strengths:
                        edited = code.copy()
                        edited[:, layers, :] += boundary * strength
                        edited_codes.append(edited[0])
                        filenames.append(strength_filename(strength))
                    edited_batch = torch.from_numpy(np.stack(edited_codes, axis=0)).float().to(device)
                    images = postprocess(generator.synthesis(edited_batch)["image"])
                    for strength, filename, image in zip(args.strengths, filenames, images):
                        output_path = semantic_dir / filename
                        save_image(image, output_path)
                        row_images.append(image)
                        log(
                            f"  generated | latent_path={path} "
                            f"latent_shape={stats['shape']} semantic_index={semantic_idx} "
                            f"strength={strength:g} output_image_path={output_path}"
                        )
                    semantic_rows.append(row_images)
            col_labels = ["Original"] + [f"{strength:g}" if strength < 0 else f"+{strength:g}" if strength > 0 else "0" for strength in args.strengths]
            row_labels = [f"Semantic {semantic_idx:03d}" for semantic_idx in args.semantic_indices]
            contact_path = sample_dir / f"contact_sheet_image_{image_id}.png"
            make_contact_sheet(semantic_rows, contact_path, row_labels=row_labels, col_labels=col_labels)
            log(f"  contact_sheet: {contact_path}")
            log("  image_generation_succeeded: yes")
        except Exception as exc:
            log("  image_generation_succeeded: no")
            log(f"  error_message: {type(exc).__name__}: {exc}")

    log_file.close()


if __name__ == "__main__":
    main()
