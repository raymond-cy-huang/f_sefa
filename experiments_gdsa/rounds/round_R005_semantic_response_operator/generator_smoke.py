"""One-sample StyleGAN1/InterFaceGAN W-space gradient smoke test."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import torch
import torch.nn.functional as functional


ROUND_DIR = Path(__file__).resolve().parent
ROOT = ROUND_DIR.parents[2]
sys.path.insert(0, str(ROOT))

from utils import load_generator  # noqa: E402


MODEL_NAME = "stylegan_ffhq_2019"
CHECKPOINT = ROOT / "checkpoints/stylegan_ffhq_2019.pth"
BOUNDARY = Path("/home/rz/raymond_project/interfacegan/boundaries/stylegan_ffhq_smile_w_boundary.npy")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def main() -> None:
    seed = 20261025
    configure(seed)
    # The repository's legacy loader resolves ``checkpoints/`` relative to the
    # process working directory rather than this module.
    os.chdir(ROOT)
    generator = load_generator(MODEL_NAME)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    device = next(generator.parameters()).device
    latent_generator = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn(1, generator.z_space_dim, generator=latent_generator, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        if w.ndim == 3:
            w = w[:, 0, :]
    w = w.detach().requires_grad_(True)
    boundary = np.load(BOUNDARY).reshape(-1)
    direction = torch.as_tensor(boundary, device=device, dtype=w.dtype).reshape_as(w)
    direction = direction / direction.norm()
    resolution = 32
    yy, xx = torch.meshgrid(
        torch.arange(resolution, device=device),
        torch.arange(resolution, device=device),
        indexing="ij",
    )
    # Provisional, fixed geometric border mask. This is only a gradient smoke
    # measurement and is not the independent audit mask for a semantic claim.
    mask = ((xx < 8) | (xx >= 24) | (yy < 8) | (yy >= 24)).to(torch.float32)
    mask = mask[None, None, :, :]
    selected_count = int(mask.sum().item()) * 3

    def response(code: torch.Tensor) -> torch.Tensor:
        wp = code[:, None, :].expand(-1, generator.num_layers, -1)
        image = generator.synthesis(wp)["image"]
        image = functional.interpolate(image, (resolution, resolution), mode="bilinear", align_corners=False)
        return (image * mask).reshape(-1) / np.sqrt(selected_count)

    baseline_a = response(w)
    baseline_b = response(w)
    deterministic_max_abs = float((baseline_a - baseline_b).abs().max().detach().cpu())
    _, jvp = torch.autograd.functional.jvp(response, w, direction, create_graph=False, strict=True)
    rng = torch.Generator(device=device).manual_seed(seed + 1)
    output_probe = torch.randn(jvp.shape, generator=rng, device=device, dtype=jvp.dtype)
    output = response(w)
    vjp = torch.autograd.grad(output, w, grad_outputs=output_probe, retain_graph=False)[0]
    adjoint_left = float(torch.dot(jvp, output_probe).detach().cpu())
    adjoint_right = float(torch.sum(direction * vjp).detach().cpu())
    adjoint_relative_error = abs(adjoint_left - adjoint_right) / max(
        abs(adjoint_left), abs(adjoint_right), 1e-12
    )
    finite_difference_checks = []
    for epsilon in (0.1, 0.03, 0.01, 0.003):
        with torch.no_grad():
            finite_difference = (response(w + epsilon * direction) - response(w - epsilon * direction)) / (2 * epsilon)
        relative_error = float(
            ((jvp - finite_difference).norm() / finite_difference.norm().clamp_min(1e-12)).detach().cpu()
        )
        cosine = float(
            (torch.dot(jvp, finite_difference) / (jvp.norm() * finite_difference.norm()).clamp_min(1e-12)).detach().cpu()
        )
        finite_difference_checks.append(
            {
                "epsilon": epsilon,
                "relative_error": relative_error,
                "cosine": cosine,
                "finite_difference_norm": float(finite_difference.norm().detach().cpu()),
            }
        )
    best_fd = min(finite_difference_checks, key=lambda item: item["relative_error"])
    jvp_fd_relative_error = float(best_fd["relative_error"])
    target_tau = 0.5
    edited = w.detach() + target_tau * direction.detach()
    target_progress_error = abs(float(torch.sum(direction.detach() * (edited - w.detach())).cpu()) - target_tau)
    summary = {
        "status": "PASS" if (
            generator.w_space_dim == 512
            and generator.num_layers == 18
            and deterministic_max_abs == 0.0
            and adjoint_relative_error <= 1e-3
            and jvp_fd_relative_error <= 0.05
            and target_progress_error <= 1e-6
        ) else "FAIL",
        "scope": "one-sample numerical smoke; not a semantic effectiveness pilot",
        "model": MODEL_NAME,
        "checkpoint": str(CHECKPOINT),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "boundary": str(BOUNDARY),
        "boundary_sha256": sha256(BOUNDARY),
        "boundary_norm": float(np.linalg.norm(boundary)),
        "w_space_dimension": generator.w_space_dim,
        "num_style_layers": generator.num_layers,
        "response_resolution": resolution,
        "fixed_border_selected_values": selected_count,
        "deterministic_response_max_abs": deterministic_max_abs,
        "jvp_norm": float(jvp.norm().detach().cpu()),
        "jvp_finite_difference_relative_error": jvp_fd_relative_error,
        "jvp_finite_difference_best_epsilon": best_fd["epsilon"],
        "jvp_finite_difference_checks": finite_difference_checks,
        "adjoint_left": adjoint_left,
        "adjoint_right": adjoint_right,
        "adjoint_absolute_error": abs(adjoint_left - adjoint_right),
        "adjoint_relative_error": adjoint_relative_error,
        "target_progress_absolute_error": target_progress_error,
        "mask_warning": "provisional geometric border; not an independent semantic audit mask",
    }
    output_path = ROUND_DIR / "results/generator_smoke.json"
    output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

