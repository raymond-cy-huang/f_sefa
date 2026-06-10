"""PyTorch utilities for SeFa directions and explicit nonlinear trajectories.

The module is intentionally generator-agnostic:

* ``extract_sefa_directions()`` only needs a weight matrix.
* ``generate_linear_trajectory()`` only needs a generator and one latent code.
* ``generate_explicit_trajectory()`` combines SeFa directions with explicit
  Legendre basis functions and decodes the resulting latent path.

For real StyleGAN/StyleGAN2 usage, pass a loaded generator whose forward method
accepts a latent batch, or provide ``decode_fn`` to adapt the call, for example:

    decode_fn = lambda G, w: G.synthesis(w)["image"]
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


ImageDecodeFn = Callable[[nn.Module, torch.Tensor], torch.Tensor]


def extract_sefa_directions(weight: torch.Tensor, k: int) -> torch.Tensor:
    """Extracts the top-k orthonormal SeFa directions from a weight matrix.

    Args:
        weight: A 2-D affine weight matrix ``W`` with shape ``[out_dim, w_dim]``.
            For StyleGAN-like affine layers, this is usually the modulation
            affine weight. The semantic direction ``n`` lives in the input
            latent dimension, so we need the right singular vectors of ``W``.
        k: Number of semantic directions to return.

    Returns:
        A tensor with shape ``[k, w_dim]``. Each row is unit-normalized and the
        rows are mutually orthogonal up to numerical precision.

    Mathematical note:
        SeFa solves ``max_n ||W n||^2``. This is equivalent to taking the top
        eigenvectors of ``W.T @ W`` or the top right singular vectors of ``W``.
    """

    if weight.ndim != 2:
        raise ValueError(f"Expected a 2-D weight matrix, got shape {tuple(weight.shape)}.")
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > min(weight.shape):
        raise ValueError(f"k={k} is larger than min(weight.shape)={min(weight.shape)}.")

    # torch.linalg.svd returns W = U @ diag(S) @ Vh. The rows of Vh are the
    # right singular vectors, i.e. directions in the latent input space.
    _, _, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    directions = vh[:k]
    return torch.nn.functional.normalize(directions, p=2, dim=1)


def make_image_grid(images: torch.Tensor, nrow: int | None = None) -> torch.Tensor:
    """Packs ``[N, C, H, W]`` images into a single grid tensor.

    This avoids a hard dependency on torchvision. The returned grid keeps the
    original value range, e.g. ``[-1, 1]`` for many GANs.
    """

    if images.ndim != 4:
        raise ValueError(f"Expected image batch [N, C, H, W], got {tuple(images.shape)}.")

    num_images, channels, height, width = images.shape
    if nrow is None:
        nrow = num_images
    nrow = max(1, min(nrow, num_images))
    ncol = (num_images + nrow - 1) // nrow

    grid = images.new_zeros(channels, ncol * height, nrow * width)
    for idx, image in enumerate(images):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * height : (row + 1) * height, col * width : (col + 1) * width] = image
    return grid


def default_decode(generator: nn.Module, latents: torch.Tensor) -> torch.Tensor:
    """Decodes latents with common generator calling conventions."""

    output = generator(latents)
    if isinstance(output, dict):
        if "image" not in output:
            raise KeyError("Generator returned a dict without an 'image' key.")
        return output["image"]
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _single_latent_without_batch(w0: torch.Tensor) -> torch.Tensor:
    """Returns one latent sample without the leading batch dimension."""

    if w0.ndim == 0:
        raise ValueError("w0 must be at least 1-D.")
    if w0.ndim >= 2 and w0.shape[0] == 1:
        return w0[0]
    return w0


def _expand_direction(direction: torch.Tensor, latent_shape: torch.Size) -> torch.Tensor:
    """Broadcasts a semantic direction to match one latent sample shape."""

    if direction.shape == latent_shape:
        expanded = direction
    elif direction.ndim == 1 and latent_shape[-1] == direction.shape[0]:
        expanded = direction.view(*([1] * (len(latent_shape) - 1)), -1).expand(latent_shape)
    else:
        try:
            expanded = torch.broadcast_to(direction, latent_shape)
        except RuntimeError as exc:
            raise ValueError(
                f"Direction shape {tuple(direction.shape)} cannot broadcast to "
                f"latent shape {tuple(latent_shape)}."
            ) from exc
    return torch.nn.functional.normalize(expanded.reshape(-1), p=2, dim=0).view(latent_shape)


def _decode_trajectory(
    generator: nn.Module,
    latents: torch.Tensor,
    decode_fn: ImageDecodeFn | None,
    return_grid: bool,
) -> torch.Tensor:
    """Runs a latent trajectory through the generator."""

    decoder = decode_fn or default_decode
    with torch.no_grad():
        images = decoder(generator, latents)
    return make_image_grid(images, nrow=images.shape[0]) if return_grid else images


def generate_linear_trajectory(
    generator: nn.Module,
    w0: torch.Tensor,
    direction: torch.Tensor,
    lambdas: torch.Tensor,
    decode_fn: ImageDecodeFn | None = None,
    return_grid: bool = True,
) -> torch.Tensor:
    """Generates images along ``w_i = w0 + lambda_i * direction``.

    Args:
        generator: Loaded generator model. By default this function calls
            ``generator(latents)``. Pass ``decode_fn`` for models that need a
            custom call such as ``generator.synthesis(latents)["image"]``.
        w0: Initial latent code. Accepts shapes like ``[1, 512]``,
            ``[1, num_layers, 512]``, ``[512]``, or ``[num_layers, 512]``.
        direction: One semantic direction. Shape can match ``w0`` without batch,
            or be a single ``[512]`` direction broadcast across W+ layers.
        lambdas: 1-D movement coefficients, e.g. ``torch.linspace(-3, 3, 7)``.
        decode_fn: Optional adapter ``decode_fn(generator, latent_batch)``.
        return_grid: If true, returns one ``[C, H_grid, W_grid]`` grid. If false,
            returns the raw image batch ``[len(lambdas), C, H, W]``.
    """

    if lambdas.ndim != 1:
        raise ValueError("lambdas must be a 1-D tensor.")

    base = _single_latent_without_batch(w0).to(dtype=torch.float32)
    direction = _expand_direction(direction.to(device=base.device, dtype=base.dtype), base.shape)
    steps = lambdas.to(device=base.device, dtype=base.dtype)

    view_shape = (len(steps),) + (1,) * base.ndim
    latents = base.unsqueeze(0) + steps.view(view_shape) * direction.unsqueeze(0)
    return _decode_trajectory(generator, latents, decode_fn, return_grid)


def generate_explicit_trajectory(
    generator: nn.Module,
    w0: torch.Tensor,
    direction1: torch.Tensor,
    direction2: torch.Tensor,
    alpha1: float,
    alpha2: float,
    t: torch.Tensor,
    direction3: torch.Tensor | None = None,
    alpha3: float = 0.0,
    decode_fn: ImageDecodeFn | None = None,
    return_grid: bool = True,
) -> torch.Tensor:
    """Generates images along a Legendre explicit nonlinear trajectory.

    The path is:

        V(t) = w0 + alpha1 * P1(t) * n1
                  + alpha2 * P2(t) * n2
                  + alpha3 * P3(t) * n3

    with:

        P1(t) = t
        P2(t) = 0.5 * (3t^2 - 1)
        P3(t) = 0.5 * (5t^3 - 3t)

    Args are the same as ``generate_linear_trajectory``. The third direction is
    optional to keep existing two-axis calls valid.
    """

    if t.ndim != 1:
        raise ValueError("t must be a 1-D tensor.")
    if torch.any(t < -1.0) or torch.any(t > 1.0):
        raise ValueError("Explicit Legendre trajectory expects t in [-1, 1].")

    base = _single_latent_without_batch(w0).to(dtype=torch.float32)
    n1 = _expand_direction(direction1.to(device=base.device, dtype=base.dtype), base.shape)
    n2 = _expand_direction(direction2.to(device=base.device, dtype=base.dtype), base.shape)
    n3 = None
    if direction3 is not None:
        n3 = _expand_direction(direction3.to(device=base.device, dtype=base.dtype), base.shape)

    t = t.to(device=base.device, dtype=base.dtype)
    p1 = t
    p2 = 0.5 * (3.0 * t.square() - 1.0)
    p3 = 0.5 * (5.0 * t.pow(3) - 3.0 * t)

    view_shape = (len(t),) + (1,) * base.ndim
    latents = (
        base.unsqueeze(0)
        + float(alpha1) * p1.view(view_shape) * n1.unsqueeze(0)
        + float(alpha2) * p2.view(view_shape) * n2.unsqueeze(0)
    )
    if n3 is not None and alpha3 != 0.0:
        latents = latents + float(alpha3) * p3.view(view_shape) * n3.unsqueeze(0)
    return _decode_trajectory(generator, latents, decode_fn, return_grid)


class ToyGenerator(nn.Module):
    """Tiny decoder used by ``main()`` so this file can run without a GAN."""

    def __init__(self, latent_dim: int = 512, image_size: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 3 * image_size * image_size),
            nn.Tanh(),
        )
        self.image_size = image_size

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        flat = w.reshape(w.shape[0], -1)
        image = self.net(flat)
        return image.view(w.shape[0], 3, self.image_size, self.image_size)


def main() -> None:
    """Minimal runnable example with dummy W, w0, and generator."""

    torch.manual_seed(7)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_dim = 512
    weight = torch.randn(1024, latent_dim, device=device)
    generator = ToyGenerator(latent_dim=latent_dim, image_size=32).to(device).eval()
    w0 = torch.randn(1, latent_dim, device=device)

    directions = extract_sefa_directions(weight, k=3)
    lambdas = torch.linspace(-3.0, 3.0, 7, device=device)
    t = torch.linspace(-1.0, 1.0, 7, device=device)

    linear_grid = generate_linear_trajectory(
        generator=generator,
        w0=w0,
        direction=directions[0],
        lambdas=lambdas,
    )
    explicit_grid = generate_explicit_trajectory(
        generator=generator,
        w0=w0,
        direction1=directions[0],
        direction2=directions[1],
        direction3=directions[2],
        alpha1=2.0,
        alpha2=-1.5,
        alpha3=0.75,
        t=t,
    )

    print("=== Explicit SeFa Mapping Demo ===")
    print(f"directions shape: {tuple(directions.shape)}")
    print(f"linear grid shape: {tuple(linear_grid.shape)}")
    print(f"explicit grid shape: {tuple(explicit_grid.shape)}")
    print(f"direction dot product: {torch.dot(directions[0], directions[1]).item():.6e}")


if __name__ == "__main__":
    main()
