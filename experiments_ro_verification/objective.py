"""Differentiable GAN image-response objective for function-space directions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as functional


@dataclass
class AnchorSet:
    name: str
    coefficients: torch.Tensor  # CPU tensor [N, D]
    base_low_resolution: torch.Tensor  # CPU tensor [N, C, H, W]

    def __len__(self) -> int:
        return int(self.coefficients.shape[0])


class GANImageObjective:
    """J(v)=effect(v)-beta*curvature(v), with v=Qz in function space."""

    def __init__(
        self,
        generator,
        complement_q: torch.Tensor,
        epsilon_h: float,
        curvature_beta: float,
        resolution: int,
        microbatch: int = 1,
    ):
        self.generator = generator
        self.device = next(generator.parameters()).device
        self.q = complement_q.to(device=self.device, dtype=torch.float32)
        self.epsilon_h = float(epsilon_h)
        self.curvature_beta = float(curvature_beta)
        self.resolution = int(resolution)
        self.microbatch = int(microbatch)
        self.num_layers = int(generator.num_layers)
        self.latent_dimension = int(generator.w_space_dim)
        if self.q.shape[0] != self.latent_dimension:
            raise ValueError("Complement basis and generator W dimension disagree.")
        for parameter in self.generator.parameters():
            parameter.requires_grad_(False)
        self.generator.eval()

    def _psi(self, coefficients: torch.Tensor) -> torch.Tensor:
        """Exact inverse of the raw Legendre coefficient representation.

        In Stage 01, Phi stores W directly as Legendre coefficients, so Psi is
        the identity in coefficient coordinates. Keeping this operation explicit
        documents where a future non-trivial differentiable inverse belongs.
        """
        return coefficients

    def _to_wp(self, coefficients: torch.Tensor) -> torch.Tensor:
        latent = self._psi(coefficients)
        return latent[:, None, :].expand(-1, self.num_layers, -1)

    def _decode_low(self, coefficients: torch.Tensor) -> torch.Tensor:
        output = self.generator.synthesis(self._to_wp(coefficients))["image"]
        if output.ndim != 4:
            raise ValueError(f"Expected image batch [N,C,H,W], got {tuple(output.shape)}.")
        if output.shape[-2:] == (self.resolution, self.resolution):
            return output
        return functional.interpolate(
            output,
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )

    def prepare_anchors(self, name: str, coefficients: torch.Tensor) -> AnchorSet:
        coefficients = coefficients.detach().to(dtype=torch.float32, device="cpu")
        base_images = []
        with torch.no_grad():
            for start in range(0, len(coefficients), self.microbatch):
                batch = coefficients[start : start + self.microbatch].to(self.device)
                base_images.append(self._decode_low(batch).detach().cpu())
        return AnchorSet(name, coefficients, torch.cat(base_images, dim=0))

    def _batch_terms(
        self,
        reduced: torch.Tensor,
        coefficients: torch.Tensor,
        base_low: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        direction = self.q @ reduced
        plus = coefficients + self.epsilon_h * direction[None, :]
        minus = coefficients - self.epsilon_h * direction[None, :]
        paired = self._decode_low(torch.cat([plus, minus], dim=0))
        plus_image, minus_image = torch.chunk(paired, 2, dim=0)
        first_response = (plus_image - minus_image) / (2.0 * self.epsilon_h)
        second_response = (
            plus_image - 2.0 * base_low + minus_image
        ) / (self.epsilon_h**2)
        effect_per_sample = first_response.square().flatten(1).mean(dim=1)
        curvature_per_sample = second_response.square().flatten(1).mean(dim=1)
        return (
            effect_per_sample.mean(),
            curvature_per_sample.mean(),
            first_response,
            second_response,
        )

    def value_and_gradient(
        self, reduced: torch.Tensor, anchors: AnchorSet
    ) -> tuple[dict[str, float], torch.Tensor]:
        point = reduced.detach().to(self.device, dtype=torch.float32).clone()
        point.requires_grad_(True)
        totals = {"objective": 0.0, "effect": 0.0, "curvature": 0.0}
        count = len(anchors)
        for start in range(0, count, self.microbatch):
            end = min(start + self.microbatch, count)
            weight = float(end - start) / float(count)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            effect, curvature, _, _ = self._batch_terms(point, coefficients, base)
            objective = effect - self.curvature_beta * curvature
            (weight * objective).backward()
            totals["effect"] += weight * float(effect.detach().cpu())
            totals["curvature"] += weight * float(curvature.detach().cpu())
            totals["objective"] += weight * float(objective.detach().cpu())
        if point.grad is None or not torch.all(torch.isfinite(point.grad)):
            raise RuntimeError("Objective produced a missing or non-finite gradient.")
        return totals, point.grad.detach()

    @torch.no_grad()
    def evaluate(
        self,
        reduced: torch.Tensor,
        anchors: AnchorSet,
        collect_responses: bool = False,
    ) -> tuple[dict[str, float], np.ndarray | None]:
        point = reduced.detach().to(self.device, dtype=torch.float32)
        totals = {"objective": 0.0, "effect": 0.0, "curvature": 0.0}
        responses = []
        count = len(anchors)
        for start in range(0, count, self.microbatch):
            end = min(start + self.microbatch, count)
            weight = float(end - start) / float(count)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            effect, curvature, first, _ = self._batch_terms(point, coefficients, base)
            objective = effect - self.curvature_beta * curvature
            totals["effect"] += weight * float(effect.cpu())
            totals["curvature"] += weight * float(curvature.cpu())
            totals["objective"] += weight * float(objective.cpu())
            if collect_responses:
                responses.append(first.flatten(1).cpu().numpy())
        return totals, (np.concatenate(responses, axis=0) if responses else None)

    @torch.no_grad()
    def render_traversal(
        self,
        reduced: torch.Tensor,
        coefficient: torch.Tensor,
        hilbert_distances: list[float] | tuple[float, ...],
    ) -> list[np.ndarray]:
        point = reduced.detach().to(self.device, dtype=torch.float32)
        direction = self.q @ point
        base = coefficient.detach().to(self.device, dtype=torch.float32).reshape(1, -1)
        images = []
        for distance in hilbert_distances:
            output = self.generator.synthesis(
                self._to_wp(base + float(distance) * direction[None, :])
            )["image"]
            image = output[0].detach().clamp(-1.0, 1.0)
            image = ((image + 1.0) * 127.5).round().byte()
            images.append(image.permute(1, 2, 0).cpu().numpy())
        return images

