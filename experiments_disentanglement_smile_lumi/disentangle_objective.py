"""Differentiable Smile/luminance response objectives on a Hilbert subspace."""

from __future__ import annotations

import numpy as np
import torch

from objective import GANImageObjective, AnchorSet


class DisentanglementObjective(GANImageObjective):
    def __init__(
        self,
        generator,
        complement_q,
        epsilon_h,
        resolution,
        microbatch,
        mode,
        smile_scale=1.0,
        luminance_scale=1.0,
        curvature_scale=1.0,
        leakage_weight=1.0,
        curvature_weight=0.05,
    ):
        super().__init__(
            generator,
            complement_q,
            epsilon_h,
            curvature_beta=0.0,
            resolution=resolution,
            microbatch=microbatch,
        )
        if mode not in ("smile", "luminance"):
            raise ValueError("mode must be `smile` or `luminance`.")
        self.mode = mode
        self.smile_scale = max(float(smile_scale), 1e-12)
        self.luminance_scale = max(float(luminance_scale), 1e-12)
        self.curvature_scale = max(float(curvature_scale), 1e-12)
        self.leakage_weight = float(leakage_weight)
        self.curvature_weight = float(curvature_weight)
        self._templates = {}
        self._luma_weights = torch.tensor(
            [0.2126, 0.7152, 0.0722], dtype=torch.float32
        ).view(1, 3)

    def register_smile_template(self, anchors: AnchorSet, flat_responses: np.ndarray):
        response = torch.from_numpy(np.asarray(flat_responses)).float().reshape(
            -1, 3, self.resolution, self.resolution
        )
        response = response - response.mean(dim=(2, 3), keepdim=True)
        y0, y1 = int(round(0.52 * self.resolution)), int(round(0.84 * self.resolution))
        x0, x1 = int(round(0.24 * self.resolution)), int(round(0.76 * self.resolution))
        mouth = response[:, :, y0:y1, x0:x1]
        flat = mouth.flatten(1)
        flat = flat / torch.clamp(torch.linalg.vector_norm(flat, dim=1, keepdim=True), min=1e-12)
        self._templates[anchors.name] = flat.cpu()

    def _semantic_terms(self, first_response, second_response, templates):
        centered = first_response - first_response.mean(dim=(2, 3), keepdim=True)
        y0, y1 = int(round(0.52 * self.resolution)), int(round(0.84 * self.resolution))
        x0, x1 = int(round(0.24 * self.resolution)), int(round(0.76 * self.resolution))
        mouth = centered[:, :, y0:y1, x0:x1].flatten(1)
        template = templates.to(device=mouth.device, dtype=mouth.dtype)
        smile_amplitude = torch.sum(mouth * template, dim=1) / np.sqrt(mouth.shape[1])
        smile_power = smile_amplitude.square().mean()
        channel_mean = first_response.mean(dim=(2, 3))
        luma = torch.sum(
            channel_mean * self._luma_weights.to(channel_mean.device), dim=1
        )
        luminance_power = luma.square().mean()
        effect = first_response.square().flatten(1).mean(dim=1).mean()
        curvature = second_response.square().flatten(1).mean(dim=1).mean()
        normalized_smile = smile_power / self.smile_scale
        normalized_luminance = luminance_power / self.luminance_scale
        normalized_curvature = curvature / self.curvature_scale
        if self.mode == "smile":
            objective = (
                normalized_smile
                - self.leakage_weight * normalized_luminance
                - self.curvature_weight * normalized_curvature
            )
        else:
            objective = (
                normalized_luminance
                - self.leakage_weight * normalized_smile
                - self.curvature_weight * normalized_curvature
            )
        return {
            "objective": objective,
            "effect": effect,
            "curvature": curvature,
            "smile_power": smile_power,
            "luminance_power": luminance_power,
            "normalized_smile_power": normalized_smile,
            "normalized_luminance_power": normalized_luminance,
        }

    def value_and_gradient(self, reduced, anchors):
        if anchors.name not in self._templates:
            raise KeyError(f"No Smile template registered for `{anchors.name}`.")
        point = reduced.detach().to(self.device, dtype=torch.float32).clone()
        point.requires_grad_(True)
        keys = (
            "objective", "effect", "curvature", "smile_power", "luminance_power",
            "normalized_smile_power", "normalized_luminance_power",
        )
        totals = {key: 0.0 for key in keys}
        count = len(anchors)
        templates = self._templates[anchors.name]
        for start in range(0, count, self.microbatch):
            end = min(start + self.microbatch, count)
            weight = float(end - start) / float(count)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            _, _, first, second = self._batch_terms(point, coefficients, base)
            terms = self._semantic_terms(first, second, templates[start:end])
            (weight * terms["objective"]).backward()
            for key in keys:
                totals[key] += weight * float(terms[key].detach().cpu())
        if point.grad is None or not torch.all(torch.isfinite(point.grad)):
            raise RuntimeError("Disentanglement objective produced a non-finite gradient.")
        return totals, point.grad.detach()

    @torch.no_grad()
    def evaluate(self, reduced, anchors, collect_responses=False):
        if anchors.name not in self._templates:
            raise KeyError(f"No Smile template registered for `{anchors.name}`.")
        point = reduced.detach().to(self.device, dtype=torch.float32)
        keys = (
            "objective", "effect", "curvature", "smile_power", "luminance_power",
            "normalized_smile_power", "normalized_luminance_power",
        )
        totals = {key: 0.0 for key in keys}
        responses = []
        count = len(anchors)
        templates = self._templates[anchors.name]
        for start in range(0, count, self.microbatch):
            end = min(start + self.microbatch, count)
            weight = float(end - start) / float(count)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            _, _, first, second = self._batch_terms(point, coefficients, base)
            terms = self._semantic_terms(first, second, templates[start:end])
            for key in keys:
                totals[key] += weight * float(terms[key].cpu())
            if collect_responses:
                responses.append(first.flatten(1).cpu().numpy())
        return totals, (np.concatenate(responses) if responses else None)

