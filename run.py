"""Local Riemannian refinement around the known F-SeFa Smile direction."""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch

EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, ".."))
VERIFICATION_DIR = os.path.join(ROOT_DIR, "experiments_ro_verification")
for path in (EXPERIMENT_DIR, ROOT_DIR, VERIFICATION_DIR):
    if path not in sys.path:
        sys.path.append(path)

from config import Config
from evaluation import save_csv, save_json, save_traversal_grid
from geometry import legendre_gram_diagonal
from objective import GANImageObjective
from optimizer import optimize_sphere
from utils import factorize_weight, load_generator


SEMANTIC_RANKS = {"gender": 1, "age": 3, "pose": 7, "glasses": 10, "smile": 18}
NUISANCE_NAMES = ("gender", "age", "pose", "glasses")


def normalize_h(vector, gram):
    return vector / np.sqrt(max(float(np.dot(vector * gram, vector)), 1e-30))


def inner_h(left, right, gram):
    return float(np.dot(left * gram, right))


def sample_anchors(generator, count, seed):
    device = next(generator.parameters()).device
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)
    z = torch.randn(count, generator.z_space_dim, generator=rng, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(w, trunc_psi=0.7, trunc_layers=8)
    if w.ndim == 3:
        w = w[:, 0, :]
    return w.detach().cpu().float()


def sefa_directions(generator):
    _, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    order = np.argsort(eigenvalues)[::-1]
    directions = {}
    metadata = {}
    gram = legendre_gram_diagonal(boundaries.shape[1])
    for name, rank in SEMANTIC_RANKS.items():
        source = int(order[rank])
        direction = normalize_h(boundaries[source].reshape(-1), gram)
        directions[name] = direction
        metadata[name] = {
            "rank": rank,
            "source_index": source,
            "eigenvalue": float(eigenvalues[source]),
        }
    return directions, metadata, gram


def centered_templates(responses):
    flat = responses.reshape(len(responses), -1).astype(np.float64)
    flat = flat - flat.reshape(len(flat), 3, -1).mean(axis=2, keepdims=True).reshape(len(flat), -1)
    return flat / np.maximum(np.linalg.norm(flat, axis=1, keepdims=True), 1e-12)


def response_amplitudes(flat, templates):
    return np.sum(flat * templates, axis=1) / np.sqrt(flat.shape[1])


class LocalSmileObjective(GANImageObjective):
    def __init__(self, generator, q, config, templates, scales, x0, tau, beta, gamma):
        super().__init__(generator, q, config.epsilon_h, 0.0, config.resolution, config.microbatch)
        self.templates = {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in templates.items()}
        self.scales = scales
        self.x0 = torch.as_tensor(x0, dtype=torch.float32, device=self.device)
        self.tau = float(tau)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.eta = float(config.eta)
        self.luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=self.device).view(1, 3)

    def _terms(self, first, second, point):
        centered = first - first.mean(dim=(2, 3), keepdim=True)
        flat = centered.flatten(1)
        smile_amp = torch.sum(flat * self.templates["smile"], dim=1) / math.sqrt(flat.shape[1])
        smile_mean = smile_amp.mean() / self.scales["smile"]
        consistency = (smile_amp > 0).float().mean()
        nuisance_terms = {}
        nuisance_values = []
        for name in NUISANCE_NAMES:
            amp = torch.sum(flat * self.templates[name], dim=1) / math.sqrt(flat.shape[1])
            value = amp.square().mean() / self.scales[name]
            nuisance_terms[name] = value
            nuisance_values.append(value)
        known_leakage = torch.stack(nuisance_values).mean()
        channel_mean = first.mean(dim=(2, 3))
        luma = torch.sum(channel_mean * self.luma_weights, dim=1)
        lumi_leakage = luma.square().mean() / self.scales["luminance"]
        cosine = torch.dot(point, self.x0)
        locality = torch.relu(self.tau - cosine).square()
        objective = smile_mean - self.beta * known_leakage - self.gamma * lumi_leakage - self.eta * locality
        return {
            "objective": objective,
            "smile_mean": smile_mean,
            "smile_consistency": consistency,
            "known_leakage": known_leakage,
            "luminance_leakage": lumi_leakage,
            "locality_penalty": locality,
            "cosine_to_fsefa": cosine,
            "curvature": second.square().flatten(1).mean(),
            **{f"{name}_leakage": value for name, value in nuisance_terms.items()},
        }

    def value_and_gradient(self, reduced, anchors):
        point = reduced.detach().to(self.device, dtype=torch.float32).clone()
        point.requires_grad_(True)
        keys = ("objective", "smile_mean", "smile_consistency", "known_leakage", "luminance_leakage", "locality_penalty", "cosine_to_fsefa", "curvature") + tuple(f"{n}_leakage" for n in NUISANCE_NAMES)
        totals = {k: 0.0 for k in keys}
        for start in range(0, len(anchors), self.microbatch):
            end = min(start + self.microbatch, len(anchors))
            weight = (end - start) / len(anchors)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            _, _, first, second = self._batch_terms(point, coefficients, base)
            terms = self._terms(first, second, point)
            (weight * terms["objective"]).backward()
            for key in keys:
                totals[key] += weight * float(terms[key].detach().cpu())
        return totals, point.grad.detach()

    @torch.no_grad()
    def evaluate(self, reduced, anchors, collect_responses=False):
        point = reduced.detach().to(self.device, dtype=torch.float32)
        keys = ("objective", "smile_mean", "smile_consistency", "known_leakage", "luminance_leakage", "locality_penalty", "cosine_to_fsefa", "curvature") + tuple(f"{n}_leakage" for n in NUISANCE_NAMES)
        totals = {k: 0.0 for k in keys}
        responses = []
        for start in range(0, len(anchors), self.microbatch):
            end = min(start + self.microbatch, len(anchors))
            weight = (end - start) / len(anchors)
            coefficients = anchors.coefficients[start:end].to(self.device)
            base = anchors.base_low_resolution[start:end].to(self.device)
            _, _, first, second = self._batch_terms(point, coefficients, base)
            terms = self._terms(first, second, point)
            for key in keys:
                totals[key] += weight * float(terms[key].cpu())
            if collect_responses:
                responses.append(first.flatten(1).cpu().numpy())
        return totals, (np.concatenate(responses) if responses else None)


class FullSphereGeometry:
    def __init__(self, q, gram):
        self.q, self.gram = q, gram
    def constraint_errors(self, direction):
        norm = float(np.dot(direction * self.gram, direction))
        return {"hilbert_norm_squared": norm, "hilbert_norm_error": abs(norm - 1.0), "orthogonality_l2": 0.0, "orthogonality_max_abs": 0.0}


def make_templates(generator, directions, anchor_set, config):
    templates = {}
    baseline = {}
    evaluator = GANImageObjective(
        generator, torch.eye(config.dimension, device=next(generator.parameters()).device),
        config.epsilon_h, 0.0, config.resolution, config.microbatch,
    )
    for name, direction in directions.items():
        _, flat = evaluator.evaluate(
            torch.as_tensor(direction, dtype=torch.float32, device=evaluator.device),
            anchor_set, collect_responses=True,
        )
        template = centered_templates(flat.reshape(len(anchor_set), 3, config.resolution, config.resolution))
        templates[name] = template
        amp = response_amplitudes(flat, template)
        baseline[name] = max(float(np.mean(amp ** 2)), 1e-8)
    return templates, baseline


def metrics_from_response(flat, templates, directions, gram, d0, scales):
    result = {}
    smile_amp = response_amplitudes(flat, templates["smile"])
    result["smile_mean"] = float(np.mean(smile_amp) / scales["smile"])
    result["smile_median"] = float(np.median(smile_amp) / scales["smile"])
    result["smile_consistency"] = float(np.mean(smile_amp > 0))
    result["smile_std"] = float(np.std(smile_amp) / scales["smile"])
    for name in NUISANCE_NAMES:
        amp = response_amplitudes(flat, templates[name])
        result[f"{name}_leakage"] = float(np.mean(amp ** 2) / scales[name])
    result["known_leakage"] = float(np.mean([result[f"{n}_leakage"] for n in NUISANCE_NAMES]))
    images = flat.reshape(len(flat), 3, -1)
    luma = np.sum(images.mean(axis=2) * np.asarray([0.2126, 0.7152, 0.0722]), axis=1)
    result["luminance_leakage"] = float(np.mean(luma ** 2) / scales["luminance"])
    result["hilbert_cos_to_fsefa"] = inner_h(d0, directions, gram)
    result["hilbert_angle_to_fsefa"] = float(np.degrees(np.arccos(np.clip(result["hilbert_cos_to_fsefa"], -1, 1))))
    return result


def main():
    config = Config()
    started = time.perf_counter()
    for directory in ("directions", "traversals", "metrics", "figures"):
        os.makedirs(os.path.join(config.output_dir, directory), exist_ok=True)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    generator = load_generator(config.model)
    device = next(generator.parameters()).device
    directions, metadata, gram = sefa_directions(generator)
    d0 = directions["smile"]
    q = torch.diag(torch.as_tensor(1.0 / np.sqrt(gram), dtype=torch.float32, device=device))
    x0 = torch.as_tensor(np.sqrt(gram) * d0, dtype=torch.float32, device=device)
    geometry = FullSphereGeometry(q.detach().cpu().double().numpy(), gram)

    all_anchors = sample_anchors(generator, config.num_search + config.num_validation + config.num_test, config.seed + 11)
    search = all_anchors[:config.num_search]
    validation = all_anchors[config.num_search:config.num_search + config.num_validation]
    test = all_anchors[config.num_search + config.num_validation:]
    anchor_builder = GANImageObjective(generator, q, config.epsilon_h, 0.0, config.resolution, config.microbatch)
    search_set = anchor_builder.prepare_anchors("search", search)
    validation_set = anchor_builder.prepare_anchors("validation", validation)
    test_set = anchor_builder.prepare_anchors("test", test)
    templates, baseline_power = make_templates(generator, directions, search_set, config)
    scales = {name: baseline_power[name] for name in SEMANTIC_RANKS}
    scales["luminance"] = max(1e-8, float(np.mean([
        np.mean((response(generator, d0, search_set, config).reshape(len(search_set), 3, -1).mean(axis=2) @ np.asarray([0.2126, 0.7152, 0.0722])) ** 2
    ])))

    # Reference responses are evaluated on validation/test using templates built only on search.
    reference_flat = {}
    for split, aset in (("validation", validation_set), ("test", test_set)):
        reference_flat[split] = {}
        for name, direction in directions.items():
            reference_flat[split][name] = response(generator, direction, aset, config).reshape(len(aset), -1)

    summary_rows = []
    all_candidate_rows = []
    experiment_configs = [(f"tau_{tau:.2f}", tau, config.beta, config.gamma) for tau in config.taus]
    experiment_configs += [("ablation_beta0", 0.90, 0.0, config.gamma), ("ablation_gamma0", 0.90, config.beta, 0.0)]
    for setting, tau, beta, gamma in experiment_configs:
        setting_dir = os.path.join(config.output_dir, setting)
        os.makedirs(setting_dir, exist_ok=True)
        local_objective = LocalSmileObjective(generator, q, config, templates, scales, x0, tau, beta, gamma)
        candidates = []
        for candidate_id in range(config.num_candidates):
            rng = np.random.default_rng(config.seed + candidate_id + int(tau * 1000) + int(beta * 100) + int(gamma * 10))
            z = torch.as_tensor(rng.normal(size=config.dimension), dtype=torch.float32, device=device)
            z = z - x0 * torch.dot(x0, z)
            z = z / torch.linalg.vector_norm(z)
            alpha = [0.02, 0.05, 0.10, 0.15][candidate_id % 4]
            init = torch.cos(torch.tensor(alpha, device=device)) * x0 + torch.sin(torch.tensor(alpha, device=device)) * z
            optimized, history, runtime = optimize_sphere(
                local_objective, search_set, init, geometry,
                learning_rate=config.learning_rate, min_learning_rate=config.min_learning_rate,
                armijo_constant=1e-4, line_search_steps=config.line_search_steps,
                max_iterations=config.max_iterations, gradient_tolerance=config.gradient_tolerance,
            )
            direction = (q @ optimized).detach().cpu().double().numpy()
            search_metrics, search_flat = local_objective.evaluate(optimized, search_set, True)
            validation_metrics, validation_flat = local_objective.evaluate(optimized, validation_set, True)
            test_metrics, test_flat = local_objective.evaluate(optimized, test_set, True)
            target_response = float(validation_metrics["smile_mean"])
            row = {
                "setting": setting, "tau": tau, "beta": beta, "gamma": gamma,
                "candidate_id": candidate_id, "runtime_seconds": runtime,
                "iterations": len(history) - 1,
                "initial_objective": history[0]["objective"],
                "final_objective": history[-1]["objective"],
                "gradient_norm": history[-1]["riemannian_gradient_norm"],
                "norm_error": abs(float(np.dot(direction * gram, direction)) - 1.0),
                "cos_h": inner_h(direction, d0, gram),
                "angular_deviation": float(np.degrees(np.arccos(np.clip(inner_h(direction, d0, gram), -1, 1)))),
                "search_objective": search_metrics["objective"],
                "validation_objective": validation_metrics["objective"],
                "test_objective": test_metrics["objective"],
            }
            for split, flat, metrics in (("search", search_flat, search_metrics), ("validation", validation_flat, validation_metrics), ("test", test_flat, test_metrics)):
                sem = metrics_from_response(flat, templates, direction, gram, d0, scales)
                for key, value in sem.items(): row[f"{split}_{key}"] = value
            all_candidate_rows.append(row)
            candidates.append((row, direction, optimized, history))
            np.savez_compressed(os.path.join(setting_dir, f"candidate_{candidate_id:02d}.npz"), direction=direction, gram_diagonal=gram)
            save_csv(os.path.join(setting_dir, f"candidate_{candidate_id:02d}_convergence.csv"), history)
        candidates.sort(key=lambda item: (item[0]["validation_known_leakage"] + 0.5 * item[0]["validation_luminance_leakage"], -item[0]["validation_smile_mean"]))
        best_row, best_direction, _, best_history = candidates[0]
        np.savez_compressed(os.path.join(config.output_dir, "directions", f"{setting}_best.npz"), direction=best_direction, gram_diagonal=gram)
        distances = [config.epsilon_h * x for x in (-3, -2, -1, 0, 1, 2, 3)]
        rows = []
        for index in range(min(8, len(test_set))):
            coefficient = test_set.coefficients[index]
            for name, direction in (("SeFa-Smile", directions["smile"]), ("F-SeFa-Smile", d0), ("Local-RO-best", best_direction)):
                renderer = GANImageObjective(generator, torch.eye(config.dimension, device=device), config.epsilon_h, 0.0, config.resolution, config.microbatch)
                images = renderer.render_traversal(torch.as_tensor(direction, dtype=torch.float32, device=device), coefficient, distances)
                rows.append((f"seed {index:03d} / {name}", images))
        save_traversal_grid(os.path.join(config.output_dir, "traversals", f"{setting}.png"), rows, distances)
        summary_rows.append({"setting": setting, "tau": tau, "beta": beta, "gamma": gamma, "best_candidate_id": best_row["candidate_id"], **{k: v for k, v in best_row.items() if k.startswith("validation_") or k in ("cos_h", "angular_deviation", "norm_error", "gradient_norm", "iterations")}})
        print(f"Completed {setting}: candidate={best_row['candidate_id']}", flush=True)

    save_csv(os.path.join(config.output_dir, "metrics", "all_candidates.csv"), all_candidate_rows)
    save_csv(os.path.join(config.output_dir, "metrics", "final_summary.csv"), summary_rows)
    save_json(os.path.join(config.output_dir, "manifest.json"), {"experiment": "local-smile-disentanglement", "semantic_ranks": SEMANTIC_RANKS, "num_search": config.num_search, "num_validation": config.num_validation, "num_test": config.num_test, "num_candidates": config.num_candidates, "runtime_seconds": time.perf_counter() - started, "settings": [x[0] for x in experiment_configs]})
    with open(os.path.join(config.output_dir, "SUMMARY.md"), "w", encoding="utf-8") as handle:
        handle.write("# Local Smile Disentanglement\n\n")
        handle.write("This experiment starts from F-SeFa Smile and performs local soft Riemannian refinement. It does not search the complete orthogonal complement.\n\n")
        handle.write("## Outputs\n\nSee `metrics/final_summary.csv`, `metrics/all_candidates.csv`, and `traversals/`.\n")
    print(f"Finished in {time.perf_counter() - started:.1f} seconds", flush=True)


def response(generator, direction, anchors, config):
    import torch
    device = next(generator.parameters()).device
    evaluator = GANImageObjective(generator, torch.eye(config.dimension, device=device), config.epsilon_h, 0.0, config.resolution, config.microbatch)
    _, flat = evaluator.evaluate(torch.as_tensor(direction, dtype=torch.float32, device=device), anchors, collect_responses=True)
    return flat


if __name__ == "__main__":
    main()
