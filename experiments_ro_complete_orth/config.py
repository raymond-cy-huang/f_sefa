"""Configuration for sequential complete-orthogonal Riemannian search."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))

SEMANTIC_RANKS = {
    "gender": 1,
    "age": 3,
    "pose": 7,
    "glasses": 10,
    "smile": 18,
}


@dataclass(frozen=True)
class CompleteOrthConfig:
    model: str = "e4e_ffhq_encode"
    latent_dimension: int = 512
    known_semantics: tuple[str, ...] = (
        "age", "gender", "pose", "glasses", "smile"
    )
    num_rays: int = 10
    num_search: int = 32
    num_test: int = 100
    max_iterations: int = 40
    epsilon_h: float = 0.1
    curvature_beta: float = 0.1
    objective_resolution: int = 128
    learning_rate: float = 0.25
    min_learning_rate: float = 1e-4
    armijo_constant: float = 1e-4
    line_search_steps: int = 6
    max_attempts_per_ray: int = 5
    minimum_search_improvement: float = 1e-4
    gradient_tolerance: float = 1e-6
    microbatch: int = 1
    seed: int = 0
    trunc_psi: float = 0.7
    trunc_layers: int = 8
    gpu_id: str = "0"
    traversal_multipliers: tuple[float, ...] = (-3, -2, -1, 0, 1, 2, 3)
    grid_test_samples: int = 4
    output_dir: str = EXPERIMENT_DIR
    smoke: bool = False
    quick_all: bool = False
    resume: bool = False

    def to_dict(self):
        payload = asdict(self)
        payload["known_semantics"] = list(self.known_semantics)
        payload["traversal_multipliers"] = list(self.traversal_multipliers)
        return payload


def parse_args() -> CompleteOrthConfig:
    defaults = CompleteOrthConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Find multiple function-space GAN variation rays sequentially, "
            "excluding every accepted ray from all subsequent searches."
        )
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--num-rays", type=int, default=defaults.num_rays)
    parser.add_argument("--num-search", type=int, default=defaults.num_search)
    parser.add_argument("--num-test", type=int, default=defaults.num_test)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument("--epsilon-h", type=float, default=defaults.epsilon_h)
    parser.add_argument("--curvature-beta", type=float, default=defaults.curvature_beta)
    parser.add_argument(
        "--objective-resolution", type=int, default=defaults.objective_resolution
    )
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument(
        "--min-learning-rate", type=float, default=defaults.min_learning_rate
    )
    parser.add_argument("--line-search-steps", type=int, default=defaults.line_search_steps)
    parser.add_argument(
        "--max-attempts-per-ray", type=int, default=defaults.max_attempts_per_ray
    )
    parser.add_argument(
        "--minimum-search-improvement",
        type=float,
        default=defaults.minimum_search_improvement,
    )
    parser.add_argument(
        "--gradient-tolerance", type=float, default=defaults.gradient_tolerance
    )
    parser.add_argument("--microbatch", type=int, default=defaults.microbatch)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trunc-psi", type=float, default=defaults.trunc_psi)
    parser.add_argument("--trunc-layers", type=int, default=defaults.trunc_layers)
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument(
        "--known-semantics",
        nargs="+",
        choices=tuple(SEMANTIC_RANKS),
        default=list(defaults.known_semantics),
    )
    parser.add_argument(
        "--traversal-multipliers",
        type=float,
        nargs="+",
        default=list(defaults.traversal_multipliers),
    )
    parser.add_argument(
        "--grid-test-samples", type=int, default=defaults.grid_test_samples
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run three rays with 2 search anchors, 4 test anchors, and 2 RO steps.",
    )
    parser.add_argument(
        "--quick-all",
        action="store_true",
        help=(
            "Run all 507 complement rays with the validated quick settings "
            "(2 search, 4 test, 2 RO steps, one rendered identity)."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume after the highest ray recorded in complete_orth_summary.csv.",
    )
    values = parser.parse_args()
    if values.smoke:
        values.num_rays = min(values.num_rays, 3)
        values.num_search = min(values.num_search, 2)
        values.num_test = min(values.num_test, 4)
        values.max_iterations = min(values.max_iterations, 2)
        values.objective_resolution = min(values.objective_resolution, 64)
        values.grid_test_samples = min(values.grid_test_samples, 2)
    if values.quick_all:
        values.num_rays = 512 - len(values.known_semantics)
        values.num_search = 2
        values.num_test = 4
        values.max_iterations = 2
        values.objective_resolution = 64
        values.grid_test_samples = 1

    config = CompleteOrthConfig(
        model=values.model,
        known_semantics=tuple(values.known_semantics),
        num_rays=values.num_rays,
        num_search=values.num_search,
        num_test=values.num_test,
        max_iterations=values.max_iterations,
        epsilon_h=values.epsilon_h,
        curvature_beta=values.curvature_beta,
        objective_resolution=values.objective_resolution,
        learning_rate=values.learning_rate,
        min_learning_rate=values.min_learning_rate,
        line_search_steps=values.line_search_steps,
        max_attempts_per_ray=values.max_attempts_per_ray,
        minimum_search_improvement=values.minimum_search_improvement,
        gradient_tolerance=values.gradient_tolerance,
        microbatch=values.microbatch,
        seed=values.seed,
        trunc_psi=values.trunc_psi,
        trunc_layers=values.trunc_layers,
        gpu_id=values.gpu_id,
        traversal_multipliers=tuple(values.traversal_multipliers),
        grid_test_samples=values.grid_test_samples,
        output_dir=os.path.abspath(values.output_dir),
        smoke=values.smoke,
        quick_all=values.quick_all,
        resume=values.resume,
    )
    validate_config(config)
    return config


def validate_config(config: CompleteOrthConfig):
    for name in (
        "num_rays", "num_search", "num_test", "max_iterations",
        "objective_resolution", "microbatch", "max_attempts_per_ray",
    ):
        if getattr(config, name) <= 0:
            raise ValueError(f"{name} must be positive.")
    if config.num_rays > config.latent_dimension - len(config.known_semantics):
        raise ValueError("num_rays exceeds the available orthogonal-complement dimension.")
    if config.epsilon_h <= 0 or config.learning_rate <= 0:
        raise ValueError("epsilon_h and learning_rate must be positive.")
    if config.curvature_beta < 0:
        raise ValueError("curvature_beta must be non-negative.")
    if config.minimum_search_improvement < 0:
        raise ValueError("minimum_search_improvement must be non-negative.")

