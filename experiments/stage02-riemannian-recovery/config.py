"""Configuration for Stage 02 Riemannian semantic-direction recovery."""

import argparse
import os
from dataclasses import asdict, dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "e4e_ffhq_encode"
    basis: str = "legendre"
    latent_dimension: int = 512
    semantic: str = "smile"
    num_samples: int = 32
    delta: float = 1.0
    learning_rate: float = 0.5
    max_iterations: int = 500
    gradient_tolerance: float = 1e-12
    seed: int = 0
    trunc_psi: float = 0.7
    trunc_layers: int = 8
    gpu_id: str = "0"
    lambdas: tuple = (-6.0, -3.0, 0.0, 3.0, 6.0)
    normalized_displacement_ablation: bool = True
    output_dir: str = EXPERIMENT_DIR

    def to_dict(self):
        result = asdict(self)
        result["lambdas"] = list(self.lambdas)
        return result


def parse_args():
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Recover a known SeFa semantic direction with manual Riemannian "
            "gradient descent on the coefficient-space unit sphere."
        )
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--basis", default=defaults.basis, choices=("legendre",))
    parser.add_argument("--latent-dimension", type=int, default=defaults.latent_dimension)
    parser.add_argument(
        "--semantic",
        default=defaults.semantic,
        choices=("smile",),
    )
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--delta", type=float, default=defaults.delta)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--max-iterations", type=int, default=defaults.max_iterations)
    parser.add_argument(
        "--gradient-tolerance", type=float, default=defaults.gradient_tolerance
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trunc-psi", type=float, default=defaults.trunc_psi)
    parser.add_argument("--trunc-layers", type=int, default=defaults.trunc_layers)
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(defaults.lambdas))
    parser.add_argument(
        "--no-normalized-displacement-ablation",
        action="store_true",
        help="Disable the additional normalized-displacement covariance diagnostic.",
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    return ExperimentConfig(
        model=values.model,
        basis=values.basis,
        latent_dimension=values.latent_dimension,
        semantic=values.semantic,
        num_samples=values.num_samples,
        delta=values.delta,
        learning_rate=values.learning_rate,
        max_iterations=values.max_iterations,
        gradient_tolerance=values.gradient_tolerance,
        seed=values.seed,
        trunc_psi=values.trunc_psi,
        trunc_layers=values.trunc_layers,
        gpu_id=values.gpu_id,
        lambdas=tuple(values.lambdas),
        normalized_displacement_ablation=(
            not values.no_normalized_displacement_ablation
        ),
        output_dir=os.path.abspath(values.output_dir),
    )

