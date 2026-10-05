"""Configuration for the Stage 02 semantic-intensity experiment."""

import argparse
import os
from dataclasses import asdict, dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class ExperimentConfig:
    semantic: str = "smile"
    basis: str = "legendre"
    model: str = "e4e_ffhq_encode"
    latent_dimension: int = 512
    num_samples: int = 1
    lambdas: tuple = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
    trunc_psi: float = 0.7
    trunc_layers: int = 8
    seed: int = 0
    gpu_id: str = "0"
    output_dir: str = os.path.join(EXPERIMENT_DIR, "outputs")
    enable_lpips: bool = False

    def to_dict(self):
        result = asdict(self)
        result["lambdas"] = list(self.lambdas)
        return result


def parse_args():
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description="Stage 02: known semantic intensity in raw-basis Hilbert geometry.")
    parser.add_argument("--semantic", default=defaults.semantic,
                        choices=("age", "gender", "smile", "glasses", "pose"))
    parser.add_argument("--basis", default=defaults.basis,
                        choices=("legendre", "chebyshev"))
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--latent-dimension", type=int,
                        default=defaults.latent_dimension)
    parser.add_argument("--num-samples", type=int, default=defaults.num_samples)
    parser.add_argument("--lambdas", type=float, nargs="+",
                        default=list(defaults.lambdas))
    parser.add_argument("--trunc-psi", type=float, default=defaults.trunc_psi)
    parser.add_argument("--trunc-layers", type=int,
                        default=defaults.trunc_layers)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument(
        "--enable-lpips", action="store_true",
        help="Compute LPIPS only when the optional lpips package is installed.")
    values = parser.parse_args()
    output_dir = values.output_dir or defaults.output_dir
    return ExperimentConfig(
        semantic=values.semantic.lower(),
        basis=values.basis.lower(),
        model=values.model,
        latent_dimension=values.latent_dimension,
        num_samples=values.num_samples,
        lambdas=tuple(values.lambdas),
        trunc_psi=values.trunc_psi,
        trunc_layers=values.trunc_layers,
        seed=values.seed,
        gpu_id=values.gpu_id,
        output_dir=os.path.abspath(output_dir),
        enable_lpips=values.enable_lpips,
    )
