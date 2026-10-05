"""Configuration for the Stage 02 function-space convex-hull experiment."""

import argparse
import os
from dataclasses import asdict, dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class ExperimentConfig:
    model: str = "e4e_ffhq_encode"
    basis: str = "legendre"
    latent_dimension: int = 512
    lambda_0: float = 1.0
    epsilon_h: float = 0.005
    num_rays: int = 2
    probe_distances_h: tuple = (
        0.0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32,
        0.40, 0.48, 0.56, 0.64, 0.80, 0.96, 1.12, 1.28
    )
    seed: int = 0
    trunc_psi: float = 0.7
    trunc_layers: int = 8
    gpu_id: str = "0"
    membership_tolerance: float = 1e-9
    output_dir: str = EXPERIMENT_DIR

    def to_dict(self):
        result = asdict(self)
        result["probe_distances_h"] = list(self.probe_distances_h)
        return result


def parse_args():
    defaults = ExperimentConfig()
    parser = argparse.ArgumentParser(
        description=(
            "Stage 02: paired crossings of a known-semantic convex hull "
            "constructed entirely in raw-Legendre function space."
        )
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--basis", default=defaults.basis, choices=("legendre",))
    parser.add_argument("--latent-dimension", type=int, default=defaults.latent_dimension)
    parser.add_argument("--lambda-0", type=float, default=defaults.lambda_0)
    parser.add_argument("--epsilon-h", type=float, default=defaults.epsilon_h)
    parser.add_argument("--num-rays", type=int, default=defaults.num_rays)
    parser.add_argument(
        "--probe-distances-h",
        type=float,
        nargs="+",
        default=list(defaults.probe_distances_h),
        help="Hilbert distances beyond the candidate hull boundary.",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--trunc-psi", type=float, default=defaults.trunc_psi)
    parser.add_argument("--trunc-layers", type=int, default=defaults.trunc_layers)
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument(
        "--membership-tolerance", type=float,
        default=defaults.membership_tolerance,
    )
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    return ExperimentConfig(
        model=values.model,
        basis=values.basis,
        latent_dimension=values.latent_dimension,
        lambda_0=values.lambda_0,
        epsilon_h=values.epsilon_h,
        num_rays=values.num_rays,
        probe_distances_h=tuple(values.probe_distances_h),
        seed=values.seed,
        trunc_psi=values.trunc_psi,
        trunc_layers=values.trunc_layers,
        gpu_id=values.gpu_id,
        membership_tolerance=values.membership_tolerance,
        output_dir=os.path.abspath(values.output_dir),
    )

