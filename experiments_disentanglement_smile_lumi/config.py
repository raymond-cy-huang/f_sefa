"""Configuration for Smile + luminance candidate scanning."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict, dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class ScanConfig:
    model: str = "e4e_ffhq_encode"
    epsilon_h: float = 0.1
    response_resolution: int = 64
    microbatch: int = 1
    quantile_threshold: float = 0.75
    gpu_id: str = "0"
    source_dir: str = os.path.abspath(
        os.path.join(EXPERIMENT_DIR, "..", "experiments_ro_complete_orth")
    )
    output_dir: str = os.path.join(EXPERIMENT_DIR, "outputs")

    def to_dict(self):
        return asdict(self)


def parse_args() -> ScanConfig:
    defaults = ScanConfig()
    parser = argparse.ArgumentParser(
        description="Scan all complete-orthogonal rays for joint Smile and luminance response."
    )
    parser.add_argument("--model", default=defaults.model)
    parser.add_argument("--epsilon-h", type=float, default=defaults.epsilon_h)
    parser.add_argument(
        "--response-resolution", type=int, default=defaults.response_resolution
    )
    parser.add_argument("--microbatch", type=int, default=defaults.microbatch)
    parser.add_argument(
        "--quantile-threshold", type=float, default=defaults.quantile_threshold
    )
    parser.add_argument("--gpu-id", default=defaults.gpu_id)
    parser.add_argument("--source-dir", default=defaults.source_dir)
    parser.add_argument("--output-dir", default=defaults.output_dir)
    values = parser.parse_args()
    config = ScanConfig(
        model=values.model,
        epsilon_h=values.epsilon_h,
        response_resolution=values.response_resolution,
        microbatch=values.microbatch,
        quantile_threshold=values.quantile_threshold,
        gpu_id=values.gpu_id,
        source_dir=os.path.abspath(values.source_dir),
        output_dir=os.path.abspath(values.output_dir),
    )
    if config.epsilon_h <= 0 or config.response_resolution <= 0 or config.microbatch <= 0:
        raise ValueError("epsilon_h, response_resolution, and microbatch must be positive.")
    if not 0.0 < config.quantile_threshold < 1.0:
        raise ValueError("quantile_threshold must lie strictly between 0 and 1.")
    return config

