"""Configuration for known SeFa semantic-direction disentanglement."""

from __future__ import annotations

import os
from dataclasses import dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class Config:
    model: str = "e4e_ffhq_encode"
    epsilon_h: float = 0.1
    resolution: int = 64
    microbatch: int = 1
    candidate_quantile: float = 0.75
    minimum_target_retention: float = 0.70
    angle_samples: int = 721
    source_dir: str = os.path.abspath(
        os.path.join(EXPERIMENT_DIR, "..", "experiments_ro_complete_orth")
    )
    output_dir: str = os.path.join(EXPERIMENT_DIR, "outputs")

