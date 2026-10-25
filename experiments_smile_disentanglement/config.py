"""Configuration for local Riemannian Smile disentanglement."""

from __future__ import annotations

import os
from dataclasses import dataclass


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))


@dataclass(frozen=True)
class Config:
    model: str = "e4e_ffhq_encode"
    dimension: int = 512
    epsilon_h: float = 0.1
    resolution: int = 64
    microbatch: int = 1
    num_search: int = 16
    num_validation: int = 16
    num_test: int = 32
    num_candidates: int = 16
    max_iterations: int = 12
    learning_rate: float = 0.15
    min_learning_rate: float = 1e-4
    line_search_steps: int = 6
    gradient_tolerance: float = 1e-5
    taus: tuple[float, ...] = (0.95, 0.90, 0.85)
    beta: float = 1.0
    gamma: float = 0.5
    eta: float = 100.0
    seed: int = 20261005
    output_dir: str = os.path.join(EXPERIMENT_DIR, "outputs")

