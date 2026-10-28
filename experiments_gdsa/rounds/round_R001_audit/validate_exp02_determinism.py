"""Audit repeatability of Experiment 02 generator decoding within one process."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
RUN_PATH = ROOT / "experiments_gdsa/02_generator_jacobian_ro/run.py"
spec = importlib.util.spec_from_file_location("gdsa_exp02", RUN_PATH)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = module
spec.loader.exec_module(module)


def main() -> None:
    config = module.Config(num_seeds=1)
    generator = module.load_generator(config.model)
    generator.eval()
    for parameter in generator.parameters():
        parameter.requires_grad_(False)
    point = module.sample_w(generator, 1, config)
    first = module.decode_low(generator, point, config.response_resolution)
    second = module.decode_low(generator, point, config.response_resolution)
    difference = (first - second).abs()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)
    deterministic_first = module.decode_low(generator, point, config.response_resolution)
    deterministic_second = module.decode_low(generator, point, config.response_resolution)
    deterministic_difference = (deterministic_first - deterministic_second).abs()
    result = {
        "default": {
            "torch_equal": bool(torch.equal(first, second)),
            "max_abs_difference": float(difference.max().cpu()),
            "mean_abs_difference": float(difference.mean().cpu()),
        },
        "deterministic_algorithms": {
            "torch_equal": bool(torch.equal(deterministic_first, deterministic_second)),
            "max_abs_difference": float(deterministic_difference.max().cpu()),
            "mean_abs_difference": float(deterministic_difference.mean().cpu()),
        },
        "randomize_noise": False,
        "interpretation": "Within-process repeat decode check at identical W.",
    }
    output = Path(__file__).resolve().parent / "exp02_determinism.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
