# Function-Space Riemannian Verification

This experiment compares each random feasible function direction with Riemannian optimization initialized from that same direction.

## Scope

RQ1/RQ2 test structured GAN variation and held-out generalization. They do not by themselves establish semantic discovery; controlled semantic evidence requires the separate leave-one-out experiment.

## Geometry

- Ambient function dimension: `512`
- Known semantic rank: `5`
- Feasible sphere dimension: `506`
- Maximum known/complement Hilbert inner product: `2.671e-16`
- Maximum complement Gram error: `1.887e-15`

## Paired held-out result

- Restarts: `1`
- Mean RO - Random test objective: `9.559204e-01`
- Bootstrap 95% interval: `[9.559204e-01, 9.559204e-01]`
- Fraction of restarts with positive held-out improvement: `1.000`
- Total runtime (seconds): `10.95`

## Configuration

```json
{
  "model": "e4e_ffhq_encode",
  "latent_dimension": 512,
  "known_semantics": [
    "age",
    "gender",
    "pose",
    "glasses",
    "smile"
  ],
  "num_search": 2,
  "num_test": 4,
  "num_restarts": 1,
  "max_iterations": 2,
  "epsilon_h": 0.1,
  "curvature_beta": 0.1,
  "objective_resolution": 64,
  "learning_rate": 0.25,
  "min_learning_rate": 0.0001,
  "armijo_constant": 0.0001,
  "line_search_steps": 6,
  "gradient_tolerance": 1e-06,
  "microbatch": 1,
  "seed": 0,
  "trunc_psi": 0.7,
  "trunc_layers": 8,
  "gpu_id": "0",
  "traversal_multipliers": [
    -3,
    -2,
    -1,
    0,
    1,
    2,
    3
  ],
  "grid_restarts": 1,
  "grid_test_samples": 2,
  "output_dir": "/home/rz/raymond_project/f_sefa/experiments_ro_verification/outputs",
  "smoke": true
}
```
