# Experiment 02: Generator Jacobian and Riemannian Optimization

## Objective

Combine the second and third GDSA stages in one executable pilot:

1. Measure the generator directional Jacobian `J_G(w) v` without constructing
   the full image Jacobian.
2. Use the sinh-induced metric to compute a position-dependent direction.
3. Compare Euclidean, fixed-metric, and dynamic-metric paths at identical
   linear semantic-proxy progress.

The target direction is the existing post-annotated SeFa Smile rank. Its sign
and image-level semantic validity are not assumed by this experiment.

## Compared paths

| Method | Direction rule |
| --- | --- |
| B1 Euclidean | normalized fixed SeFa target direction |
| B2 fixed metric | minimum-cost direction under `g(w_0)` |
| B3 dynamic metric | minimum-cost direction under the current `g(w_k)` |

Every direction is scaled so `d_t.T @ v = 1`. Consequently, a step of
`delta_tau` produces exactly the same linear target-proxy progress for all
methods.

The local geometry-aware direction is

```text
v = g(w)^(-1) d_t / (d_t.T @ g(w)^(-1) d_t).
```

Pymanopt independently solves the equivalent local unit-metric-budget problem
on a sphere. Its result is compared against the analytical direction.

## Generator Jacobian measurement

For each path endpoint and direction, the experiment estimates the
matrix-free directional Jacobian with a central difference:

```text
J_G(w) v ~= [G(w + epsilon*v) - G(w - epsilon*v)] / (2*epsilon).
```

The reported RMS measures local output sensitivity. It is an image-response
magnitude, not an independent Smile score.

## Run

From the repository root:

```bash
conda run -n gdsa python experiments_gdsa/02_generator_jacobian_ro/run.py
```

The default pilot uses two latent seeds, three alpha values, five matched
semantic-progress steps, and 64-pixel Jacobian measurements. Outputs are saved
under the ignored `outputs/` directory.

## Interpretation boundary

This experiment can establish that the direction and generator response depend
on position under the selected coordinate metric. It cannot establish Smile
quality or disentanglement without an independent image-level semantic
evaluator.

## Recorded pilot

The default pilot completed successfully on an RTX 3070 with two W-space
samples, five steps to `tau = 0.5`, and `alpha` values 0, 0.5, and 1.0.

```text
status: PASS
maximum target-progress error: 2.220e-16
minimum Pymanopt/analytical direction cosine: 1.000000
runtime after CUDA-extension warmup: 6.86 seconds
```

Mean results across the two latent samples:

| alpha | Method | Dynamic-metric path energy | W displacement | Start JVP RMS | End JVP RMS | Direction start/end cosine |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 0.0 | B1 Euclidean | `5.000e-2` | `0.5000` | `1.185e-1` | `1.224e-1` | `1.000000` |
| 0.0 | B2 fixed | `5.000e-2` | `0.5000` | `1.184e-1` | `1.224e-1` | `1.000000` |
| 0.0 | B3 dynamic | `5.000e-2` | `0.5000` | `1.185e-1` | `1.224e-1` | `1.000000` |
| 0.5 | B1 Euclidean | `5.412e-2` | `0.5000` | `1.185e-1` | `1.224e-1` | `1.000000` |
| 0.5 | B2 fixed | `5.301e-2` | `0.5025` | `1.208e-1` | `1.244e-1` | `1.000000` |
| 0.5 | B3 dynamic | `5.301e-2` | `0.5025` | `1.208e-1` | `1.247e-1` | `0.999985` |
| 1.0 | B1 Euclidean | `7.981e-2` | `0.5000` | `1.184e-1` | `1.224e-1` | `1.000000` |
| 1.0 | B2 fixed | `5.985e-2` | `0.5137` | `1.205e-1` | `1.248e-1` | `1.000000` |
| 1.0 | B3 dynamic | `5.984e-2` | `0.5137` | `1.205e-1` | `1.245e-1` | `0.999859` |

At `alpha = 0`, all methods reduce to the same Euclidean path, as required.
At `alpha = 1`, B2 and B3 reduce the measured dynamic-metric energy by about
25% relative to B1 at matched target-proxy progress. B2 and B3 remain almost
identical in this short pilot, so the observed change is currently attributable
mainly to static metric weighting, not to repeated position-dependent updates.

The directional generator response changes along every path, but the pilot has
no independent image-level Smile measurement. Visual differences in the saved
grid are subtle and do not support a semantic-quality claim.
