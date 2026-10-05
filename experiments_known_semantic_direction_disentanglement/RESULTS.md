# Known SeFa Semantic Direction Disentanglement

Recorded: 2026-10-04

## Experiment

The experiment scanned all 507 existing complete-orthogonal rays on the same
four held-out anchors. The five project-confirmed SeFa directions were Gender
rank 1, Age rank 3, Pose rank 7, Glasses rank 10, and Smile rank 18.

For each target, the top 25% image-response matches (127 rays) were retained as
candidates. Each candidate was projected into the target SeFa direction's
G-orthogonal complement. The selected projected partner and target then formed
a two-dimensional Hilbert plane. A unit-circle search retained at least 70% of
the linearized target response and minimized mean response to the other four
known semantics.

## Held-out results

| Semantic | Partner | Rotation | Target retained | Known nuisance before | Known nuisance after | Nuisance reduction |
|---|---|---:|---:|---:|---:|---:|
| Smile | ray046 | 57.00 deg | 0.697 | 0.0614 | 0.0351 | 42.8% |
| Gender | ray386 | -19.25 deg | 0.834 | 0.0745 | 0.0551 | 26.1% |
| Age | ray232 | -17.25 deg | 0.749 | 0.0947 | 0.0599 | 36.8% |
| Glasses | ray223 | -14.00 deg | 0.726 | 0.0462 | 0.0294 | 36.4% |
| Pose | ray224 | -23.50 deg | 0.810 | 0.1111 | 0.0869 | 21.7% |

All target and complement pairs are numerically G-orthogonal, with absolute
cross cosine below `3.2e-16`.

Smile's generator-reevaluated target retention is `0.697`, slightly below the
`0.70` linearized selection constraint because the response of a rotated
direction is not perfectly linear at finite step size. This deviation is
reported rather than rounded into a pass.

## Interpretation

This experiment demonstrates a reproducible reduction in leakage into the
other four *known SeFa response templates*. It does not prove independence
from every unknown visual semantic. Visual traversal remains the primary check
for identity-dependent effects, illumination, color, and artifacts that are
not represented by the five-template nuisance score.

## Outputs

- `outputs/all_ray_semantic_scores.csv`: all 507 rays against all five semantics.
- `outputs/summary.csv`: consolidated held-out result.
- `outputs/<semantic>/candidates.csv`: 127 candidates and projected scores.
- `outputs/<semantic>/rotation_curve.csv`: complete S1 search.
- `outputs/<semantic>/metrics.csv`: generator-reevaluated final metrics.
- `outputs/<semantic>/directions.npz`: SeFa, partner, disentangled, and complement directions.
- `outputs/<semantic>/traversal.png`: four-anchor visual comparison.
- `outputs/<semantic>/manifest.json`: semantic-specific provenance.

