# Preliminary Finding 01 - Brightness / Illumination Direction

Recorded: 2026-10-04

## Status

This is a reproducible end-to-end smoke-run finding and a candidate semantic
direction. It is not yet a formal multi-restart result.

The experiment began from a random feasible function-space direction and used
Riemannian gradient ascent on

\[
\mathcal M=\{v:U^TGv=0,\;v^TGv=1\},
\]

where \(U\) contains the five known SeFa directions Age, Gender, Pose,
Glasses, and Smile. The resulting traversal showed a strong, smooth, and
cross-sample-consistent brightness / illumination response.

## Geometry verification

- Ambient Legendre coefficient dimension: 512.
- Known semantic rank: 5.
- Orthogonal-complement dimension: 507.
- Maximum \(U^TGQ\) error: \(2.671\times10^{-16}\).
- Maximum \(Q^TGQ-I\) error: \(1.887\times10^{-15}\).
- Final optimized-direction orthogonality error: \(7.57\times10^{-17}\).
- Final Hilbert unit-norm error: \(4.34\times10^{-8}\) in float32 execution.

Thus, the observed image response was not obtained by leaving the constrained
Hilbert manifold or reintroducing one of the five excluded directions.

## Smoke-run configuration

- Search anchors: 2.
- Held-out anchors: 4.
- Independent restarts: 1.
- RO iterations: 2.
- Objective resolution: 64 x 64.
- Hilbert central-difference step: \(\epsilon_H=0.1\).
- Traversal distances: \(-0.3,-0.2,-0.1,0,0.1,0.2,0.3\).

## Held-out Random vs RO result

| Metric | Random feasible direction | RO direction | Change |
|---|---:|---:|---:|
| Objective \(J=E-\beta C\) | -2.1981 | -1.2476 | +0.9505 |
| Image effect \(E\) | 0.6430 | 1.3696 | +0.7266 |
| Curvature \(C\) | 28.4117 | 26.1726 | -2.2391 |
| Mean response cosine | 0.1752 | 0.6263 | +0.4511 |

The optimized direction increased observable image response, reduced the local
curvature penalty, and produced a more consistent response across held-out
faces.

## Brightness monotonicity from the rendered traversal

Mean sRGB luminance was measured directly from the saved comparison grid. This
is a post-hoc diagnostic on resized rendered images, not a metric used by the
optimizer.

| Held-out identity | Direction | Luminance slope per unit \(d_H\) | Linear \(R^2\) | Strictly monotonic |
|---|---|---:|---:|---|
| Seed 000 | Random | 0.0859 | 0.9739 | Yes |
| Seed 000 | RO | 0.4630 | 0.9998 | Yes |
| Seed 001 | Random | 0.1045 | 0.9993 | Yes |
| Seed 001 | RO | 0.3099 | 0.9971 | Yes |

For the two rendered held-out identities, RO amplified the luminance-response
slope by approximately 5.39x and 2.97x relative to their paired random starting
direction. Both RO traversals were strictly monotonic over all seven Hilbert
distances.

## Interpretation

The current evidence supports the following preliminary statement:

> Hilbert-constrained Riemannian optimization recovered a stable candidate
> brightness / illumination direction from the orthogonal complement of five
> known SeFa semantics, without retraining the generator.

The result does not yet establish a final semantic discovery claim. A formal
claim requires more held-out identities, independent initializations, direct
luminance measurements on unresized generated images, identity-preservation
checks, and evidence that the response is not merely global RGB gain or image
saturation.

## Evidence files

- `outputs/grids/random_vs_ro/run_00.png`: paired Random and RO traversals.
- `outputs/paired_metrics.csv`: held-out quantitative metrics.
- `outputs/runs/run_00/convergence.csv`: optimization trajectory.
- `outputs/runs/run_00/direction.npz`: initial and optimized directions.
- `outputs/experiment_manifest.json`: model, geometry, and configuration.

## Registered next experiment - Sequential RO

After formally validating the brightness direction \(v_1\), subsequent
directions will be searched sequentially. Each accepted direction is appended
to the excluded function subspace:

\[
U_{k+1}=[U,v_1,\ldots,v_k].
\]

The next optimization is constrained by

\[
U_{k+1}^TGv_{k+1}=0,
\qquad
v_{k+1}^TGv_{k+1}=1.
\]

This prevents later restarts from repeatedly converging to the brightness
direction and allows the experiment to search for additional independent GAN
variation factors. Every accepted direction will retain the same held-out,
constraint, convergence, consistency, and image-traversal validation protocol.

