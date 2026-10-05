# Stage 02 — Function-Space Convex-Hull Crossing

This experiment studies paired crossings of a finite reference region:

\[
C_{\mathrm{known}}^H
=
\operatorname{Conv}\{p_{\mathrm{gender}}^H,
p_{\mathrm{age}}^H,
p_{\mathrm{pose}}^H,
p_{\mathrm{glasses}}^H,
p_{\mathrm{smile}}^H\}.
\]

It does **not** claim that this region is the unknown true semantic-valid
region. The crossed boundary is `boundary(C_known^H)`, not a discovered
validity boundary.

## Fixed inputs

- Generator: `e4e_ffhq_encode` StyleGAN2 decoder.
- Base source: one seeded, truncated W code (`seed=0`, `trunc_psi=0.7`,
  `trunc_layers=8`).
- Function basis: raw Legendre orders 0 through 511.
- Endpoint intensity: `lambda_0=1.0`.
- Paired crossing distance: `epsilon_H=0.005`.
- Random rays: 2.
- Exploratory outside distances from the candidate boundary:
  `0, 0.005, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.40, 0.48, 0.56,
  0.64, 0.80, 0.96, 1.12, 1.28` in Hilbert norm.
- Semantic directions:
  - Eigen 001 — Gender
  - Eigen 003 — Age
  - Eigen 007 — Pose
  - Eigen 010 — Glasses
  - Eigen 018 — Smile

The actual source eigen indices, eigenvalues, checkpoint checksum, simplex
singular values, and numerical tolerances are recorded in
`experiment_config.json` after a run.

## Function-space-only construction

The Stage 01 map is

\[
\Phi(w)(x)=\sum_{n=0}^{511}w_nP_n(x).
\]

After mapping the base and five source directions, all endpoint construction,
convex combinations, SVD rank checks, barycentric membership checks, ray
construction, boundary computation, and distance calculations operate only on
raw-Legendre function coefficients. No latent interpolation, extrapolation,
normalization, distance, hull, or boundary operation participates.

The raw-Legendre Gram matrix is

\[
G_{nn}=\frac{2}{2n+1},
\qquad
d_H(a,b)=\sqrt{(a-b)^TG(a-b)}.
\]

For each random Dirichlet ray, the implementation finds the exact candidate
hull boundary in barycentric coordinates and constructs a paired inside and
outside point at equal Hilbert distance `epsilon_H` from that boundary.

Only after all function-space coordinates are fixed does the experiment apply
the Stage 01 weighted projection inverse `Psi` to obtain the generator input.
The `Phi(Psi(c))` round trip is measured only as an inverse-consistency metric;
it never modifies the original sample.

## Run

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  experiments/stage02-convex-hull/run.py
```

## Outputs

```text
vertices.png
paired_inside.png
paired_boundary.png
paired_outside.png
paired_crossings.png
boundary_probe.png
sampling.csv
experiment_config.json
comparison_figures/
images/
```

`paired_crossings.png` contains one row per ray with:

```text
Original | Inside | Boundary | Outside
```

The `comparison_figures/` directory also contains one full-size comparison
figure per ray and one outward boundary-probe sweep per ray.

`boundary_probe.png` continues along each fixed function-space ray after
crossing `boundary(C_known^H)`. It is an exploratory visual sweep for locating
possible degradation. Visual validity must be assessed manually; the experiment
does not use a classifier or automatic artifact threshold and therefore does
not automatically declare any probe distance to be the true validity boundary.

## Current two-ray observation

For the fixed seed-0 base image, both rays remain visually coherent through
approximately `+0.32 H`. Larger distances produce increasing scene, lighting,
and identity drift. Under a strict image-quality interpretation, the first
clear structural degradation is visually bracketed around:

- ray 00: between `+1.12 H` and `+1.28 H`;
- ray 01: between `+0.80 H` and `+0.96 H`.

These are manual, ray-specific exploratory brackets, not estimates of a global
semantic-valid boundary. A stricter identity- or attribute-preservation
definition would place the transition earlier and requires a separately stated
validity criterion.

The original 20-ray results are retained under
`outputs/previous-20-rays/`; they are not overwritten by this two-ray run.

