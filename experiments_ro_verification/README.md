# Function-Space Riemannian Verification

This experiment tests whether Hilbert-constrained Riemannian optimization (RO)
finds stronger, smoother, and more transferable GAN image variations than the
same randomly initialized feasible directions.

The optimization variable is a function-space direction represented by raw
Legendre coefficients. The Hilbert inner product is

\[
\langle a,b\rangle_H=a^TGb,
\qquad G_{kk}=\frac{2}{2k+1}.
\]

For known semantic directions collected in \(U\), the feasible manifold is

\[
\mathcal M=\{v:U^TGv=0,\;v^TGv=1\}.
\]

A basis \(Q\) is constructed such that \(U^TGQ=0\) and \(Q^TGQ=I\). Thus
\(v=Qz\), \(\|z\|_2=1\), and optimization is standard sphere RO in reduced
coordinates while preserving the original Hilbert geometry exactly.

## Image objective

For a frozen generator and function coefficient anchor \(c_n\),

\[
I_n^\pm=G_{GAN}(\Psi(c_n\pm\epsilon v)).
\]

The optimized objective is

\[
J(v)=E(v)-\beta C(v),
\]

where \(E\) is the mean squared central first image response and \(C\) is the
mean squared central second image response. Images are deterministically
downsampled before these quantities are measured.

Cross-sample response consistency is evaluated only after optimization. It is
not included in the first objective and is reported as a proxy, not as proof of
semantic identity.

## Recorded finding

The first end-to-end smoke run produced a smooth and held-out-consistent
brightness / illumination candidate direction. The complete numerical record,
scope, caveats, and evidence paths are in `BRIGHTNESS_FINDING.md`.

The registered follow-up is Sequential RO: after a direction is formally
accepted, it is appended to the excluded Hilbert subspace before searching for
the next direction. This prevents repeated convergence to the same brightness
factor and supports discovery of multiple mutually G-orthogonal candidates.

## Validation levels

1. Geometry: complement orthogonality, Hilbert unit norm, tangent gradients,
   retraction, finite-difference derivatives, and Taylor remainder.
2. Optimization: paired before/after objective from the same random feasible
   initialization.
3. Generalization: all metrics are recomputed on held-out latent anchors.
4. Image evidence: Random and RO traversals are rendered side by side at equal
   Hilbert distances.

RQ1/RQ2 establish structured GAN variation, not semantic discovery by
themselves. Controlled semantic evidence requires the subsequent leave-one-out
experiment, where a hidden known semantic direction provides ground truth.

## CPU unit tests

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  -m unittest discover -s experiments_ro_verification/tests -v
```

## Small end-to-end GPU run

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  experiments_ro_verification/run.py --smoke
```

The smoke run uses two search anchors, four held-out anchors, one paired
restart, two RO iterations, and a 64-pixel objective.

## Formal RQ1/RQ2 run

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  experiments_ro_verification/run.py
```

The default configuration uses 32 search anchors, 100 held-out anchors, ten
paired restarts, deterministic generator noise, and a 128-pixel image
objective. On an 8 GB GPU, generation uses microbatches of one.

## Main outputs

```text
outputs/
  experiment_manifest.json
  search_anchors.npy
  test_anchors.npy
  paired_metrics.csv
  aggregate_metrics.json
  paired_comparison.png
  SUMMARY.md
  runs/run_XX/
    convergence.csv
    convergence.png
    direction.npz
    metrics.json
  grids/random_vs_ro/run_XX.png
```

Each `direction.npz` stores both the random initialization and optimized
direction in reduced coordinates and in ambient function coefficients.

