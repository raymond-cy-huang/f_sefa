# Stage 02 - Riemannian Semantic-Direction Recovery

This minimum viable experiment validates manifold-constrained optimization in
the existing F-SeFa explicit function-space representation. It uses the known
SeFa Smile direction (Eigen 018) as the target and Age, Pose, Glasses, and
Gender as leakage distractors.

The purpose is not to claim that Riemannian optimization outperforms the
closed-form eigendecomposition. With the current strictly linear Stage 01 map,
the centered semantic displacements should be identical, the covariance should
be rank one, and all three baselines should recover the same direction.

## Representation

The raw-Legendre coefficient representation is

\[
\Phi(w)(x)=\sum_{n=0}^{511}w_nP_n(x).
\]

For sampled StyleGAN W points, the primary displacements are

\[
\Delta c_i=
\frac{\Phi(w_i+\delta d_s)-\Phi(w_i-\delta d_s)}{2\delta}.
\]

No per-sample normalization is applied to the primary covariance:

\[
C_s=\frac1N\sum_i\Delta c_i\Delta c_i^T.
\]

Normalized displacements are retained only as an optional ablation.

## Sphere optimization

The MVP uses the canonical Euclidean unit sphere in raw coefficient
coordinates:

\[
\mathcal M=S^{D-1}=\{u:u^Tu=1\}.
\]

For

\[
f(u)=-u^TC_su,
\]

the implementation uses

\[
\nabla_Ef(u)=-2C_su,
\qquad
\operatorname{grad}f(u)=
(I-uu^T)\nabla_Ef(u),
\]

followed by the normalization retraction

\[
R_u(v)=\frac{u+v}{\|u+v\|_2}.
\]

This matches the sphere-submanifold construction in Nicolas Boumal's
Riemannian optimization tutorial. The raw-Legendre Gram metric is not used as
the sphere constraint in this MVP; that would define a different manifold
experiment.

## Baselines

1. Direct mapped SeFa direction `normalize(Phi_dir(d_smile))`.
2. Principal eigenvector of `C_s`.
3. Manual Riemannian gradient descent on the sphere.

The optional `pymanopt` path is not required. The current environment does not
have it installed, so the manual implementation is the primary validation.

## Run

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  experiments/stage02-riemannian-recovery/run.py
```

## Outputs

```text
metrics.csv
convergence.csv
convergence.png
delta_cosine_matrix.csv
delta_cosine_matrix.png
traversal_metrics.csv
semantic_traversal.png
experiment_config.json
SUMMARY.md
images/
```

