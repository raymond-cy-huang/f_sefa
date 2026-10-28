# GDSA Experiments

The latest experiment is GDSA-E01 Semantic Response Operator. Its mathematical
controls, generator smoke, and contribution verdict are documented in
`rounds/round_R005_semantic_response_operator/experimental_report_20261025_R05_E01.md`.

All new GDSA experiments live in this directory.

## Setup dependencies

The setup samples use the following packages installed in the `gdsa` Conda
environment:

```bash
python -m pip install "pymanopt[torch]==2.2.1" "scipy==1.15.3"
```

## Directory convention

Each experiment uses a zero-padded numeric prefix and a descriptive name:

```text
NN_experiment_name/
```

Experiment-specific source code, configuration, outputs, and findings should
remain inside that experiment's directory. Generated outputs should not be
shared implicitly between experiments; cross-experiment inputs must be declared
explicitly in the consuming experiment.

## Experiments

- `00_setup_ro_sample/` — verifies the Pymanopt installation with an
  optimization problem on a sphere manifold.
- `00_setup_jocobian_sample/` — verifies
  `scipy.differentiate.jacobian` against an analytical Jacobian.
- `01_sinh_none_linear_mapping_inversion/` — sinh nonlinear mapping and
  analytical inversion.
- `02_generator_jacobian_ro/` — combines matrix-free generator directional
  Jacobian measurements with local Riemannian direction optimization.

## Research framework

GDSA uses one position-dependent geometry for three related optimization
problems. Let

- `w` be a 512-dimensional StyleGAN W-space latent vector;
- `L` be a 512 by 512 orthogonal Legendre transform with `L.T @ L = I`;
- `alpha` be the strength of the nonlinear coordinate map; and
- `d_t` be a normalized target semantic direction.

The common coordinate map, its Jacobian, and the induced metric are

```text
h = T_alpha(w) = sinh(alpha * Lw) / alpha
J_T(w) = diag(cosh(alpha * Lw)) @ L
g(w) = J_T(w).T @ J_T(w)
```

The `alpha = 0` case is defined by continuous extension:

```text
T_0(w) = Lw
J_T(w) = L
g(w) = I
```

This metric is the pull-back of the Euclidean metric through `T_alpha`. It is
position dependent but intrinsically flat because `T_alpha` is a global smooth
reparameterization. The initial experiments therefore study its local
direction weighting, conditioning, numerical behavior, and finite-step paths;
they do not assume that sinh alone creates semantic meaning or generator
curvature.

### Problem 1: semantic direction optimization

For the linear target proxy

```text
s_t(w) = d_t.T @ (w - w_0),
```

choose the direction with the greatest local target progress under a geometric
step budget:

```text
maximize    d_t.T @ v
subject to  v.T @ g(w_k) @ v <= epsilon^2
```

The closed-form solution is

```text
v_k = epsilon * g(w_k)^(-1) @ d_t
      / sqrt(d_t.T @ g(w_k)^(-1) @ d_t).
```

This tests whether the local direction changes meaningfully with position and
`alpha`. The optimized quantity remains a latent semantic proxy; image-level
semantic improvement requires independent evaluation.

### Problem 2: semantic intensity optimization

Given a target proxy intensity `tau`, solve

```text
minimize    0.5 * ||T_alpha(w) - T_alpha(w_0)||^2
subject to  d_t.T @ (w - w_0) = tau.
```

The baseline for a normalized direction is `w_linear = w_0 + tau * d_t`.
Evaluation compares target error, geometric distance, W-space displacement,
generated-image change, and independently measured semantic consistency.

### Problem 3: semantic disentanglement optimization

Let `d_t` be the target direction and let `d_1, d_2, ...` be nuisance semantic
directions. The endpoint problem is

```text
minimize    0.5 * ||T_alpha(w) - T_alpha(w_0)||^2
subject to  d_t.T @ (w - w_0) = tau
            d_j.T @ (w - w_0) = 0 for each nuisance direction d_j.
```

For `A = [d_t, d_1, d_2, ...]` and `b = [1, 0, 0, ...].T`, the local
minimum-cost constrained direction is

```text
v_k = g(w_k)^(-1) @ A
      @ (A.T @ g(w_k)^(-1) @ A)^(-1) @ b.
```

The equality constraints control linear latent proxies. They do not by
themselves prove that nuisance attributes remain unchanged in generated
images.

### Required baselines

Every optimization experiment compares the same semantic target, constraints,
and stopping rule across:

| Group | Metric | Purpose |
| --- | --- | --- |
| B1 | `I` | Euclidean baseline |
| B2 | `g(w_0)` | Fixed metric from the initial point |
| B3 | `g(w_k)` | Position-dependent sinh-induced metric |

B2 versus B3 separates static direction weighting from the effect of updating
the Jacobian along the path. Intensity and disentanglement comparisons must be
made at matched target progress `tau`.

### Experiment sequence

1. Validate the Legendre transform, sinh map, inverse, and Jacobian.
2. Run semantic direction optimization and measure position dependence.
3. Add target intensity constraints and matched-progress evaluation.
4. Add nuisance-direction constraints for disentanglement.
5. Decode all selected W-space paths with the frozen generator and evaluate
   image-level target progress and off-target change independently from the
   latent proxies.

The unifying endpoint formulation is

```text
minimize    E_g(w_0, w)
subject to  s_t(w) = tau
            s_j(w) belongs to C_j.
```

`E_g` is the selected geometric movement cost, `s_t` is the target semantic
measurement or proxy, and `C_j` defines the permitted change of each nuisance
measurement.

### Methodological references

- Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*,
  Sections 3.7–3.8 and 4.3.
- Nocedal, J., and Wright, S. J. (2006). *Numerical Optimization*, second
  edition, Chapters 12 and 16.
- Debnath, L., and Mikusinski, P. (2005). *Hilbert Spaces with Applications*,
  third edition, Chapters 3–4.
