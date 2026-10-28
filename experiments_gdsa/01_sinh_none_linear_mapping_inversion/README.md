# Experiment 01: Sinh Nonlinear Mapping and Inversion

## Objective

Validate the numerical foundation required by later GDSA optimization:

```text
c = Lw
h = sinh(alpha * c) / alpha
w_reconstructed = L.T @ asinh(alpha * h) / alpha
```

`L` is constructed from normalized Legendre polynomials evaluated at
Gauss–Legendre nodes with quadrature weights. The `alpha = 0` implementation
uses the exact linear limit.

This experiment does not make semantic or image-quality claims. It verifies
the coordinate map, analytical inverse, differential, and induced metric.

## Checks

- `L.T @ L` is numerically equal to the identity.
- Forward/inverse round-trip reconstructs random W-space samples.
- The analytical Jacobian agrees with `scipy.differentiate.jacobian` on a
  reduced-dimensional diagnostic.
- Forward and inverse JVPs compose to the identity.
- `g(w) = J_T(w).T @ J_T(w)` is symmetric positive definite.
- Float32 and float64 behavior is recorded over an alpha sweep.

## Run

From the repository root:

```bash
conda run -n gdsa python \
  experiments_gdsa/01_sinh_none_linear_mapping_inversion/run.py
```

Results are written under this experiment's ignored `outputs/` directory.

## Acceptance criteria

The default run passes when:

- float64 Legendre orthogonality error is at most `1e-10`;
- all forward values remain finite;
- float64 maximum relative latent round-trip error is at most `1e-10`;
- float32 maximum relative latent round-trip error is at most `1e-5`;
- analytical/numerical Jacobian maximum absolute error is at most `1e-7`;
- the minimum sampled metric eigenvalue is positive.

## Recorded baseline run

The default experiment completed successfully with NumPy 1.26.4 and SciPy
1.15.3.

```text
status: PASS
dimension: 512
samples: 100
seed: 42
alphas: 0.0, 0.1, 0.5, 1.0, 2.0
```

Legendre transform diagnostics:

| Metric | Value |
| --- | ---: |
| Maximum orthogonality error | `1.528e-12` |
| Condition number | `1.0000000001` |

Mapping and metric diagnostics:

| dtype | alpha | Maximum relative round-trip error | Metric condition number |
| --- | ---: | ---: | ---: |
| float64 | 0.0 | `1.988e-11` | `1.000e+00` |
| float64 | 0.1 | `1.988e-11` | `1.071e+00` |
| float64 | 0.5 | `1.988e-11` | `4.003e+00` |
| float64 | 1.0 | `1.988e-11` | `4.908e+01` |
| float64 | 2.0 | `1.988e-11` | `9.439e+03` |
| float32 | 0.0 | `6.276e-07` | `1.000e+00` |
| float32 | 0.1 | `6.362e-07` | `1.071e+00` |
| float32 | 0.5 | `6.289e-07` | `4.003e+00` |
| float32 | 1.0 | `6.247e-07` | `4.908e+01` |
| float32 | 2.0 | `6.254e-07` | `9.439e+03` |

At `alpha = 2.0`, the reduced eight-dimensional analytical Jacobian agreed
with SciPy's numerical Jacobian with maximum absolute error `1.403e-13` and
relative Frobenius error `9.397e-15`.

The first result establishes numerical invertibility over the tested random
sample distribution. It also shows that increasing `alpha` rapidly worsens the
local metric condition number. This is a numerical and coordinate-geometric
finding; it is not yet evidence of improved semantic traversal.
