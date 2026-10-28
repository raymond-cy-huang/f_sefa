# GDSA Experimental Report — R003 / E01

## 1. Experiment identity and timestamp

- Round: R003
- Experiment: E01
- Date: 2026-10-25 (Asia/Taipei)
- Track: Track A representation feasibility
- Outcome: SUPPORTED

## 2. Research question

Can Identity, normalized Legendre, and normalized first-kind Chebyshev transforms
all be constructed as numerically valid finite-dimensional orthogonal maps under
their explicitly declared measures?

## 3. Falsifiable hypothesis

**H-BASIS-003:** all three transforms meet common orthogonality and inversion
tolerances when their nodes, weights, and polynomial norms are handled
correctly. Failure of any stated tolerance rejects that implementation.

## 4. Sources

The R002 report was read in full and the complete 100-file source manifest was
regenerated unchanged. The complete local text index was searched.

1. L. Debnath and P. Mikusiński, *Hilbert Spaces with Applications*, Elsevier,
   2005, p. 126, Eqs. (3.14) onward. Local source:
   `/home/rz/raymond_project/literature_review_papers/Hilbert Space/2005_Hilbert_Spaces_with_Applications.pdf`.
   It gives Legendre orthogonality on `[-1,1]` under unit weight and the norm
   required for orthonormalization.
2. NIST DLMF §18.3, Table 18.3.1, accessed 2026-10-25,
   <https://dlmf.nist.gov/18.3>. It specifies the first-kind Chebyshev measure
   \((1-x^2)^{-1/2}\), norm \(\pi\) for degree zero and \(\pi/2\) otherwise.
3. NIST DLMF §3.5.22–23, accessed 2026-10-25,
   <https://dlmf.nist.gov/3.5>. It supplies first-kind Gauss–Chebyshev nodes and
   equal weights.
4. SciPy `roots_chebyt` documentation, accessed 2026-10-25,
   <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.roots_chebyt.html>.
   It was used as an independent implementation check, not as the definition.

The local corpus lacked a complete Gauss–Chebyshev formula, so web fallback was
required and recorded before implementation.

## 5. Configuration

- Environment: `gdsa`; NumPy 1.26.4; SciPy 1.15.3
- Dimension: 512
- Samples: 100 paired standard-normal vectors
- Seed: 42
- Precisions: float64 and float32
- Common float64 orthogonality tolerance: `1e-10`
- Round-trip relative tolerance: `1e-10` float64, `1e-5` float32

## 6. Mathematical definitions

Legendre uses Gauss–Legendre nodes \(x_i\), weights \(q_i\), and
\(\|P_k\|^2=2/(2k+1)\):

\[
L^{Leg}_{ik}=\sqrt{q_i}P_k(x_i)/\sqrt{2/(2k+1)}.
\]

Chebyshev uses

\[
x_i=\cos\frac{(2i-1)\pi}{2n},\quad q_i=\frac\pi n,
\]

and \(\|T_0\|^2=\pi\), \(\|T_k\|^2=\pi/2\) for \(k>0\):

\[
L^{Cheb}_{ik}=\sqrt{q_i}T_k(x_i)/\|T_k\|.
\]

Identity uses the canonical Euclidean coordinate basis. The transpose is used as
an inverse only after measuring \(L^TL\approx I\).

## 7. Objective and constraints

This is a representation-only numerical validation. No semantic objective and
no generator image evaluation are involved.

## 8. Baselines and controls

- Identity exact control.
- Closed-form Chebyshev nodes/weights versus NumPy `chebgauss`.
- SciPy `roots_chebyt` as a second independent Chebyshev implementation.
- Small 16/32-dimensional unit tests before the 512-dimensional run.
- Identical sample pool across bases and dtypes.

## 9. Numerical results

| Basis | dtype | orthogonality max error | condition | max relative round-trip | max absolute round-trip |
|---|---|---:|---:|---:|---:|
| Identity | float64 | 0 | 1 | 0 | 0 |
| Identity | float32 | 0 | 1 | 0 | 0 |
| Legendre | float64 | `1.528e-12` | `1.0000000001` | `2.426e-11` | `5.009e-11` |
| Legendre | float32 | `1.528e-12` | `1.0000000001` | `4.611e-7` | `1.788e-6` |
| Chebyshev | float64 | `4.627e-14` | `1.0000000000` | `5.787e-13` | `1.364e-12` |
| Chebyshev | float32 | `4.627e-14` | `1.0000000000` | `4.187e-7` | `1.907e-6` |

Chebyshev independent-definition errors:

- closed-form node maximum error: `0`;
- closed-form weight maximum error: `0`;
- SciPy node maximum error: `2.220e-16`;
- SciPy weight maximum error: `0`.

All tests and acceptance criteria passed.

## 10. Image and proxy results

Not applicable. No images or semantic proxies were generated.

## 11. Statistical assessment

The 100 samples are paired numerical probes. Maximum and mean reconstruction
errors were evaluated directly; no population semantic inference is made.

## 12. Observations

1. All three finite transforms are numerically orthogonal and invertible.
2. Legendre has larger construction error than Chebyshev at dimension 512, but
   both remain inside the frozen tolerance.
3. Float32 reconstruction error is around `4e-7` for both polynomial bases.
4. The bases use different continuous measures; this distinction is now encoded
   explicitly rather than hidden behind a common “orthogonal basis” label.
5. Each normalized linear transform preserves Euclidean quadratic forms and
   angles up to numerical error.

## 13. Interpretation

H-BASIS-003 is **SUPPORTED**. Identity, normalized Legendre, and normalized
first-kind Chebyshev are valid Track A representations.

Because all three are orthogonal finite matrices, the alpha-zero pullback metric
is \(L^TL=I\). Thus a linear comparison among them cannot produce an intrinsic
geometric or semantic difference. Any later difference must come from the
coordinate distribution interacting with a nonlinear map, numerical error, or
an incorrectly transformed metric—not from “orthogonal basis” alone.

## 14. Limitations and failures

- Random Gaussian probes are not the actual generator W distribution.
- Runtime measurements include BLAS warm-up and are descriptive only.
- The transform matrices are dense and consume about 2 MiB in float64 or 1 MiB
  in float32 each; scaling to higher dimensions should be reassessed.
- No nonlinear mapping, Jacobian, or semantic evaluation was performed.

## 15. Decision and next recommendation

Decision: **ACCEPT** all three transforms for Track A.

R004 should measure coefficient distributions on paired actual shared-W samples
and compute per-basis alpha values for matched dimensionless nonlinearity
\(\beta=\alpha Q_{0.99}(|c|)\). This must precede any raw-alpha basis comparison.

## 16. Reproduction commands and artifacts

```bash
cd experiments_gdsa/rounds/round_R003_basis_validation
conda run -n gdsa python -m unittest -v test_transforms.py
conda run -n gdsa python run.py
```

Artifacts: `run.py`, `test_transforms.py`, `outputs/basis_metrics.csv`,
`outputs/summary.json`, and `tracing_log.md`.

