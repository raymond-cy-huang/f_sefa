# GDSA Experimental Report — R004 / E01

## 1. Experiment identity and timestamp

- Round: R004; Experiment: E01
- Date: 2026-10-25 (Asia/Taipei)
- Track: Track A nonlinearity calibration
- Outcome: SUPPORTED

## 2. Research question

Does equal raw `sinh` alpha impose unequal effective nonlinearity across accepted
bases on actual shared-W samples, and does matched beta correct the bulk-scale
comparison?

## 3. Falsifiable hypothesis

**H-BETA-004:** basis coefficient Q99 values differ materially, so raw alpha is
not comparable; \(\alpha_B=\beta/Q_{.99}(|L_Bw|)\) aligns Q99 nonlinear
arguments and corresponding Q99 activation statistics.

## 4. Sources

R003 was read in full; the 100-file source manifest was unchanged; the full
local corpus was searched before implementation.

1. J. Nocedal and S. Wright, *Numerical Optimization*, 2nd ed., Springer,
   Sec. 6.1, p. 139 (PDF p. 159). Local source:
   `/home/rz/raymond_project/literature_review_papers/CV_Foundation/Numerical_Optimization_v2.pdf`.
   It motivates nondimensional norms so conclusions do not depend on arbitrary
   units.
2. T. Cormen et al., *Introduction to Algorithms*, 4th ed., MIT Press,
   Sec. 9.3, p. 242. Local source:
   `/home/rz/raymond_project/literature_review_papers/CV_Foundation/Introduction.to.Algorithms.4th.pdf`.
   It supplies the order-statistic/quantile concept used for robust calibration.

No web fallback was needed for this round.

## 5. Configuration

- 100 deterministic shared-W samples from e4e decoder mapping; seed 42
- Sample artifact SHA-256:
  `041b263802c7ab879cf1e68f5f7a2f7b848ea51ce1f575898dcbd01b130f8ef5`
- Accepted R003 Identity, Legendre, Chebyshev-I transforms
- Raw alpha: 0.1, 0.3, 0.5, 0.7, 1.0; 2.0 stress
- Matched beta: 0.1, 0.3, 0.5, 0.7, 1.0
- Two complete independent runs

## 6. Mathematical definitions

For basis \(B\), \(c_B=L_Bw\) and

\[
q_B=Q_{.99}(|c_B|),\qquad \beta=\alpha q_B,
\qquad \alpha_B(\beta)=\beta/q_B.
\]

The nonlinear map, mapping ratio, and derivative are

\[
h=\sinh(\alpha c)/\alpha,\qquad
r(c)=h/c,\qquad \partial h/\partial c=\cosh(\alpha c).
\]

## 7. Objective and constraints

This round calibrates scale only. It does not optimize a path or semantic
objective. The sample pool, transform definitions, and quantile estimator are
paired across bases.

## 8. Baselines and controls

- Equal raw alpha comparison.
- Equal target beta comparison.
- Identity basis control.
- Complete duplicate-run byte comparison.
- Non-finite counts and per-sample metric-condition tail checks.

## 9. Numerical results

Coordinate distributions:

| Basis | mean | SD | abs median | abs Q99 | abs max |
|---|---:|---:|---:|---:|---:|
| Identity | 0.0951 | 0.4636 | 0.1844 | 1.7725 | 4.0871 |
| Legendre | 0.0102 | 0.4732 | 0.3028 | 1.2878 | 4.5298 |
| Chebyshev-I | 0.0061 | 0.4732 | 0.3033 | 1.2954 | 4.1658 |

Q99 relative spread was `0.3763`, above the frozen 5% materiality threshold.

At raw alpha=1:

| Basis | empirical beta | Q99 map ratio | Q99 derivative | median sample condition | max sample condition |
|---|---:|---:|---:|---:|---:|
| Identity | 1.7725 | 1.6123 | 3.0276 | 26.90 | 887.47 |
| Legendre | 1.2878 | 1.3003 | 1.9504 | 14.06 | 2150.72 |
| Chebyshev-I | 1.2954 | 1.3041 | 1.9632 | 11.87 | 1038.74 |

At matched beta=1:

| Basis | calibrated alpha | Q99 argument | Q99 map ratio | Q99 derivative | median condition | max condition |
|---|---:|---:|---:|---:|---:|---:|
| Identity | 0.5642 | 1.0 | 1.175201 | 1.543081 | 3.98 | 25.67 |
| Legendre | 0.7765 | 1.0 | 1.175201 | 1.543081 | 6.06 | 284.39 |
| Chebyshev-I | 0.7719 | 1.0 | 1.175201 | 1.543081 | 5.27 | 155.77 |

Maximum matched-beta error was `1.11e-16`; no non-finite values occurred. All
four duplicate artifacts were byte-identical.

## 10. Image and proxy results

Not applicable. No images or semantic proxies were generated.

## 11. Statistical assessment

Quantiles pool 51,200 coordinates from 100 paired W samples. They calibrate the
empirical sample distribution and are not treated as independent semantic
observations. Duplicate runs verify computational reproducibility.

## 12. Observations

1. Raw alpha substantially overdrives Identity relative to the two mixed bases
   at the Q99 scale.
2. Legendre and Chebyshev-I bulk distributions are similar but not identical.
3. Matched beta exactly aligns Q99 argument, map ratio, and derivative.
4. Tail maxima remain basis dependent; Legendre has the largest observed tail
   condition despite matched beta.
5. Alpha=2 remains only a stress condition and must not be used as the initial
   semantic setting.

## 13. Interpretation

H-BETA-004 is **SUPPORTED**. Equal raw alpha is not a controlled basis
comparison. Matched beta is adopted for bulk-scale comparisons, but does not
equalize tails or full condition-number distributions. Both matched-beta and
matched-conditioning views remain necessary.

This is numerical calibration, not evidence that any basis or `sinh` improves
semantic traversal.

## 14. Limitations and failures

- Q99 is a design choice; other robust scales could change calibrated alpha.
- The W sample pool is specific to the current decoder mapping and seed policy.
- Coordinate pooling hides dependence among dimensions and samples.
- Maximum condition is sensitive to rare coordinates; it is intentionally
  retained as a safety measure rather than optimized away.
- Miniconda and trusted-checkpoint warnings from prior rounds persist.

## 15. Decision and next recommendation

Adopt the R004 paired W artifact and per-basis calibrated alpha table for Track
A. R005 should execute full nonlinear forward/inverse/Jacobian/metric validation
for all three bases at matched beta, plus alpha=2 stress, with float32/64 and
condition distributions. A later round should derive matched-conditioning
settings from those results.

## 16. Reproduction commands and artifacts

```bash
cd experiments_gdsa/rounds/round_R004_beta_calibration
conda run -n gdsa python run.py --output-dir outputs_a
conda run -n gdsa python run.py --output-dir outputs_b
```

Artifacts: `run.py`, two identical output sets containing
`shared_w_samples.npy`, `coordinate_distributions.csv`,
`raw_alpha_metrics.csv`, `matched_beta_metrics.csv`, and `summary.json`, plus
`tracing_log.md`.

