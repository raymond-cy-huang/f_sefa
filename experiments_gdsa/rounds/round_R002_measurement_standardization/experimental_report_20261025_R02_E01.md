# GDSA Experimental Report — R002 / E01

## 1. Experiment identity and timestamp

- Round: R002
- Experiment: E01
- Date: 2026-10-25 (Asia/Taipei)
- Track: measurement standardization
- Status: COMPLETED
- Outcome: SUPPORTED

## 2. Research question

Can deterministic generator execution and a midpoint discretization provide
repeatable, refinement-stable path-energy and generator-response proxy
measurements without breaking the target-progress constraint?

## 3. Falsifiable hypothesis

**H-MEASURE-002:** deterministic decoding plus midpoint path energy produces
repeatable, step-refinement-stable proxy measurements without changing target
constraint satisfaction. The six acceptance criteria were frozen in the R002
tracing log before implementation.

## 4. Sources

Before R002, the complete 100-file local source manifest was regenerated and
matched R001. The preceding R001 report was read in full.

Sources used:

1. N. Boumal, *An Introduction to Optimization on Smooth Manifolds*, Cambridge
   University Press, 2023, Sec. 10.1, Eq. (10.1). It defines Riemannian curve
   length as an integral of the pointwise speed norm. Local source:
   `/home/rz/raymond_project/literature_review_papers/Hilbert Space/5_heros/IntroOptimManifolds_Boumal_2022.pdf`.
2. NIST Digital Library of Mathematical Functions, §3.5(iv), “Interpolatory
   Quadrature Rules,” accessed 2026-10-25,
   <https://dlmf.nist.gov/3.5>. It identifies the midpoint rule as an open
   quadrature rule and supplies the general quadrature framework.
3. OpenStax, *Calculus Volume 2*, §3.6, “Numerical Integration,” Theorem 3.5,
   accessed 2026-10-25,
   <https://openstax.org/books/calculus-volume-2/pages/3-6-numerical-integration>.
   It gives the composite midpoint error bound proportional to \(n^{-2}\) for
   an integrand with bounded second derivative.
4. R001 immutable report and tracing log. These supplied the observed default
   CUDA nondeterminism and the legacy energy definition requiring correction.

The local corpus did not contain a sufficiently explicit midpoint error bound,
so the required web fallback was used. The selected external sources are an
authoritative numerical reference and a textbook.

## 5. Configuration

- Repository revision: `22cca23ed7001c77a033b4a3d1978041079d677e`
- Environment: `gdsa`
- Python 3.10.18; NumPy 1.26.4; PyTorch 2.5.1+cu121
- GPU: NVIDIA GeForce RTX 3070
- Samples: 2 paired shared-W codes; seed 42
- Alpha: 0, 0.5, 1.0
- Step counts: 1, 2, 5, 10, 20, 50
- Target progress: 0.5
- Proxy step count: 5
- Generator finite-difference epsilon: 0.01
- Proxy image resolution: 64 × 64
- Determinism: CUBLAS workspace fixed, cuDNN benchmark disabled, cuDNN
  deterministic enabled, PyTorch deterministic algorithms required.

## 6. Mathematical definitions

For path parameter \(t\), the standardized energy is

\[
E=\frac12\int_0^\tau \dot w(t)^TM(w(t))\dot w(t)\,dt.
\]

With \(\Delta t_k=t_{k+1}-t_k\), \(\Delta w_k=w_{k+1}-w_k\), and
\(w_{mid,k}=(w_k+w_{k+1})/2\), R002 uses

\[
E_{mid}=\frac12\sum_k
\frac{\Delta w_k^TM(w_{mid,k})\Delta w_k}{\Delta t_k}.
\]

The historical quantity is retained under the explicit name

\[
E_{legacy,left}=\sum_k \Delta w_k^TM(w_k)\Delta w_k.
\]

These quantities differ in both scaling and quadrature point and are not
interchangeable.

## 7. Objective and constraints

The direction at each step still satisfies \(d^Tv=1\), so target progress is
the path parameter. R002 changes measurement and deterministic execution, not
the B1/B2/B3 optimization objectives.

## 8. Baselines and controls

- Analytic constant metric: \(M=I\), \(\tau=0.5\), expected \(E=0.25\).
- Historical five-step energy: expected `0.05` under the legacy definition.
- Scalar nonlinear metric with closed-form integral.
- Two independent complete executions (`outputs_run_a`, `outputs_run_b`).
- Step-refinement sequence 1→2→5→10→20→50.

## 9. Numerical results

All three standard-library unit tests passed:

- constant metric analytic energy and target progress;
- explicit preservation/naming of legacy energy;
- monotonically decreasing scalar nonlinear midpoint error.

Both complete runs passed:

- repeated identical decode maximum difference: `0.0`;
- maximum target-progress error: `1.9984014443252818e-15`;
- maximum alpha-zero analytic energy error: `6.047662370889384e-13`;
- refinement CSVs: byte-for-byte equal;
- proxy CSVs: byte-for-byte equal.

Mean standardized energy over two seeds at 50 steps:

| alpha | B1 | B2 | B3 |
|---:|---:|---:|---:|
| 0.0 | 0.2500000000 | 0.2500000000 | 0.2500000000 |
| 0.5 | 0.2705692987 | 0.2650626552 | 0.2650558666 |
| 1.0 | 0.3983039679 | 0.2993631549 | 0.2992907911 |

Absolute difference between 20- and 50-step mean energy:

| alpha | B1 | B2 | B3 |
|---:|---:|---:|---:|
| 0.0 | `8.33e-17` | `1.67e-16` | `1.67e-16` |
| 0.5 | `1.90e-8` | `1.77e-8` | `9.26e-8` |
| 1.0 | `1.79e-7` | `9.38e-8` | `1.08e-6` |

The nonlinear dynamic path is the slowest of the tested cases to stabilize, but
its 20→50 difference is still approximately one part in \(2.8\times10^5\).

## 10. Image and proxy results

No new semantic image claim was tested. The deterministic proxy CSV confirms
that low-resolution finite-difference response values can now be reproduced
exactly on this hardware/software configuration. Semantic polarity remains
unvalidated, so all generator-response results remain proxy-only.

## 11. Statistical assessment

This round validates a measurement implementation rather than a population
effect. Two seeds are adequate for paired code-path checks but not semantic
inference. Refinement error was assessed against analytic controls and successive
discretizations, not by treating path steps as independent samples.

## 12. Observations

1. Deterministic PyTorch settings eliminate the R001 repeated-decode variance.
2. Geometry and GPU proxy outputs are exactly reproducible across two runs.
3. The midpoint implementation agrees with the analytic constant metric.
4. Refinement differences decrease to small values for all tested cases.
5. B2 and B3 remain extremely close; R002 does not interpret this as semantic
   equivalence because no evaluator was used.
6. Standard energy is five times the legacy value in the alpha-zero five-step
   example, exactly as implied by the two definitions.

## 13. Interpretation

H-MEASURE-002 is **SUPPORTED**. The measurement foundation is now adequate for
future comparisons on this environment. The result supports deterministic and
standardized measurement only; it does not support `sinh`, B2, B3, or a semantic
benefit.

## 14. Limitations and failures

- `pytest` was unavailable; no package was added. Tests were converted to
  standard-library `unittest` and passed.
- Miniconda login continues to emit unrelated broken-symlink warnings.
- Deterministic results are validated on one RTX 3070 stack; other hardware may
  require a new determinism check.
- `warn_only=False` may reject future unsupported deterministic CUDA operations;
  such rejection should be treated as an explicit failure, not disabled silently.
- Only the mapping metric was measured; no generator metric B4 was implemented.

## 15. Decision and next recommendation

Decision: **ACCEPT** the deterministic and midpoint-energy measurement layer.

R003 should start Track A representation construction with the smallest
discriminating task: independently construct and validate Identity, normalized
Legendre, and normalized Chebyshev transforms, documenting discrete nodes,
weights, orthogonality, inverse error, and conditioning before applying `sinh`.

## 16. Reproduction commands and artifacts

```bash
cd experiments_gdsa/rounds/round_R002_measurement_standardization
conda run -n gdsa python -m unittest -v test_energy.py
conda run -n gdsa python run.py --output-dir outputs_run_a
conda run -n gdsa python run.py --output-dir outputs_run_b
cmp outputs_run_a/refinement_metrics.csv outputs_run_b/refinement_metrics.csv
cmp outputs_run_a/proxy_metrics.csv outputs_run_b/proxy_metrics.csv
```

Artifacts:

- `energy.py`
- `test_energy.py`
- `run.py`
- `outputs_run_a/{summary.json,refinement_metrics.csv,proxy_metrics.csv}`
- `outputs_run_b/{summary.json,refinement_metrics.csv,proxy_metrics.csv}`
- `tracing_log.md`

