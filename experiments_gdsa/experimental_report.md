# GDSA Cumulative Experimental Report

## Program status

- Planned rounds: 100 (`R001`-`R100`)
- Current round: R005 closed
- Completed rounds: 5 closed
- Current phase: stopped after requested SRO experiment report; no automatic R006–R100 execution
- Evidence status: Experiment 01 is reproducible numerical pilot evidence;
  Experiment 02 is reproducible solver/proxy evidence requiring deterministic
  execution and standardized energy.

## Cumulative conclusions

R001 reproduced both legacy pilots. Experiment 01 matched its historical CSV
exactly. Experiment 02 reproduced target constraints and the Pymanopt check, but
generator-response fields varied slightly under default CUDA execution. Repeated
identical decoding became bitwise stable when PyTorch deterministic algorithms
were enabled. No image-level semantic conclusion is supported; Experiment 02
remains proxy-only. Before new semantic experiments, R002 must standardize
determinism and midpoint path energy.

R002 established the shared measurement layer. Deterministic generator settings
made two complete proxy runs byte-for-byte identical. The midpoint path-energy
implementation passed analytic constant and nonlinear scalar controls. Across
the tested mapping paths, the largest mean 20→50-step change was `1.08e-6`.
This validates measurement reproducibility, not semantic superiority.

R003 independently constructed Identity, normalized Legendre, and normalized
first-kind Chebyshev transforms. All passed shared float64/float32 orthogonality
and round-trip gates. In the linear normalized setting they are Euclidean
isometries; no basis-dependent semantic claim is possible at alpha zero.

R004 showed that equal raw alpha is not a controlled comparison on actual
shared-W samples: coefficient Q99 differs by 37.6% across bases. Matched beta
aligns Q99 argument and derivative statistics exactly, but not tail condition
numbers. Future basis comparisons must report both bulk and tail stability.

R005 established the Semantic Response Operator experiment and its negative
control. Across 15 exact 2D/16D cases, full orthogonal-Hilbert solutions mapped
back to W-space with maximum relative error `1.21e-15`; this is coordinate
invariance, not a Hilbert advantage. Diagonal approximation was basis dependent,
but the StyleGAN derivative smoke failed the frozen 5% finite-difference gate
(`9.51%` best error). The N=20 semantic pilot was therefore not run. Operator,
basis, dynamic-geometry, and novelty claims remain inconclusive.

## Round index

| Round | Track | Question | Outcome | Immutable report |
|---:|---|---|---|---|
| R001 | Foundation | Are the repository, source corpus, and existing pilot claims reproducible and internally consistent? | ACCEPT WITH REQUIRED REVISIONS | `rounds/round_R001_audit/experimental_report_20261025_R01_E01.md` |
| R002 | Measurement | Can deterministic decoding and standardized midpoint energy provide repeatable, refinement-stable proxy measurements? | SUPPORTED | `rounds/round_R002_measurement_standardization/experimental_report_20261025_R02_E01.md` |
| R003 | Track A | Are Identity, normalized Legendre, and normalized Chebyshev discrete transforms numerically valid under their declared measures? | SUPPORTED | `rounds/round_R003_basis_validation/experimental_report_20261025_R03_E01.md` |
| R004 | Track A | Does matched beta correct unequal effective nonlinearity across bases on actual shared-W samples? | SUPPORTED | `rounds/round_R004_beta_calibration/experimental_report_20261025_R04_E01.md` |
| R005 | SRO | Can a generator-response operator and Hilbert basis yield value beyond equivalent W-space methods? | INCONCLUSIVE; exact invariance PASS, generator gate FAIL | `rounds/round_R005_semantic_response_operator/experimental_report_20261025_R05_E01.md` |
