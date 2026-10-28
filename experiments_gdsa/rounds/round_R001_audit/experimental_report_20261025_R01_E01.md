# GDSA Experimental Report — R001 / E01

## 1. Experiment identity and timestamp

- Round: R001
- Experiment: E01
- Date: 2026-10-25 (Asia/Taipei)
- Track: foundation audit
- Status: COMPLETED
- Outcome: ACCEPT WITH REQUIRED REVISIONS

## 2. Research question

Are the authoritative local source corpus, the repository definitions, and the
reported results of existing GDSA Experiments 01 and 02 reproducible and
internally consistent enough to serve as Version 2.0 pilot evidence?

## 3. Falsifiable hypothesis

**H-AUDIT-001:** clean reruns of Experiments 01 and 02 in the documented `gdsa`
environment reproduce their acceptance metrics within declared numerical
tolerances, and code inspection confirms that the reported quantities match
their implementations.

The hypothesis is rejected if a clean rerun fails, differs beyond numerical
tolerance, or computes a materially different object from the report.

## 4. Sources inspected

The complete local corpus was inventoried before the experiment: 100 non-`.git`
files, 903,726,158 bytes. It contains 95 `.pdf` paths, 2 DOCX, 2 HTML, and 1
Markdown file. Five `.pdf` paths in `Springer_Format_Reference` are actually HTML
documents and are not treated as papers. All 100 corpus files plus the required
repository analysis were successfully converted to a searchable cache.

Primary sources used in this round:

1. Y. Shen and B. Zhou, “Closed-Form Factorization of Latent Semantics in
   GANs,” CVPR 2021, pp. 1532–1540, especially Sec. 2.1–2.2 and Eqs. (4)–(7).
   Local source:
   `/home/rz/raymond_project/literature_review_papers/Latent_Space_Manipulation/2021_SeFa_Close-Form Factorization_Of_Semantics_In_GANs.pdf`.
   The paper establishes that SeFa directions are eigenvectors of \(A^TA\) for
   the selected generator affine weights; it does not establish the semantic
   polarity of a repository-specific rank annotation.
2. Y. Shen, J. Gu, X. Tang, and B. Zhou, “Interpreting the Latent Space of GANs
   for Semantic Face Editing,” CVPR 2020, pp. 9243–9252, especially Sec. 2.2 and
   Sec. 3.3. Local source:
   `/home/rz/raymond_project/literature_review_papers/Latent_Space_Entanglement/2020_Interpreting the Latent Space of GANs for Semantic Face Editing.pdf`.
   It supports the linear-boundary and conditional-projection baseline, but its
   assumptions and evaluation do not turn a SeFa rank into an independent
   semantic evaluator.
3. N. Boumal, *An Introduction to Optimization on Smooth Manifolds*, Cambridge
   University Press, 2023 pre-publication version, especially Sec. 3.7–3.8 and
   Sec. 4.8. Local source:
   `/home/rz/raymond_project/literature_review_papers/Hilbert Space/5_heros/IntroOptimManifolds_Boumal_2022.pdf`.
   It supports the distinction between a declared Riemannian metric, its
   gradient, and numerical derivative checking.
4. `docs/stage02_literature_deep_analysis_2026-09-28.md`, read in full. This is
   a repository analysis note rather than a primary source. It independently
   warns that representation, geometry, semantic constraint, and finite-edit
   validity must be separated.

No web retrieval was required because the local corpus contained the primary
sources needed for this audit.

## 5. Configuration

- Repository revision: `22cca23ed7001c77a033b4a3d1978041079d677e`
- Working tree before/after audit: pre-existing modified `README.md`; the
  `experiments_gdsa` tree is untracked in Git.
- Environment: `gdsa`
- Python: 3.10.18
- NumPy: 1.26.4
- SciPy: 1.15.3
- PyTorch: 2.5.1+cu121
- Pymanopt: 2.2.1
- GPU: NVIDIA GeForce RTX 3070
- e4e checkpoint SHA-256:
  `2ace1d9a8c05c10a399bcd500b8dda118f759ff1aac89dbdab7435f2136a0999`

The login shell emitted pre-existing Miniconda broken-symlink warnings. They did
not change exit codes or experiment outputs, but remain an environment hygiene
issue.

## 6. Mathematical definitions audited

Experiment 01 implements

\[
L_{ik}=\sqrt{q_i}\,P_k(x_i)/\sqrt{2/(2k+1)},
\quad h=\sinh(\alpha Lw)/\alpha,
\quad w'=L^T\operatorname{asinh}(\alpha h)/\alpha,
\]

with the linear limit at \(\alpha=0\). For the Gauss–Legendre construction,
the tested finite matrix is numerically orthogonal, so the transpose inverse is
valid to the measured tolerance.

Experiment 02 uses the matched-progress direction

\[
v=M^{-1}d/(d^TM^{-1}d)
\]

for mapping-induced \(M=g_s(w)\), ensuring \(d^Tv=1\). It also estimates the
generator directional response by centered finite differences.

## 7. Objectives and constraints

This audit did not optimize a new semantic objective. It checked:

- mapping/inverse consistency;
- analytic versus numerical Jacobian agreement;
- positivity of the sampled mapping metric;
- satisfaction of the target-progress constraint;
- agreement of the analytic direction with a Pymanopt sphere formulation;
- reproducibility of generator-response proxy metrics;
- consistency between code, README, and stored output.

## 8. Baselines and controls

- Historical output versus clean isolated rerun.
- `float64` versus `float32` in Experiment 01.
- \(\alpha=0\) linear limit.
- B1 Euclidean, B2 fixed mapping metric, B3 dynamic mapping metric.
- Default CUDA execution versus PyTorch deterministic algorithms for identical
  repeated decoding.

## 9. Numerical results

### Experiment 01 reproduction

- Status: PASS
- Dimension: 512
- Samples: 100
- Seed: 42
- Maximum Legendre orthogonality error: `1.5279413119539287e-12`
- Transform condition number: `1.0000000001146119`
- Reduced Jacobian maximum absolute error: `1.4033219031261979e-13`
- Reduced Jacobian relative Frobenius error: `9.397168855612884e-15`
- Historical and reproduction CSV: 10/10 rows, zero differing fields.

The historical float32/float64 round-trip and metric-conditioning values were
reproduced exactly.

### Experiment 02 reproduction

- Status: PASS under its original criteria.
- Maximum target-progress error: `2.220446049250313e-16`.
- Minimum Pymanopt/analytic absolute direction cosine: `1.0`.
- Pymanopt CSV: 3/3 rows, zero differing fields.
- Path CSV: 18/18 rows; geometry-only fields reproduced exactly.
- Generator-response fields differed slightly between runs:
  maximum absolute field difference `3.077983856201172e-4`.

Mean values over two seeds:

| alpha | method | stored path energy | latent displacement | end generator JVP RMS | direction-change cosine |
|---:|---|---:|---:|---:|---:|
| 0.0 | B1 | 0.0500000 | 0.5000000 | 0.1223930 | 1.0000000 |
| 0.0 | B2 | 0.0500000 | 0.5000000 | 0.1223808 | 1.0000000 |
| 0.0 | B3 | 0.0500000 | 0.5000000 | 0.1223969 | 1.0000000 |
| 0.5 | B1 | 0.0541154 | 0.5000000 | 0.1223863 | 1.0000000 |
| 0.5 | B2 | 0.0530065 | 0.5025273 | 0.1244016 | 1.0000000 |
| 0.5 | B3 | 0.0530057 | 0.5025249 | 0.1247112 | 0.9999847 |
| 1.0 | B1 | 0.0798051 | 0.5000000 | 0.1223965 | 1.0000000 |
| 1.0 | B2 | 0.0598460 | 0.5136831 | 0.1248099 | 1.0000000 |
| 1.0 | B3 | 0.0598366 | 0.5137219 | 0.1245717 | 0.9998594 |

These values reproduce the pilot observation that B2 and B3 are nearly
identical. The lower mapping-metric objective is not independent evidence of a
semantic improvement.

### Decode determinism audit

The adapter explicitly calls `randomize_noise=False`; StyleGAN noise injection
is therefore fixed. Nevertheless, two identical decodes in one process under
default CUDA execution were not bitwise equal:

- maximum absolute difference: `1.0165572166442871e-4`;
- mean absolute difference: `7.681548481741629e-7`.

After setting `torch.backends.cudnn.benchmark=False`,
`torch.backends.cudnn.deterministic=True`, and
`torch.use_deterministic_algorithms(True, warn_only=True)`, repeated decodes
were bitwise equal with zero maximum and mean difference.

## 10. Image and proxy results

The reproduced contact sheet was visually inspected. All rows generated valid
faces without gross corruption. At the small target displacement, visual
differences between B1/B2/B3 were subtle and do not support an independent
semantic conclusion.

Experiment 02 measures low-resolution pixel-space directional response. It has
no validated smile evaluator, target polarity check, identity metric, or
off-target attribute evaluator. Its image result remains **proxy-only**.

## 11. Statistical assessment

Experiment 01 has 100 paired random samples and directly tests numerical
invariants. Experiment 02 has only two seeds, so mean differences cannot support
population-level semantic inference and no meaningful 95% confidence interval
is claimed. Failure count was zero under the original script criteria.

## 12. Observations

1. Experiment 01 is exactly reproducible at CSV level.
2. Experiment 02's constraint and analytic/Pymanopt checks are reproducible.
3. B2 and B3 remain practically identical over the tested five-step path.
4. Default CUDA execution introduces small repeated-decode differences even
   with fixed StyleGAN noise; deterministic PyTorch settings remove them.
5. The Experiment 02 energy is accumulated as
   \(\sum_k\Delta t^2v_k^TM(w_k)v_k\), not the Version 2.0 standardized midpoint
   energy \(\frac12\sum_k\Delta w_k^TM(w_{mid,k})\Delta w_k/\Delta t_k\).
6. The SeFa rank annotation has `polarity_validated=false`.
7. Generator finite-difference JVP is an evaluation proxy, not a B4
   generator-metric optimizer.

## 13. Interpretation

H-AUDIT-001 is accepted for the limited claim that the historical pilots run
and their documented acceptance metrics correspond to their code. The evidence
does not promote Experiment 02 to semantic confirmation. Its generator-response
comparison requires deterministic execution and its energy requires a new,
explicit standard before cross-round comparison.

Experiment 01 is valid numerical pilot evidence for one Legendre construction;
it does not establish basis superiority. Experiment 02 is valid solver/proxy
pilot evidence; it does not establish a Riemannian or semantic advantage.

## 14. Limitations and failures

- The source extraction initially failed on five misleading `.pdf` HTML files;
  content-type detection fixed the failure and all sources were indexed.
- WSL lacks `rg`; the fallback `grep` was used.
- One initial non-login manifest call failed because `conda` was not on `PATH`;
  login-shell execution succeeded.
- The Miniconda login shell emits broken-symlink warnings.
- The PyTorch checkpoint is loaded through pickle with `weights_only=False`;
  it is a trusted local checkpoint but the warning must remain visible.
- No independent image semantic evaluator was available.
- Experiment 02's two-sample design is not inferential.
- This round did not render every PDF page; it indexed the full corpus and
  inspected complete relevant sections in the original extracted sources.

## 15. Decision and next recommendation

Decision: **ACCEPT WITH REQUIRED REVISIONS**.

R002 must be a measurement-standardization round before any new semantic claim:

1. enforce deterministic generator execution;
2. implement and unit-test the Version 2.0 midpoint path-energy definition;
3. preserve legacy energy under a distinct field name;
4. validate energy convergence under step refinement;
5. verify that the deterministic change removes repeat-decode variance; and
6. keep all conclusions proxy-only.

## 16. Reproduction commands and artifacts

```bash
conda run -n gdsa python \
  experiments_gdsa/01_sinh_none_linear_mapping_inversion/run.py \
  --output-dir experiments_gdsa/rounds/round_R001_audit/reproduction_exp01

conda run -n gdsa python \
  experiments_gdsa/02_generator_jacobian_ro/run.py \
  --output-dir experiments_gdsa/rounds/round_R001_audit/reproduction_exp02

conda run -n gdsa python \
  experiments_gdsa/rounds/round_R001_audit/validate_exp02_determinism.py
```

Artifacts:

- `reproduction_exp01/summary.json`
- `reproduction_exp01/roundtrip_metrics.csv`
- `reproduction_exp02/summary.json`
- `reproduction_exp02/path_metrics.csv`
- `reproduction_exp02/pymanopt_checks.csv`
- `reproduction_exp02/traversal_seed_000.png`
- `exp02_determinism.json`
- `tracing_log.md`

