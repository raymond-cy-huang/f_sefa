# GDSA Research Guide

## Version 2.0 — Dual-Track Experimental Framework

This document is the repository-specific operating guide for Geometry-Driven
Semantic Analysis (GDSA). It converts the Version 2.0 research strategy into a
falsifiable, auditable, and compute-bounded workflow for `f_sefa`.

The purpose of this guide is not to prove that `sinh`, an orthogonal basis, or
Riemannian optimization must work. The purpose is to determine which geometric
mechanism, if any, produces an independently measurable improvement in semantic
traversal.

> Current status: planning only. Existing Experiments 01 and 02 are prior pilot
> evidence. They must be audited under this guide before their results are used
> as confirmation evidence.

---

## 1. Research mission

The research process must:

1. inspect and understand the actual repository before changing it;
2. discover and cite relevant primary literature;
3. reproduce and audit prior results before extending them;
4. formulate falsifiable hypotheses with explicit success and failure criteria;
5. run controlled, paired experiments;
6. report positive, negative, inconclusive, and invalid outcomes;
7. preserve a cumulative evidence trail; and
8. select each next experiment for information value rather than exhaustively
   sweeping arbitrary combinations.

Prohibited practices:

- assuming that nonlinear coordinates or an orthogonal basis are beneficial;
- treating a lower value of the optimized objective as independent evidence;
- hiding failures or changing the evaluation criterion after observing results;
- claiming an experiment was executed when it was only planned;
- fabricating measurements, citations, or background execution;
- launching an unbounded Cartesian hyperparameter sweep.

---

## 2. Primary research question

Can nonlinear mapping-induced geometry, generator-aware differential
information, or their combination improve semantic traversal under explicit
objectives and constraints?

The question is divided into three research objectives:

- **RQ1 — Semantic direction:** Can geometry produce a lower-cost direction at
  matched target progress?
- **RQ2 — Semantic intensity:** Can geometry reach a requested target intensity
  more accurately or efficiently?
- **RQ3 — Semantic disentanglement:** Can geometry reduce off-target variation
  while preserving matched target progress?

### Scope restrictions

- Do not train or fine-tune the generator.
- Do not train a new semantic classifier inside the optimization loop.
- Use pretrained generators and documented checkpoints.
- Define every mapping and inverse explicitly.
- Separate latent-space proxy measurements from image-space evaluation.
- Cite sources for externally derived claims.
- Do not introduce kernel methods into the initial study.

---

## 3. Mathematical foundation

### 3.1 Effective generator domain

The current decoder consumes a repeated StyleGAN latent. Therefore the object
being differentiated is the effective shared-`W` generator

\[
F(w) = G(\operatorname{repeat}(w, 18)), \qquad w \in \mathbb{R}^{512}.
\]

Unless an experiment explicitly studies unrestricted `W+`, all Jacobians,
directions, constraints, and metrics must be defined on this 512-dimensional
shared-`W` domain. Differentiating an unconstrained `W+` tensor would answer a
different research question.

### 3.2 Functional coordinate transform

For basis \(B\), let \(L_B\) denote a numerically normalized transform:

\[
c = L_B w.
\]

Initial basis choices are:

- identity control;
- normalized Legendre transform;
- normalized Chebyshev transform.

Hermite, Laguerre, Fourier, and other bases are out of scope until the initial
basis comparison provides a reason to add them.

### 3.3 Nonlinear reparameterization

Define

\[
T_{B,\alpha}(w) =
\begin{cases}
L_Bw, & \alpha = 0,\\
\frac{1}{\alpha}\sinh(\alpha L_Bw), & \alpha > 0.
\end{cases}
\]

If and only if \(L_B\) is orthogonal under the implemented finite-dimensional
normalization, the inverse is

\[
D_{B,\alpha}(h) =
\begin{cases}
L_B^T h, & \alpha = 0,\\
L_B^T\frac{1}{\alpha}\operatorname{asinh}(\alpha h), & \alpha > 0.
\end{cases}
\]

Otherwise the inverse must use a validated linear solve or pseudoinverse. The
code must never substitute \(L_B^T\) merely because the continuous basis is
orthogonal in theory.

The mapping Jacobian and its pullback metric are

\[
J_T(w)=\frac{\partial T(w)}{\partial w}, \qquad
g_s(w)=J_T(w)^T J_T(w).
\]

`g_s` is a coordinate-induced, position-dependent pullback metric. Its
position dependence alone is not evidence of nonzero intrinsic curvature.

### 3.4 Generator-aware metric

For the effective generator,

\[
J_F(w)=\frac{\partial F(w)}{\partial w}, \qquad
g_G(w)=J_F(w)^T J_F(w).
\]

The full image Jacobian must not be materialized unless memory estimates first
show that doing so is safe. The preferred implementation is a matrix-free
metric-vector product:

\[
g_G(w)v = J_F(w)^T\bigl(J_F(w)v\bigr),
\]

computed with validated automatic-differentiation JVP/VJP operations. Finite
differences may be used for spot validation, but not silently substituted for
the intended optimizer metric.

### 3.5 Combined metric

When evidence justifies integration, use

\[
M(w)=\lambda_s\,\widetilde g_s(w)
    +\lambda_G\,\widetilde g_G(w)
    +\mu I.
\]

The two metric components must be normalized before applying weights. A
normalization rule—such as trace per dimension or median directional quadratic
form—must be selected using discovery data, recorded, and then frozen. Raw
matrix scale must not determine the scientific conclusion.

---

## 4. Dual-track evidence model

### Track A — Representation feasibility

Track A asks whether a basis and nonlinear coordinate map are numerically valid
and meaningfully distinct. It does not make semantic claims.

Initial variants:

| ID | Basis | Mapping |
|---|---|---|
| A1 | Identity | Linear |
| A2 | Legendre | Linear |
| A3 | Chebyshev | Linear |
| A4 | Identity | `sinh` |
| A5 | Legendre | `sinh` |
| A6 | Chebyshev | `sinh` |

Every basis implementation must document:

- discrete domain and nodes;
- quadrature or weighting measure;
- finite-dimensional normalization;
- orthogonality error;
- invertibility and condition number;
- runtime and memory cost.

Legendre and Chebyshev polynomials are orthogonal under different measures.
Their transforms must be constructed under their correct discrete weights, then
compared using a clearly stated common coefficient-space convention. The old
prototype transform must not be assumed to satisfy this requirement.

Candidate nonlinearities are \(\alpha\in\{0,0.1,0.3,0.5,0.7,1.0\}\), with
\(\alpha=2.0\) used only as a stress test. Before selecting an alpha, record the
coordinate distribution and define the dimensionless nonlinearity

\[
\beta = \alpha\,Q_{0.99}(|c|).
\]

Basis comparisons must include matched-\(\beta\) and, where possible,
matched-conditioning comparisons. Equal raw alpha is not necessarily equal
nonlinearity across bases.

Track A measurements:

- transform orthogonality error;
- forward/inverse round-trip error;
- maximum absolute reconstruction error;
- analytic versus finite-difference Jacobian error;
- metric condition number and singular/eigenvalue distribution;
- transformed direction cosine preservation or distortion;
- static versus dynamic metric deviation along controlled paths;
- numerical failures, overflow, and non-finite counts;
- runtime and peak memory;
- paired `float32` and `float64` results on the same sample pool.

Track A gate: a representation may enter semantic testing only when it is
numerically stable in the precision used by the generator, invertible to a
declared tolerance, and distinguishable from a simpler control under a declared
geometric statistic. Passing this gate does not imply semantic usefulness.

### Track B — Semantic optimization feasibility

Track B asks whether an explicit geometry improves a semantic objective. The
first Track B study uses one validated Legendre+`sinh` configuration selected
for numerical reasons, not because it is presumed best.

Candidate semantic pairs include:

- Smile / Age;
- Smile / Pose;
- Age / Gender;
- Pose / Eyeglasses.

Pair selection uses a discovery subset and is frozen before held-out testing.
Selection evidence may include primary literature, direction reproducibility,
latent direction cosine, generator response coupling, evaluator availability,
and compute cost. Latent correlations and generator response coupling are
screening proxies, not image-level semantic conclusions.

Initial metric baselines:

| ID | Metric | Purpose |
|---|---|---|
| B1 | \(I\) | Euclidean control |
| B2 | \(g_s(w_0)\) | Static mapping metric |
| B3 | \(g_s(w_k)\) | Dynamic mapping metric |
| B4 | \(\widetilde g_G(w_k)+\mu I\) | Dynamic generator metric |
| B5 | Combined \(M(w_k)\) | Integrated metric, only after evidence |

A static generator-metric baseline must be included when B4 is introduced, so
that the effects of generator awareness and dynamic updating are not conflated.
The initial comparison is B1/B2/B3. Next compare B1/B3/B4 only when the previous
result makes that comparison informative. Run B5 only when both components have
independent justification.

#### Direction objective

For target direction \(d_t\), solve

\[
\min_v \frac12v^TMv
\quad\text{subject to}\quad d_t^Tv=1.
\]

Use stable linear solves, not an explicit matrix inverse. The closed-form
solution is proportional to \(M^{-1}d_t\), normalized to satisfy the constraint.

#### Intensity objective

For requested target intensity \(\tau\), solve

\[
\min_w \frac12\|T(w)-T(w_0)\|^2
\quad\text{subject to target progress}=\tau.
\]

The exact target-progress function and its measurement domain must be stated.

#### Disentanglement objective

For constraint matrix \(A\) and requested responses \(b\), solve

\[
\min_v \frac12v^TMv
\quad\text{subject to}\quad A^Tv=b.
\]

The analytic form is

\[
v=M^{-1}A\left(A^TM^{-1}A\right)^{-1}b,
\]

but implementation must use stable factorizations/solves and report rank or
conditioning failures. Compare against Euclidean projection and the applicable
InterFaceGAN conditional-manipulation baseline.

#### Independent evaluation requirement

A whole-image generator metric can reduce target-related image change as well
as unwanted change. Therefore a lower generator energy is not sufficient.
Every semantic claim requires matched independent target progress plus an
off-target or image-quality measure not identical to the optimized objective.

If no validated independent evaluator is available, classify the result as
**proxy-only** and do not make an image-level semantic claim.

Track B outcomes use exactly one of:

- **SUPPORTED** — held-out independent evidence supports the hypothesis;
- **INCONCLUSIVE** — valid experiment, insufficient precision or sample size;
- **UNSUPPORTED** — valid evidence fails the predeclared criterion;
- **INVALID** — implementation, numerical, data, or evaluation failure prevents
  interpretation.

---

## 5. Integration gate

Do not compare bases by downstream semantics until all conditions hold:

1. at least two representations pass Track A;
2. a feasible semantic optimization mechanism passes Track B validation;
3. the cross-track hypothesis is explicit and falsifiable;
4. an independent semantic evaluator or an explicitly proxy-only protocol is
   available; and
5. all other variables can be held fixed while changing the basis.

If valid bases are empirically equivalent, retain the simplest as the primary
representation and one alternative as a control. Equivalence is a result, not
a failed study.

---

## 6. Experimental protocol

### 6.1 Dataset partitions and sample sizes

- pilot: approximately 20 paired samples;
- extended study: at least 50 paired samples;
- confirmation: preferably 100 paired samples;
- discovery/tuning and held-out confirmation sets must be distinct;
- every baseline uses the same seeds, latent codes, target steps, and checkpoint.

The exact counts may change for compute reasons, but the reason and stopping
rule must be recorded before inspecting the final comparison.

### 6.2 Standardized path energy

For a path parameterized by \(t\), report midpoint energy as

\[
E \approx \frac12\sum_k
\frac{\Delta w_k^T M(w_{\mathrm{mid},k})\Delta w_k}{\Delta t_k}.
\]

The path parameterization, step widths, midpoint definition, and metric
normalization must be recorded. Energies computed with different formulas or
parameterizations are not directly comparable.

### 6.3 Statistics

Each result table must include:

- sample count and failure count;
- mean, median, and standard deviation;
- paired effect relative to the declared baseline;
- 95% confidence interval for the paired effect;
- per-sample data or a reproducible machine-readable summary.

### 6.4 Provisional semantic screening threshold

A method passes an early semantic screen only if it produces at least a 5%
reduction in normalized off-target variation at matched independent target
progress, with a compatible uncertainty interval and no unacceptable increase
in failure rate. This threshold is provisional and must be frozen before the
held-out confirmation run.

---

## 7. Adaptive research loop

After every experiment:

1. **Observe** — record raw results, failures, runtime, memory, and deviations.
2. **Validate** — run invariants, numerical checks, and reproducibility checks.
3. **Analyze** — separate measured observations from interpretation.
4. **Classify** — `ACCEPT`, `REVISE`, `REJECT`, or `INCONCLUSIVE` for the tested
   hypothesis or implementation candidate.
5. **Generate candidates** — propose no more than three next experiments.
6. **Select** — choose the smallest experiment with the highest expected
   information value.
7. **Execute** — run only that bounded experiment.
8. **Report** — create an immutable round report and update cumulative state.

### Pivot rules

- If `sinh` is unstable, reduce alpha, compare matched nonlinearity, consider a
  scaled stable implementation, and deprioritize it if instability persists.
- If B2 and B3 remain equivalent, test one longer valid path and sensitivity to
  discretization/noise; then reject the dynamic-mapping advantage claim if the
  difference remains negligible.
- If B4 outperforms B3, retain `sinh` as an ablation rather than assuming it is
  part of the mechanism.
- If B5 does not outperform B4 independently, reject the added mapping-metric
  component for that objective.
- If a semantic pair is weak, verify labels/directions and literature, choose a
  stronger pair on discovery data, then freeze it.
- If no trustworthy evaluator exists, continue only as proxy-only research.
- If bases are equivalent, keep the simplest primary basis and one control.
- Preserve and report every scientifically valid negative result.

---

## 8. Evidence and repository state

After approval, research state should live under `experiments_gdsa/`:

```text
experiments_gdsa/
├── gdsa_research_guide.md
├── README.md
├── research/
│   ├── hypotheses.md
│   ├── literature_matrix.md
│   ├── feasibility_matrix.md
│   ├── variant_registry.md
│   └── next_experiment.md
├── rounds/
│   ├── round_000_audit/
│   ├── round_001_track_a/
│   └── ...
├── experimental_report.md
├── 00_setup_ro_sample/
├── 00_setup_jocobian_sample/
├── 01_sinh_none_linear_mapping_inversion/
└── 02_generator_jacobian_ro/
```

Existing numbered experiments remain intact as historical pilots. New adaptive
rounds should reference them rather than silently rewriting their conclusions.

### Required literature matrix fields

- paper title;
- authors;
- year and venue;
- DOI or stable URL;
- exact section/page supporting the claim;
- claim used by GDSA;
- applicability to this repository;
- limitations or conflicting evidence.

### Required immutable round report

Name every new report
`experimental_report_YYYYMMDD_RXX_EXX.md`; never overwrite a prior report.
Each report contains:

1. experiment ID and timestamp;
2. track and research question;
3. falsifiable hypothesis;
4. supporting references;
5. environment, checkpoint, seeds, precision, and configuration;
6. mathematical definitions;
7. objective and constraints;
8. baselines and controls;
9. numerical validation;
10. image/proxy evaluation;
11. statistical summary;
12. observations;
13. interpretation;
14. limitations and failures;
15. decision and next recommendation;
16. source revision and exact reproduction commands.

`experimental_report.md` is the cumulative index and evidence summary. It must
link to immutable reports and distinguish planned, running, completed, and
invalid work.

---

## 9. Audit of existing pilot evidence

### Experiment 01 — nonlinear mapping and inversion

The existing result provides useful initial evidence that the implemented
Legendre+`sinh` map can round-trip accurately and that conditioning worsens as
alpha increases. It is not yet sufficient for Track A confirmation because:

- the identity and properly normalized Chebyshev controls are missing;
- raw alpha was compared without matched \(\beta\) or matched conditioning;
- basis construction has not been audited against its discrete measure;
- the semantic significance of metric anisotropy was not tested and must not be
  inferred from numerical validity alone.

### Experiment 02 — generator Jacobian and RO pilot

The existing result is evidence that the implemented constrained direction
solver is internally consistent and that static/dynamic `sinh` metrics behaved
nearly identically in a small pilot. It does not yet demonstrate a
generator-aware optimization benefit because:

- generator finite-difference JVP was used as an evaluation signal, not as the
  B4 optimization metric;
- only two samples were used;
- semantic polarity and independent image-level target progress were not
  validated;
- no independent evaluator was available, so conclusions are proxy-only;
- the reported path-energy discretization differs from this guide's standard;
- a local Pymanopt/analytic agreement validates the solver but does not, by
  itself, establish a novel Riemannian-optimization advantage.

Before comparison with future work, recompute or clearly relabel its energy
using the standardized definition. Do not retroactively replace its original
report.

---

## 10. Compute and stopping policy

- The research program contains 100 sequential adaptive rounds (`R001` through
  `R100`). A round is a complete research decision cycle, not a single
  hyperparameter trial.
- Use pilots before extended or confirmation runs.
- Track GPU hours, wall-clock runtime, peak memory, disk growth, and the number
  of configurations attempted.
- Stop a branch when its predeclared claim is unsupported after the relevant
  pivot check, when numerical validity fails, or when an independent evaluation
  is unavailable and proxy evidence cannot answer the research question.
- Do not expand the search solely because a result is negative.

Every round must be completed, reported, and re-read before the next round may
start. If a planned branch becomes invalid or unsupported, the following round
must investigate the failure, test a justified alternative, or consolidate
evidence; it must not manufacture a redundant experiment merely to consume a
round number.

### Mandatory source review before every round

The authoritative local source corpus is:

- `/home/rz/raymond_project/literature_review_papers` (all non-`.git` files);
- `/home/rz/raymond_project/f_sefa/docs/stage02_literature_deep_analysis_2026-09-28.md`.

Before each round, the agent must:

1. read the preceding immutable round report and cumulative summary;
2. regenerate and compare the complete local source manifest;
3. search the extracted local corpus for the new round's question;
4. inspect the complete relevant pages/sections in the original sources;
5. search the web only when the local corpus lacks required evidence;
6. prefer textbooks, peer-reviewed journal papers, primary conference papers,
   and official primary documentation, in that order when appropriate; and
7. cite every source used by the round with author, title, year, venue or
   publisher, DOI/stable URL when available, section/page, provenance, and
   access date.

The full corpus is inventoried every round. Unchanged documents may reuse a
validated text extraction, but the manifest and search must be rerun and every
claim used in the round must be checked against its original source.

### Mandatory tracing log

Each round has an append-only `tracing_log.md` recording timestamp, action,
inputs, outputs, validation result, and decision. Command failures and abandoned
approaches remain in the log. The trace begins before source inspection and is
closed only after the report has been re-read.

Meaningful progress includes: a reproduced result, a falsified hypothesis, a
validated numerical mechanism, an identified failure mode, an independently
supported semantic improvement, or a justified simplification of the method.

---

## 11. Planned implementation order after approval

1. **Round 000 — Audit:** inventory repository code/checkpoints, reproduce
   Experiments 01/02, validate formulas, and initialize research state files.
2. **Energy normalization:** implement one tested path-energy API and recompute
   comparable pilot summaries without overwriting historical reports.
3. **Track A:** implement identity, normalized Legendre, and normalized
   Chebyshev transforms behind one interface; compare linear and `sinh`
   mappings with matched samples, \(\beta\), precision, and conditioning.
4. **Track B discovery:** build the literature matrix, screen semantic pairs on
   a discovery subset, validate direction polarity and generator coupling, and
   freeze one pair plus its evaluator protocol.
5. **Small discriminating semantic experiment:** run B1/B2/B3 on a paired pilot.
   Introduce B4 only if the result can distinguish a generator-aware mechanism;
   introduce B5 only if both components are independently justified.
6. **Confirm or stop:** expand to held-out 50–100 samples only after the pilot
   passes its gate; otherwise document the negative result and follow the
   relevant pivot rule.

No implementation beyond this guide begins until the proposed order and gates
are approved.

---

## 12. Core references

- Boumal, N. (2023). *An Introduction to Optimization on Smooth Manifolds*.
- Debnath, L., & Mikusiński, P. (2005). *Hilbert Spaces with Applications*.
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*.
- Shen, Y., Gu, J., Tang, X., & Zhou, B. (2020). Interpreting the latent space
  of GANs for semantic face editing (InterFaceGAN), CVPR.
- Shen, Y., & Zhou, B. (2021). Closed-form factorization of latent semantics in
  GANs (SeFa), CVPR.

These references are starting points. Exact claims must be entered into the
literature matrix with section-level evidence before being relied upon.
