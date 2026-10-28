# R005 Experimental Report — GDSA-E01 Semantic Response Operator

## 1. Executive verdict

**Overall: INCONCLUSIVE; NO VALIDATED DISTINCT CONTRIBUTION.**

The experiment is now concretely specified and its mathematical core is
implemented. Exact controls prove that a full response operator in orthogonal
Hilbert coordinates is the same optimization as the W-space operator. A basis
can affect a diagonal approximation, but the present evidence does not show that
Legendre or Chebyshev is better for StyleGAN. The real-generator derivative
smoke failed the frozen accuracy gate, so an N=20 semantic pilot and any
independent semantic claim were correctly withheld.

## 2. Research question

Can local generator response reduce off-target image change at matched Smile
progress, and does a predefined Hilbert basis improve the accuracy/cost of a
diagonal operator beyond W coordinates and random orthogonal controls?

## 3. Mathematical result

For orthogonal `Q`, `K_B = Q K_W Q^T` and `a_B = Qd`. Solving
`min 0.5 u^T(K_B+mu I)u` subject to `a_B^T u=1` and mapping `v=Q^T u`
must recover the W-space solution. The 15 exact controls confirmed this with a
maximum relative direction error of `1.21e-15`. Therefore B3 matching B2 is
correctness evidence, not a Hilbert contribution.

## 4. What remains falsifiable

`diag(QKQ^T)` changes with Q. The exact controls produced off-diagonal fractions
from `0.0307` to `0.6496`. This establishes a real basis-dependent compression
question: polynomial bases must beat identity and five random bases on
generator data at comparable full response cost and independent edit quality.

## 5. Implementation

- `operator_response.py`: Q/Q-transpose transport, JVP/VJP operator, Gram
  matvec, exact full solve, diagonal solve, Hutchinson diagonal estimator.
- `run.py`: deterministic 2D/16D controls and raw CSV/JSON output.
- `test_operator_response.py`: coordinate invariance, adjoint/PSD identities,
  and exhaustive 4D Hutchinson expectation.
- `generator_smoke.py`: model-matched StyleGAN1/InterFaceGAN numerical chain.

## 6. Exact-control results

All three unit tests passed. Across 15 basis/dimension cases, maximum errors were:

| Check | Maximum |
|---|---:|
| Orthogonality spectral error | `1.30e-14` |
| Similarity Frobenius error | `1.73e-15` |
| Adjoint absolute error | `1.78e-15` |
| Full W-direction relative error | `1.21e-15` |
| Constraint absolute error | `2.22e-16` |

## 7. Repository compatibility audit

The existing GDSA Exp02 uses e4e/StyleGAN2 and a post-annotated SeFa rank. It is
not a verified InterFaceGAN Smile direction. The adjacent InterfaceGAN repo
contains unit-norm StyleGAN1 FFHQ W Smile and Pose boundaries. Consequently the
smoke used `stylegan_ffhq_2019` with its matching W-Smile boundary, not e4e.
The model exposes 512D W and 18 style layers, matching the declared shared-W
construction.

## 8. Generator smoke result

Checkpoint SHA-256:
`67e412deaec0317e57040a34f5ecd2b3f9b45865a17d56bfeb4abd064d914c39`.
Boundary SHA-256:
`5a729c200c2c933806116cb77624ccedbd6fce21aa98fca8be49aeaa35caa1c7`.

Repeated synthesis response was exact and target progress error was zero. The
best finite-difference comparison occurred at epsilon `0.1`: relative error
`0.0951`, cosine `0.9959`. This exceeds the frozen 5% gate. The adjoint error was
`1.89e-05`, roughly 0.19% of the observed inner product, also above the 0.1%
relative gate. Smaller finite-difference steps degraded because the float32
full-resolution synthesis chain did not resolve the perturbation accurately.

## 9. Why execution stopped

Matrix-free CG with a nominal residual of `1e-5` is not defensible when the
operator derivative checks disagree at percent scale. Running N=20 would turn a
known numerical defect into misleading aggregate results. The fixed geometric
border used in the smoke is also only a provisional gradient target, not the
independent audit mask needed for a semantic claim.

## 10. Hypothesis decisions

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H0 full W/H invariance | PASS | exact direction error `<=1.21e-15` |
| H1 response-operator effectiveness | INCONCLUSIVE | no authorized N=20 independent pilot |
| H2 Hilbert-basis approximation value | INCONCLUSIVE | basis dependence only on analytic toys |
| H3 dynamic geometry value | NOT TESTED | correctly gated after H1 |

## 11. Prior-art boundary

LowRankGAN already relates regional generator Jacobians to low-rank and null
subspaces; InterFaceGAN already provides semantic boundaries and conditional
projection; Shao et al. already study generator-induced Riemannian geometry.
Accordingly, neither Jacobian use, coordinate transport, nor a pull-back metric
is independently novel. A defensible result would require a verified quality–
cost advantage of the predefined basis approximation, or a distinct objective
with evidence beyond these baselines.

## 12. Missing evidence

- a derivative implementation that passes the frozen gate;
- fixed development/held-out sample IDs and a locked N=20 protocol;
- independent Smile, identity, pose, and background evaluators/masks;
- B7 LowRankGAN-style faithful comparator;
- paired uncertainty estimates and failure accounting.

## 13. Smallest next discriminating experiment

Repair the StyleGAN JVP/VJP chain on one sample first. Compare reverse-mode
`J^T(Jv)` against finite differences of `0.5||r(w)||^2` and sweep output
precision/resolution while freezing synthesis noise. Only when both the
directional and adjoint tests fall below declared tolerances should an N=5 smoke
and then N=20 B0/B2/B4/B5 pilot run. Confirmation remains optional under the
user's revised scope.

## 14. Sources used this round

- Shen et al., *InterFaceGAN*, local primary PDF and CVF Open Access.
- Zhu et al., *Low-Rank Subspaces in GANs*, NeurIPS 2021 primary page; local
  corpus contained citations but not the primary paper.
- Shao et al., *The Riemannian Geometry of Deep Generative Models*, CVF primary
  page; local corpus contained only a bibliography mention.
- Boumal, *An Introduction to Optimization on Smooth Manifolds*, local textbook
  and official book page.
- Debnath and Mikusiński, *Hilbert Spaces with Applications*, local textbook.
- Repository `docs/stage02_literature_deep_analysis_2026-09-28.md`.

## 15. Reproducibility

- Repository revision: `22cca23ed7001c77a033b4a3d1978041079d677e`
- Environment: conda `gdsa`, PyTorch `2.5.1+cu121`
- GPU: NVIDIA GeForce RTX 3070
- Seed: `20261025`
- Raw outputs: `results/toy_basis_diagnostics.csv`, `results/summary.json`,
  `results/generator_smoke.json`
- Test command and runners are recorded in `reports/GDSA_E01_math_checks.md`.

## 16. Final contribution answer

1. Semantic-response operator improvement: **INCONCLUSIVE**.
2. Independent Hilbert-basis computational benefit: **INCONCLUSIVE**.
3. Dynamic Riemannian update benefit: **NOT TESTED**.
4. Distinct novelty over LowRankGAN/InterFaceGAN: **NOT ESTABLISHED**.
5. Current defensible statement: the exact invariance/null control and the
   basis-dependent approximation test are now correctly formulated and
   executable; no semantic or novel-method contribution has yet survived an
   independent generator experiment.

