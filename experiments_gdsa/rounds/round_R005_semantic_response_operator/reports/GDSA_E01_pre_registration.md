# GDSA-E01 Pre-registration (Repository Adaptation)

## Question

Can a generator semantic-response operator reduce off-target change, and can a
predefined Hilbert basis make a diagonal approximation more accurate or cheaper
than equivalent W-space and random-orthogonal controls?

## Frozen interpretation

At `alpha=0`, an orthogonal map `c = Qw` is only a coordinate change. For
`K_W = J_off^T J_off`, the Hilbert-coordinate operator is
`K_B = Q K_W Q^T`. Therefore a full, exact B3 solve must map back to B2. This is
a negative/null control and cannot support a Hilbert-basis contribution.

The falsifiable basis claim is restricted to B5: whether `diag(K_B)` under a
predefined Legendre or Chebyshev basis offers a better quality/cost tradeoff than
`diag(K_W)` and five seeded random orthogonal bases.

## Staged protocol

1. Exact 2D/16D controls: orthogonality, similarity, adjoint identity, PSD,
   constrained full solution, exact diagonal behavior, Hutchinson identity.
2. Model-matched StyleGAN numerical smoke: shared 512D W, 18 style layers,
   deterministic synthesis, fixed response mask, JVP finite difference, VJP
   adjoint identity, and exact latent target progress.
3. Only after Stage 2 passes: N=20 development pilot for B0–B5 and feasible B7.
4. Confirmation is intentionally not executed in this task, following the
   user's revised instruction that 100 rounds/samples are unnecessary.

## Frozen gates

- Exact controls: all declared relative/absolute errors `< 1e-10`.
- Generator deterministic repeat: exact at measured response.
- Generator JVP finite-difference relative error `<= 0.05` over a documented
  epsilon sweep; JVP/VJP adjoint discrepancy `<= 0.1%` relative scale.
- Target constraint absolute error `<= 1e-6`.
- No semantic or contribution claim without independent image-level endpoints.

## Contribution decisions

- H0 invariance: PASS only when B2/B3 equivalence passes.
- H1 operator value: requires paired independent off-target improvement.
- H2 basis value: requires polynomial-basis Pareto improvement beyond identity
  and five random bases; toy basis dependence is insufficient.
- H3 dynamic value: evaluated only after static H1 evidence.
- B7 is required before a strong novelty claim relative to LowRankGAN.

This adaptation is based on the user-supplied
`GDSA_E01_Semantic_Response_Operator_Experiment.md` and the later Chinese
contribution-analysis note. Those files are treated as experiment design input,
not as evidence that the claims already hold.

