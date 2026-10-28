# GDSA-E01 Mathematical and Numerical Checks

## Outcome

The exact-control stage passes. The StyleGAN derivative smoke does not pass its
pre-registered finite-difference gate, so the semantic pilot is not authorized.

## Exact controls

The runner used dimensions 2 and 16, Identity, a fixed 2D rotation,
R003-validated Legendre/Chebyshev transforms where applicable, and five seeded
random orthogonal matrices. Across 15 rows:

- maximum orthogonality spectral error: `1.30e-14`;
- maximum `K_B - Q K_W Q^T` Frobenius error: `1.73e-15`;
- maximum adjoint absolute error: `1.78e-15`;
- maximum mapped full-direction relative error: `1.21e-15`;
- maximum target-constraint absolute error: `2.22e-16`.

Thus full SRO is coordinate invariant to numerical precision. The observed
off-diagonal fraction ranged from `0.0307` to `0.6496`, proving that diagonal
approximation is basis dependent on the controls, but not that polynomial bases
are advantageous for StyleGAN.

The unit suite also exhaustively averages all 16 Rademacher probes in a 4D case
and recovers the exact Gram diagonal, checking the Hutchinson implementation.

## Generator smoke

The model-matched pair was StyleGAN1 FFHQ (`stylegan_ffhq_2019`, 512D W, 18
layers) and `stylegan_ffhq_smile_w_boundary.npy` (unit norm). Repeated response
was exactly deterministic and latent target progress error was zero. However:

- best JVP finite-difference relative error was `0.0951` at epsilon `0.1`, above
  the frozen `0.05` gate;
- smaller epsilons degraded sharply in float32;
- JVP/VJP adjoint absolute discrepancy was `1.89e-05` (about `0.19%` of the
  measured inner product), above the frozen relative `0.1%` gate.

The derivative chain is directionally consistent (best finite-difference cosine
`0.9959`) but not accurate enough for a claimed `1e-5` CG solve. The next repair
must isolate synthesis operations causing forward/reverse derivative mismatch,
or use a verified reverse-mode construction with an error budget compatible
with the solver tolerance.

## Commands

```bash
/home/rz/miniconda3/envs/gdsa/bin/python -m unittest -v
/home/rz/miniconda3/envs/gdsa/bin/python run.py
/home/rz/miniconda3/envs/gdsa/bin/python generator_smoke.py
```

