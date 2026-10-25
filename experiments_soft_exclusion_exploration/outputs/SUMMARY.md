# Soft Exclusion Exploration

- Seeds: `8` (paired initialization across all regimes)
- Visual identities: `4`
- Regimes: exact Hard, lambda = 10, 1, 0.1, 0
- Optimization: `6` iterations, `8` anchors, resolution `64`, epsilon_H `0.1`, curvature beta `0.1`

## Numerical summary

| Regime | Mean known overlap | Mean image response |
|---|---:|---:|
| hard | 3.424687e-16 | 1.065163e+00 |
| lambda_10 | 1.675702e-03 | 6.220860e-01 |
| lambda_1 | 5.208599e-03 | 7.485646e-01 |
| lambda_0p1 | 6.273987e-03 | 8.625783e-01 |
| lambda_0 | 6.488473e-03 | 8.695207e-01 |

## Qualitative review

- Dominant observed category: **illumination / exposure**. Across the inspected contact sheets, the clearest change is a dark-to-bright or bright-to-dark traversal, often with background illumination changing at the same time.
- Hard directions are not visually inert: several Hard rows also produce strong illumination changes despite numerical orthogonality to all five known semantic directions.
- Lower exclusion pressure generally increases the strength of the exposure change, especially in seed 005, but many rows remain visually redundant across lambda values.
- No consistent glasses, pose, mouth-opening, or clean closed-mouth-to-Smile transition was identified across the four identities.
- Some skin-tone and contrast changes accompany illumination, but the grids do not justify treating them as independent semantics.
- No candidate is labeled as a new semantic. The current evidence most closely matches failure/success case **E**: relaxing orthogonality primarily releases stronger luminance and appearance changes rather than a clearly new fine-grained semantic.

Numerically, overlap is controlled as intended: Hard is effectively zero, and the mean rises from `1.68e-3` at lambda 10 to `6.49e-3` when free. Gender and Age account for the largest mean squared overlap in the weak/free groups; Smile is the smallest component on average. Increased overlap does **not** guarantee increased image response: the paired responses vary substantially by seed, and the Hard group has the largest overall mean because several Hard seeds reach strong responses.

Examples worth follow-up:

- `seed_005`: clearest increase in response as exclusion is relaxed; visually dominated by exposure/contrast.
- `seed_001` and `seed_003`: strong illumination traversals with opposite traversal orientation.
- `seed_007`: regimes are comparatively similar, useful as a near-redundancy control.

## Limitations

This first run uses 8 optimization anchors at 64 px and six RO updates. All groups share each seed's initialization, but Hard and soft optimization occur on different feasible manifolds and can follow different trajectories. The grids support qualitative screening, not final semantic or statistical claims. The small sample, short optimization horizon, and absence of classifiers prevent reliable semantic naming. Apparent skin-tone changes are especially confounded with exposure. Classifier-based labels were intentionally excluded.
