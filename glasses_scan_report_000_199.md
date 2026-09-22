# Glasses scan report: eigen_000–199

## Generation

- Model: FFHQ e4e / StyleGAN2 adapter
- Environment: WSL `pixel2style2pixel` conda environment with CUDA fused operators
- Identities: 3 fixed random identities, seed `0`
- Strengths: `-6, -3, 0, +3, +6`
- Directions: `eigen_000` through `eigen_199`

## Visual screening result

The only direction with a repeatable cross-identity eyeglasses/sunglasses
effect is:

```text
eigen_010
effective side: positive (+3 / +6)
identity_000: sunglasses are visible at positive strength
identity_001: sunglasses are visible at positive strength
identity_002: eyeglasses are visible at positive strength
```

For the remaining directions, no non-`eigen_010` direction showed a
consistent eyeglass frame or lens across the three identities. `eigen_003`
shows a one-identity sunglasses-like result at high positive strength, but it
does not repeat across the other identities and is therefore tentative only.
Changes outside
`eigen_010` were mainly age, lighting/color, hair, expression, pose, and face
shape. Isolated dark regions were treated as shadows or artifacts rather than
glasses.

## Interpretation

This supports the age/glasses entanglement hypothesis: in this SeFa run,
glasses appear in the already identified age direction rather than in a
separate eigen direction. No independent glasses candidate is selected from
`eigen_000–199`.

This is a visual screening result; it does not claim that a supervised
glasses classifier could not find a weaker residual signal.
