# Experiment 03 Results

## Verdict

No new semantic direction is validated by the current experiment.

The method successfully produced anonymous directions in a finite-dimensional
Legendre Hilbert chart, removed the five known directions in training
image-response space, and optimized cross-sample residual coherence on a
Stiefel manifold. However, the generated traversals remain dominated by skin
tone, illumination, local contrast, and background coupling. Several candidates
also show mouth or eye changes consistent with incomplete held-out leakage from
the known Smile/Expression response.

## Runs

### Centered RGB response

- Search space: first 128 SeFa eigenvectors excluding ranks 001, 003, 007,
  010, and 018.
- Legendre+sinh parameter: `alpha=0.5`.
- Training residual-response norm fractions: `0.7958`, `0.8773`, `0.7589`.
- Result: strong coherent candidates, but the visual changes are mostly global
  color, illumination, and background changes.
- Artifacts: `outputs/`.

### Face-centered structural response

- Feature operator: fixed elliptical center window followed by grayscale Sobel
  X/Y and local high-pass responses.
- Training residual-response norm fractions: `0.8193`, `0.8782`, `0.8595`.
- Held-out maximum response cosine with a known semantic ranges from roughly
  `0.27` to `0.74` across candidates.
- Result: suppressing raw color in the optimization feature did not eliminate
  low-frequency appearance changes in the final images, and known semantic
  leakage remains substantial for some held-out identities.
- Artifacts: `outputs_face_edges/`.

## Interpretation

The experiment falsifies the simple claim that removing five known responses
and maximizing cross-sample Jacobian coherence is sufficient to isolate an
unknown human-interpretable semantic. It also demonstrates why latent
orthogonality is inadequate: candidate latent cosine with the five known
directions is around `1e-3`, while held-out image-response cosine can exceed
`0.7`.

The current outputs are candidate transformations, not accepted semantic
discoveries. A next experiment needs an explicit nuisance model for
illumination/color and a stronger held-out response-exclusion constraint before
new semantic naming is defensible.
