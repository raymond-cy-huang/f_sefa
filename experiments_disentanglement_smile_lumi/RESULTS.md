# Smile-Luminance Disentanglement - First Pass

Recorded: 2026-10-04

## Candidate discovery

All 507 complete-orthogonal ray folders were rescored on the same four held-out
anchors at 64 x 64 response resolution.

- Smile reference: direct SeFa Smile image response (rank 18).
- Global per-channel means were removed before Smile comparison.
- Smile alignment was measured in the aligned FFHQ mouth crop.
- Luminance was measured as the Rec.709 mean derivative.
- Joint threshold: 75th percentile for both scores.
- Smile threshold: 0.315981.
- Luminance threshold: 0.224167.
- Joint candidates: 46 / 507.
- Source status: 44 accepted, 2 skipped.
- Candidate Hilbert-subspace rank: 46.

`ray129` was not selected by the joint threshold. Its mouth-Smile alignment was
0.351553 (above threshold), but its mean absolute luminance slope was 0.108430
(below threshold). It remains a useful Smile-leakage boundary case.

## First disentanglement result

RO first optimized a luminance direction in the 46-dimensional candidate
subspace. That direction was then removed with a G-orthogonal reduced-space
null basis before optimizing the Smile direction.

The final Hilbert cosine was

\[
\langle v_{\mathrm{lumi}},v_{\mathrm{smile}}\rangle_H
=1.35\times10^{-16}.
\]

Thus, the two function-space directions are numerically G-orthogonal.

### Held-out normalized response powers

| Direction | Smile power | Luminance power | Target purity |
|---|---:|---:|---:|
| Luminance-only | 22.555 | 34.708 | 0.606 |
| Smile-only | 18.945 | 5.229 | 0.784 |

Target purity is target power divided by target plus leakage power. The
Smile-only direction is substantially cleaner than the mixed candidates, but
still changes illumination. The Luminance-only direction remains more strongly
entangled with the Smile template and overexposes images at large positive
Hilbert distances.

## Interpretation

This first pass establishes that the mixed 46-dimensional subspace contains
two numerically orthogonal directions with substantially different response
profiles. It does not yet establish complete semantic disentanglement.

The main remaining issue is that the current image-template Smile score is
still sensitive to exposure: strong illumination changes alter tooth and mouth
contrast even after spatial-mean removal. The next revision should either:

1. increase the Smile-leakage penalty for the luminance objective;
2. replace template alignment with exposure-invariant mouth geometry or a fixed
   Smile estimator; or
3. project the Smile response itself out of the luminance response objective.

The useful traversal range is also smaller for the luminance direction; the
current +/-0.3 Hilbert endpoints saturate highlights, so formal evaluation
should include +/-0.05, +/-0.10, and +/-0.15.

## Evidence

- `outputs/all_ray_scores.csv`
- `outputs/joint_candidates.csv`
- `outputs/smile_luminance_scatter.png`
- `outputs/scan_manifest.json`
- `outputs/disentangled/disentangled_metrics.csv`
- `outputs/disentangled/disentangled_directions.npz`
- `outputs/disentangled/disentangled_traversal.png`
- `outputs/disentangled/disentanglement_manifest.json`
- `outputs/disentangled/smile_template_traversal.png`

## Ray129-centered orthogonal analysis

The follow-up analysis treats `ray129` as the fixed entangled reference rather
than constructing a new unconstrained Smile direction. Among the 46 previously
identified candidates, 27 are already numerically G-orthogonal to `ray129`
(absolute Hilbert cosine at or below `1e-8`). This is expected for accepted
rays that had already been excluded when the skipped `ray129` slot was drawn.

`ray039` is the highest-scoring existing joint candidate in that exactly
orthogonal set:

\[
\langle r_{129},r_{039}\rangle_H=3.82\times10^{-17}.
\]

The two-dimensional Hilbert plane spanned by `ray129` and `ray039` was rotated
by a held-out luminance least-squares coefficient of `0.12617111`. The rotated
axes remain numerically G-orthogonal (cross cosine `1.04e-17`).

| Direction | Mean luminance slope | RMS luminance slope |
|---|---:|---:|
| `ray129` | 0.108435 | 0.145002 |
| `ray039` | -0.921776 | 0.925964 |
| ray129 luminance-cancelled rotation | -0.008133 | 0.085268 |
| G-orthogonal complement | 0.928643 | 0.934000 |

The mean luminance leakage of the ray129-centered axis drops by about 92.5%,
while its RMS leakage across the four held-out anchors drops by about 41.2%.
The traversal retains the clearest closed-mouth-to-Smile transition on seed
000 and reduces illumination drift. This is a successful first orthogonal
plane decomposition, but not complete semantic disentanglement across all
identities.

Evidence:

- `outputs/ray129_orthogonal/candidate_orthogonality.csv`
- `outputs/ray129_orthogonal/rotation_metrics.csv`
- `outputs/ray129_orthogonal/rotated_directions.npz`
- `outputs/ray129_orthogonal/ray129_orthogonal_rotation.png`
- `outputs/ray129_orthogonal/manifest.json`

