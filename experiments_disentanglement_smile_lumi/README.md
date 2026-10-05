# Smile-Luminance Disentanglement

This experiment first identifies every current ray candidate that jointly
expresses Smile-like mouth motion and global luminance change. Disentanglement
starts only after the candidate subspace has been measured.

## Candidate definitions

Smile response is measured against the direct SeFa Smile image response, but
only after removing each channel's spatial mean. Cosine similarity is measured
inside the aligned FFHQ mouth region, reducing contamination from global
brightness and color shifts.

Luminance response is the mean Rec.709 luminance derivative of the central
image response:

\[
\frac{D(I(c+\epsilon v))-D(I(c-\epsilon v))}{2\epsilon}.
\]

A ray is a joint candidate when both its absolute mean mouth-Smile alignment
and mean absolute luminance slope are at or above the configured quantile. The
default is the 75th percentile for each metric. Continuous scores for all 507
rays are retained, so the threshold can be changed without losing evidence.

Both accepted and skipped source rays are scanned. Their original status is
preserved in every output; skipped rays are diagnostics, not RO discoveries.

## Run

```bash
env PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  python experiments_disentanglement_smile_lumi/scan_candidates.py
```

## Outputs

```text
outputs/
  all_ray_scores.csv
  joint_candidates.csv
  smile_luminance_scatter.png
  scan_manifest.json
  candidates/rayXXX.csv
```

The next stage will construct the candidate Hilbert subspace and optimize two
G-orthogonal directions: Smile with luminance leakage suppressed, and luminance
with Smile leakage suppressed.

The first candidate scan and disentanglement pass are complete. See
`RESULTS.md` for quantitative results, limitations, and the next refinement.

