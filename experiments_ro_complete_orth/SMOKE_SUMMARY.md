# Sequential RO Smoke-Run Record

Recorded: 2026-10-04

This run validates the sequential exclusion machinery with three rays, two
search anchors, four held-out anchors, and at most two RO iterations per
attempt. It is not a formal semantic result.

## Exclusion behavior

- `ray001` was accepted on attempt 1 and removed before searching `ray002`.
- `ray002` was accepted on attempt 1 and removed before searching `ray003`.
- `ray003` attempts 1 and 2 took no valid RO step and were not excluded.
- `ray003` was accepted on attempt 3 and then removed.

The maximum absolute Hilbert cosine with all directions excluded at each stage
was below \(8\times10^{-17}\). Pairwise Hilbert cosines among the three
accepted rays were also at floating-point zero.

## Preliminary held-out behavior

| Ray | Search objective change | Held-out objective change | Status |
|---|---:|---:|---|
| ray001 | +0.5148 | -0.0075 | Pipeline-valid; no held-out gain in this smoke run |
| ray002 | +0.3631 | +1.5563 | Strong held-out gain |
| ray003 | +0.2802 | +0.2090 | Positive held-out gain after retry |

Acceptance is based only on search data. Held-out results are never used to
choose or accept a ray, which preserves the independence of the test set.

The traversal images suggest that several rays still contain photometric or
illumination-related behavior. Function-space orthogonality alone does not
guarantee visually distinct semantics; the experiment therefore records both
pairwise Hilbert cosine and pairwise held-out image-response cosine.

In this run, the absolute cross-ray image-response cosines were 0.201, 0.223,
and 0.241. Thus, the three accepted rays were not only G-orthogonal in function
space; their held-out image responses were also far from collinear.

