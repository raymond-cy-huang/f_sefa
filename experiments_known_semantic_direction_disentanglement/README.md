# Known Semantic Direction Disentanglement

This experiment uses the five project-confirmed SeFa directions:

| Semantic | SeFa eigen rank |
|---|---:|
| Gender | 1 |
| Age | 3 |
| Pose | 7 |
| Glasses | 10 |
| Smile | 18 |

All 507 existing rays are evaluated on the same four held-out anchors. For
each target semantic, the top-quartile image-response matches form its
candidate set. Each candidate is projected into the target SeFa direction's
Hilbert-orthogonal complement. The best projected partner is selected by
target response divided by leakage into the other four known semantics.

The SeFa target and selected orthogonal partner span a two-dimensional Hilbert
plane. A deterministic unit-circle search rotates that plane while retaining
at least 70% normalized target response and reducing mean response to the four
known nuisance semantics.

Each semantic has its own output folder containing `candidates.csv`,
`rotation_curve.csv`, `metrics.csv`, `directions.npz`, `manifest.json`, and
`traversal.png`.

