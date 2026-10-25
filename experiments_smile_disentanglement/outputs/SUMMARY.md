# Smile Disentanglement — Visual-first Run

- Locality cap: `cos_H >= 0.9`
- Candidates: `8` (accepted: `8`)
- Optimization: signed Smile response minus soft locality penalty; nuisance penalties disabled.
- Search/validation/visual identities: `8/8/8`
- RO iterations per candidate: `4`
- Main visual range: `-3H ... +3H`, where one H step is 0.1 in the Hilbert-normalized rows.
- SeFa is retained as a qualitative raw Euclidean reference; its numeric scale is not Hilbert calibrated.

Primary artifacts are in `outputs/figures/`; all candidate vectors, including failures, are in `outputs/directions/visual_tau090/`.
