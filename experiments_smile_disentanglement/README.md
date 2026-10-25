# Local Smile Disentanglement

This experiment starts from the mapped F-SeFa Smile direction and performs
local Riemannian refinement in a Hilbert spherical cap. It is intentionally
different from the 507-ray complete-orthogonal search and from hard projection
into the four known nuisance complements.

The primary score is signed Smile response. Candidates are accepted and ranked
using held-out Smile retention, sign consistency, known leakage into Gender,
Age, Pose, and Glasses, and separate luminance leakage.

The run evaluates locality at `tau = 0.95, 0.90, 0.85`, plus `beta=0` and
`gamma=0` ablations. Anchors are split into 16 search, 16 validation, and 32
held-out test identities for this first completed run.

