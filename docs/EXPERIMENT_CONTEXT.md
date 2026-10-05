# Function-Space Riemannian Optimization: Experiment Context

Last consolidated: 2026-10-05 (Asia/Taipei)

This document is the current research context for the experiments in this
repository. It records what has been established, what remains exploratory,
and which outputs should be used as evidence.

## 1. Research objective

The project studies semantic directions in a GAN through a finite-dimensional
function-space representation. A latent direction is mapped to raw Legendre
coefficients

\[
v=(v_0,\ldots,v_{511}),
\qquad
\langle a,b\rangle_H=a^TGb,
\qquad
G_{kk}=\frac{2}{2k+1}.
\]

The central operation is Riemannian optimization on a Hilbert sphere, possibly
with known semantic directions removed:

\[
\mathcal M=\{v:U^TGv=0,\;v^TGv=1\}.
\]

The GAN is frozen. Image behavior is measured by symmetric finite differences
at equal Hilbert distance:

\[
I^\pm=G_{GAN}(\Psi(c\pm\epsilon v)).
\]

Important distinction:

- (G)-orthogonality is a mathematical property of function directions.
- Similar or different image responses are empirical generator behavior.
- Neither property alone proves that a direction is a pure human semantic.

## 2. Known semantic reference axes

The project-confirmed SeFa directions are post-annotated known axes, not
supervised ground truth labels:

| Semantic | SeFa eigen rank |
|---|---:|
| Gender | 1 |
| Age | 3 |
| Pose | 7 |
| Glasses | 10 |
| Smile | 18 |

These five directions are used as known constraints, response templates, and
nuisance dimensions in later experiments.

## 3. Experiment history and status

### 3.1 Stage 02 intensity / representation consistency

Path: `experiments/stage02-intensity/`

Purpose: compare ordinary SeFa displacement with its function-space F-SeFa
representation under the same numeric lambda sweep. The representation map
and weighted Hilbert normalization are checked before semantic RO work.

Status: completed machinery/intensity baseline. It is not a disentanglement
result and does not claim that equal lambda means equal perceptual strength.

### 3.2 Stage 02 Riemannian semantic recovery

Path: `experiments/stage02-riemannian-recovery/`

Purpose: validate tangent projection, sphere retraction, convergence, and the
decode pipeline on a deliberately linear rank-one recovery problem.

Recorded result:

- Riemannian/eigenvector cosine: 1.
- Riemannian/mapped-ground-truth cosine: 1.
- Covariance numerical rank: 1.
- Maximum displacement deviation from the mean: 0.
- Final gradient norm: approximately (8.48\times10^{-16}).

Interpretation: this validates the machinery. It is not evidence that RO beats
SeFa, because strict linearity makes direct mapping, PCA, and RO equivalent.

### 3.3 Stage 02 convex-hull crossing

Path: `experiments/stage02-convex-hull/`

Purpose: construct and cross the convex hull of the five known semantic
function-space vertices using only raw Legendre coefficients.

The hull is a finite reference region (C_{known}^H), not the unknown true
semantic-valid region. Boundary crossings are exploratory image-quality tests.
For the fixed seed-0 base, visible structural degradation was manually
bracketed around +1.12 to +1.28 H for one ray and +0.80 to +0.96 H for the
other. These are ray-specific brackets, not a global validity boundary.

### 3.4 Function-space RO verification

Path: `experiments_ro_verification/`

Purpose: test whether Hilbert-constrained RO improves an image-response
objective over the same randomly initialized feasible directions.

The smoke result showed a held-out objective improvement of approximately
0.9505 for RO over Random. Geometry checks reached approximately (10^{-16})
orthogonality error and (10^{-15}) complement Gram error.

Interpretation: RO finds a stronger/smoother structured GAN variation under the
chosen objective. This is not yet semantic discovery. The first clear image
finding was a brightness/illumination direction.

### 3.5 Sequential complete-orthogonal ray search

Path: `experiments_ro_complete_orth/`

Purpose: repeatedly optimize rays while appending every accepted ray to the
excluded Hilbert subspace. This prevents later searches from returning to an
already accepted function direction.

Run status:

- 507 ray slots executed.
- 460 accepted rays.
- 47 skipped rays.
- Every accepted pair is mathematically G-orthogonal up to numerical error.
- Image-response effective rank is much smaller than 460; many mathematically
  distinct rays produce related image changes.

This was a quick exhaustive pipeline run with small search/test counts and
64-pixel objectives. It is an exhaustive discovery scaffold, not a formal
large-sample semantic benchmark.

Important observations:

- Brightness/illumination appears repeatedly and was the first major finding.
- Many rays retain Smile-like changes; one SeFa Smile axis does not span all
  Smile-like generator behavior.
- `ray057` may resemble skin tone, but that interpretation is not formally
  established as an independent semantic.
- `ray129` is a key mixed example: its traversal clearly moves from closed
  mouth toward a visible Smile while also changing luminance.
- `ray129` was skipped by the source search-improvement criterion, but its
  recorded direction remains useful as an image-semantic reference.

### 3.6 Smile/luminance disentanglement: first pass

Path: `experiments_disentanglement_smile_lumi/`

The first pass scanned all 507 rays and found 46 joint candidates using a
SeFa-Smile mouth-response alignment plus luminance threshold. It then optimized
within the candidate span and produced mathematically G-orthogonal
Smile/luminance directions.

Limitation discovered: squared template alignment cannot distinguish
Smile-increasing from Smile-decreasing motion. A direction can score highly
while producing different signs for different identities. Therefore the first
`smile_only` result is a response-aligned direction, not a fully validated
closed-mouth-to-Smile semantic.

### 3.7 Ray129-centered orthogonal decomposition

Path: `experiments_disentanglement_smile_lumi/outputs/ray129_orthogonal/`

This is the corrected follow-up. It does not invent a random Smile direction.
It treats `ray129` as the fixed mixed reference and searches the existing
candidate set for G-orthogonal partners.

Results:

- 27 candidates were exactly G-orthogonal to `ray129` within (10^{-8}).
- `ray039` was the highest-scoring orthogonal partner.
- The pair cosine was approximately (3.82\times10^{-17}).
- A two-dimensional Hilbert rotation reduced mean luminance slope from 0.1084
  to -0.0081.
- The rotated pair cosine was approximately (1.04\times10^{-17}).

Interpretation: this is the strongest current evidence that a mixed
Smile/luminance ray can be decomposed through a known orthogonal function-space
partner. RMS luminance leakage still remained, so this is a first plane
decomposition, not complete identity-independent disentanglement.

### 3.8 Known five-semantic direction disentanglement

Path: `experiments_known_semantic_direction_disentanglement/`

Purpose: generalize the ray129 procedure to Smile, Gender, Age, Glasses, and
Pose. All 507 rays were scored against all five SeFa image-response templates.
For each target, 127 top-quartile candidates were retained, projected into the
target's (G)-orthogonal complement, and used to form a two-dimensional
Hilbert plane. A unit-circle search retained target response while minimizing
mean response to the other four known semantics.

Final held-out summary:

| Target | Selected ray | Target retained | Nuisance before | Nuisance after |
|---|---|---:|---:|---:|
| Smile | ray046 | 0.697 | 0.0614 | 0.0351 |
| Gender | ray386 | 0.834 | 0.0745 | 0.0551 |
| Age | ray232 | 0.749 | 0.0947 | 0.0599 |
| Glasses | ray223 | 0.726 | 0.0462 | 0.0294 |
| Pose | ray224 | 0.810 | 0.1111 | 0.0869 |

All final target/complement pairs were G-orthogonal below (3.2\times10^{-16}).
The generator-reevaluated Smile retention was 0.697, slightly below the 0.70
linearized selection threshold because finite-step image response is not
perfectly linear.

Interpretation: nuisance response to the other four known SeFa templates was
reduced for all five targets. This does not prove independence from unknown
attributes, illumination, color, artifacts, or all identity-dependent effects.

## 4. Current evidence hierarchy

Strongest current claims:

1. The Legendre/Hilbert representation and its metric operations are
   numerically consistent.
2. RO can improve the selected image-response objective on held-out anchors.
3. Sequential exclusion can generate hundreds of mathematically orthogonal
   function directions.
4. A mixed ray such as ray129 can be paired with an existing orthogonal ray to
   reduce a measured nuisance component.
5. The same procedure reduces response leakage among five known SeFa templates.

Claims not yet established:

- A globally pure semantic direction for any attribute.
- Independence from semantics not included in the five known templates.
- Identity-independent monotonic Smile, Age, Gender, Pose, or Glasses scores.
- A universal GAN-manifold validity radius.
- Superiority over SeFa under a fixed external semantic predictor.

## 5. Canonical output locations

- [Stage 02 RO recovery](../experiments/stage02-riemannian-recovery/SUMMARY.md)
- [RO verification](../experiments_ro_verification/outputs/SUMMARY.md)
- [Complete orthogonal RO](../experiments_ro_complete_orth/README.md)
- [Smile/luminance results](../experiments_disentanglement_smile_lumi/RESULTS.md)
- [Known five-semantic results](../experiments_known_semantic_direction_disentanglement/RESULTS.md)
- [Five-semantic aggregate CSV](../experiments_known_semantic_direction_disentanglement/outputs/summary.csv)

## 6. Recommended next research step

Replace response-template power with signed, exposure-invariant semantic
measurements. For Smile in particular, select closed-mouth anchors and verify
monotonic progression from closed mouth to Smile. For the other four semantics,
use fixed external attribute scores or carefully defined geometric scores in a
separate validation stage. Keep the function-space (G)-orthogonality and
Riemannian constraints unchanged, and report both mathematical orthogonality
and image-level semantic leakage.

