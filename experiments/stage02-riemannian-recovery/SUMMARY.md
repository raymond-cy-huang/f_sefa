# Stage 02 Riemannian Semantic Recovery - Results

This is a machinery-validation experiment on the Euclidean unit sphere in raw-Legendre coefficient coordinates. It does not claim an advantage over eigendecomposition.

## Core results

- Riemannian vs eigenvector cosine: `1`
- Riemannian vs mapped-GT cosine: `1`
- Mean pairwise displacement cosine: `1`
- Maximum displacement deviation from mean (L2): `0.000000e+00`
- Covariance numerical rank: `1`
- Iterations: `10`; final gradient norm: `8.477764e-16`

## Interpretation

The centered displacements are identical up to floating-point error, as predicted by the strict linearity of the Stage 01 map. Consequently, the semantic covariance is numerically rank one and the direct mapped direction, PCA eigenvector, and Riemannian sphere solution agree up to sign and numerical precision. This validates the tangent projection, retraction, convergence logging, and decode pipeline before a non-trivial Stiefel-manifold experiment.
