# Experiment 04: Geometry-Driven Method Search

This experiment compares multiple P0 geometric candidate-proposal objectives
on exactly the same generator, anchors, Legendre Hilbert chart, search span,
known-response exclusion, and held-out identities.

Methods:

1. `average_metric`: principal directions of the mean pullback metric;
2. `response_coherence`: cross-anchor coherent Jacobian response;
3. `common_invariant`: a Stiefel subspace that is approximately invariant
   under every anchor-specific pullback metric.

All methods include the same explicit low-frequency RGB nuisance penalty. The
five known SeFa directions are used only for exclusion and leakage measurement.
Outputs are anonymous candidate traversals plus geometric metrics; a candidate
is not called a semantic until visual consistency and leakage checks pass.

Run:

```bash
conda run -n gdsa python \
  experiments_gdsa/04_geometry_method_search/run.py
```
