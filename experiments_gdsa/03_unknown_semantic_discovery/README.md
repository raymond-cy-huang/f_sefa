# Experiment 03: Unknown Semantic Discovery

This experiment directly searches for previously unlabeled candidate semantic
directions. The five confirmed SeFa directions (`001 gender`, `003 age`, `007
pose`, `010 glasses`, `018 smile`) are used only as an exclusion set. They are
not candidate targets and are not used to name the outputs.

The search:

1. spans the first 128 SeFa eigenvectors, excluding the five known axes;
2. maps that span into the Legendre+sinh Hilbert chart;
3. estimates a face-centered structural image Jacobian (grayscale Sobel and
   high-pass features) by deterministic central differences;
4. removes the five known image-response components at every training latent;
5. builds a cross-sample coherent residual-response operator;
6. optimizes eight candidates jointly on a Stiefel manifold;
7. integrates each constant Hilbert tangent through the inverse sinh chart;
8. renders anonymous candidate traversals on held-out latent seeds.

This is an exploratory candidate-proposal experiment. Human-readable semantics
must be assigned only after viewing the outputs.

Run from the repository root:

```bash
conda run -n gdsa python \
  experiments_gdsa/03_unknown_semantic_discovery/run.py
```

The main outputs are `outputs/heldout_seed_*_candidates.png`,
`candidate_metrics.csv`, and `experiment_config.json`.

