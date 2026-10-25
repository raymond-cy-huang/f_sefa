# Soft Exclusion Exploration

Paired Riemannian optimization experiment comparing an exact Hilbert
orthogonal complement against continuously relaxed overlap penalties for the
five known SeFa semantics: Gender, Age, Pose, Glasses, and Smile.

Run with:

```bash
python experiments_soft_exclusion_exploration/run.py
```

Primary outputs are image grids in `outputs/figures/`. Every optimized
direction is checkpointed independently in `outputs/directions/`.

