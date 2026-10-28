# Pymanopt Setup Sample

This sample verifies that Pymanopt and its PyTorch automatic-differentiation
backend work in the `gdsa` Conda environment. It minimizes a Rayleigh-quotient
objective on a sphere and checks that the result recovers the dominant
eigenvalue of a known symmetric matrix.

Run from the repository root:

```bash
conda run -n gdsa python experiments_gdsa/00_setup_ro_sample/sample.py
```
