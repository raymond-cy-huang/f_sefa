# SciPy Jacobian Setup Sample

This sample verifies that `scipy.differentiate.jacobian` is available in the
`gdsa` Conda environment. It evaluates a vector-valued function numerically and
checks the result against its analytical Jacobian.

Run from the repository root:

```bash
conda run -n gdsa python experiments_gdsa/00_setup_jocobian_sample/sample.py
```
