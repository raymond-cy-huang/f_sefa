# Sequential Complete-Orthogonal RO

This experiment preserves `experiments_ro_verification/` unchanged and extends
its Hilbert-constrained GAN image objective to multiple sequential rays.

After ray \(v_k\) is accepted, it is appended to the excluded subspace:

\[
U_{k+1}=[U,v_1,\ldots,v_k].
\]

The next ray is optimized on

\[
\mathcal M_{k+1}
=\{v:U_{k+1}^TGv=0,\;v^TGv=1\}.
\]

Consequently, a later search cannot return to brightness or any other earlier
accepted ray, up to numerical precision.

## Result layout

Every ray has an independent folder:

```text
experiments_ro_complete_orth/
  ray001/
    metrics.csv
    attempts.csv
    convergence.csv
    exclusions.csv
    direction_coefficients.csv
    response_cosine_random.csv
    response_cosine_ro.csv
    response_spectrum.csv
    traversal_luminance.csv
    direction.npz
    traversal.png
    convergence.png
    SUMMARY.md
  ray002/
  ray003/
  ...
  complete_orth_summary.csv
  pairwise_hilbert_cosine.csv
  pairwise_image_response_cosine.csv
  accepted_rays.npz
  experiment_manifest.json
  completion.json
```

`exclusions.csv` is the key audit file. It reports the Hilbert cosine between
the new ray and every known semantic plus every previously accepted ray.
`attempts.csv` records retries. A trial is accepted only when at least one RO
step is taken and the search objective improves by the configured minimum;
failed trials are marked `skipped`, are never appended to the excluded
subspace, and do not stop later ray slots from running.

`pairwise_hilbert_cosine.csv` verifies mathematical G-orthogonality, while
`pairwise_image_response_cosine.csv` checks whether mathematically distinct
rays still produce similar held-out GAN image behavior.

## Smoke run

```bash
env PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  python experiments_ro_complete_orth/run.py --smoke
```

The smoke configuration generates `ray001/`, `ray002/`, and `ray003/` with two
RO iterations each. It validates sequential exclusion but is not a formal
semantic result.

## Formal run

```bash
env PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  python experiments_ro_complete_orth/run.py --num-rays 10
```

The default formal configuration uses 32 search anchors, 100 held-out anchors,
40 RO iterations per ray, and microbatches of one for an 8 GB GPU.

## Complete 507-ray quick run

```bash
env PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  python experiments_ro_complete_orth/run.py --quick-all
```

This creates `ray001/` through `ray507/`. Rays 1-506 use Sequential RO; when
only one complement dimension remains, `ray507` is the unique residual
G-orthogonal unit direction and has no non-zero tangent direction to optimize.
The quick-complete configuration is an exhaustive pipeline run, not a formal
32-search/100-test semantic study.

Add `--resume` to recover accepted rays from `accepted_rays.npz` and continue
after the highest ray recorded in `complete_orth_summary.csv`.

