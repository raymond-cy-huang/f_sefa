# Stage 02 — SeFa vs F-SeFa Lambda Sweep

This experiment fixes one semantic direction and displays editing intensity as
a lambda sweep. The default direction is the user-confirmed Smile direction,
SeFa eigen rank 18.

The comparison grid is:

```text
columns: lambda = 0, 1, 2, 3, 4, 5, 6
row 1:  SeFa
row 2:  F-SeFa
```

Both rows use the same random latent, generator, semantic eigenvector, edit
sign, and numeric lambda.

## Definitions

SeFa uses the original Euclidean-unit eigenvector scale:

```text
w_sefa(lambda) = w0 + lambda d.
```

F-SeFa represents the same direction in the Stage 01 raw-Legendre Hilbert
space and normalizes it using the Gram matrix:

```text
G[n,n] = 2 / (2n + 1)
d_hat = d / sqrt(d^T G d)
c_fsefa(lambda) = w0 + lambda d_hat
w_fsefa(lambda) = Psi(c_fsefa(lambda)).
```

Consequently, SeFa displacement is `|lambda| ||d||_H`, while F-SeFa
displacement is `|lambda|`. Equal numeric lambda values intentionally expose
the difference between the two intensity parameterizations; they are not
assumed to have equal perceptual strength.

The runner separately validates that the representation-only path
`Psi(Phi(w0 + lambda d))` reproduces ordinary SeFa.

## Run

```bash
PATH=/home/rz/miniconda3/envs/pixel2style2pixel/bin:$PATH \
/home/rz/miniconda3/envs/pixel2style2pixel/bin/python \
  experiments/stage02-intensity/run.py --enable-lpips
```

Use `--semantic` to select another user-confirmed direction and `--lambdas` to
change the sweep.

Outputs:

```text
outputs/
  smile_lambda_sweep.png
  metrics.csv
  summary.json
  images/
    sefa_smile_lambda_<value>.png
    fsefa_smile_lambda_<value>.png
```
