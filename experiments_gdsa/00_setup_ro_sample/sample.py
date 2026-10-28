"""Verify Pymanopt with the PyTorch backend on a sphere manifold."""

import os
from importlib.metadata import version

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pymanopt
import torch


def main() -> None:
    np.random.seed(42)
    torch.set_default_dtype(torch.float64)

    manifold = pymanopt.manifolds.Sphere(3)
    matrix = torch.diag(torch.tensor([1.0, 2.0, 5.0]))

    @pymanopt.function.pytorch(manifold)
    def cost(point):
        return -(point @ matrix @ point)

    problem = pymanopt.Problem(manifold=manifold, cost=cost)
    optimizer = pymanopt.optimizers.SteepestDescent(
        max_iterations=200,
        min_gradient_norm=1e-8,
        verbosity=0,
    )
    result = optimizer.run(problem)

    point = np.asarray(result.point)
    rayleigh_quotient = float(point @ matrix.numpy() @ point)
    point_norm = float(np.linalg.norm(point))

    if not np.isclose(point_norm, 1.0, atol=1e-8):
        raise AssertionError(f"Point is not on the unit sphere: {point_norm}")
    if not np.isclose(rayleigh_quotient, 5.0, atol=1e-6):
        raise AssertionError(
            "Pymanopt did not recover the dominant eigenvalue: "
            f"{rayleigh_quotient}"
        )

    print(f"pymanopt={version('pymanopt')}")
    print(f"solution={point}")
    print(f"solution_norm={point_norm:.12f}")
    print(f"rayleigh_quotient={rayleigh_quotient:.12f}")
    print("Pymanopt setup sample: PASS")


if __name__ == "__main__":
    main()
