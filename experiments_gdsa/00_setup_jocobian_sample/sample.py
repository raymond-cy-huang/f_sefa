"""Verify scipy.differentiate.jacobian against an analytical Jacobian."""

import numpy as np
import scipy
from scipy.differentiate import jacobian


def function(point: np.ndarray) -> np.ndarray:
    """Vectorized map from R^3 to R^2 used by SciPy's Jacobian API."""
    x_1, x_2, x_3 = point
    return np.stack((x_1**2 + x_2, np.sin(x_1) + x_3**3))


def analytical_jacobian(point: np.ndarray) -> np.ndarray:
    x_1, _, x_3 = point
    return np.array(
        [
            [2.0 * x_1, 1.0, 0.0],
            [np.cos(x_1), 0.0, 3.0 * x_3**2],
        ]
    )


def main() -> None:
    point = np.array([0.5, -1.0, 2.0])
    result = jacobian(
        function,
        point,
        tolerances={"atol": 1e-10, "rtol": 1e-8},
    )
    expected = analytical_jacobian(point)

    if not np.all(result.success):
        raise AssertionError(f"Jacobian evaluation failed: status={result.status}")
    if result.df.shape != (2, 3):
        raise AssertionError(f"Unexpected Jacobian shape: {result.df.shape}")
    if not np.allclose(result.df, expected, rtol=1e-8, atol=1e-10):
        raise AssertionError(
            "Numerical and analytical Jacobians differ:\n"
            f"numerical={result.df}\nanalytical={expected}"
        )

    max_abs_error = float(np.max(np.abs(result.df - expected)))
    print(f"scipy={scipy.__version__}")
    print(f"point={point}")
    print(f"jacobian=\n{result.df}")
    print(f"max_abs_error={max_abs_error:.3e}")
    print("SciPy Jacobian setup sample: PASS")


if __name__ == "__main__":
    main()
