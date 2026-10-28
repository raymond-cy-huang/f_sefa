"""Small-dimensional independent checks for R003 transform builders."""

import unittest

import numpy as np

from run import chebyshev_transform, identity_transform, legendre_transform


class TransformTests(unittest.TestCase):
    def test_transforms_are_orthogonal(self) -> None:
        for builder in (identity_transform, legendre_transform, chebyshev_transform):
            transform, _ = builder(16)
            np.testing.assert_allclose(transform.T @ transform, np.eye(16), atol=1e-12)

    def test_chebyshev_independent_definitions_agree(self) -> None:
        _, metadata = chebyshev_transform(32)
        for key in (
            "formula_node_max_abs_error",
            "formula_weight_max_abs_error",
            "scipy_node_max_abs_error",
            "scipy_weight_max_abs_error",
        ):
            self.assertLessEqual(float(metadata[key]), 1e-12)


if __name__ == "__main__":
    unittest.main()
