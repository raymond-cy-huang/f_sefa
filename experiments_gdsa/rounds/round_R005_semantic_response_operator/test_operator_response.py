import itertools
import unittest

import numpy as np

from operator_response import explicit_linear_operator
from run import bases


class OperatorResponseTests(unittest.TestCase):
    def test_full_solution_is_coordinate_invariant(self):
        rng = np.random.default_rng(7)
        jacobian = rng.normal(size=(11, 8))
        target = rng.normal(size=8)
        target /= np.linalg.norm(target)
        reference = explicit_linear_operator(jacobian, np.eye(8)).solve_full(target, 0.2)
        for q in bases(8).values():
            operator = explicit_linear_operator(jacobian, q)
            candidate = q.T @ operator.solve_full(q @ target, 0.2)
            np.testing.assert_allclose(candidate, reference, rtol=1e-11, atol=1e-11)
            np.testing.assert_allclose(target @ candidate, 1.0, rtol=0, atol=1e-12)

    def test_exact_rademacher_average_recovers_diagonal(self):
        rng = np.random.default_rng(9)
        jacobian = rng.normal(size=(7, 4))
        operator = explicit_linear_operator(jacobian, bases(4)["legendre"])
        estimates = []
        for values in itertools.product((-1.0, 1.0), repeat=4):
            z = np.asarray(values)
            estimates.append(z * operator.k_mv(z))
        exact_average = np.mean(estimates, axis=0)
        np.testing.assert_allclose(exact_average, np.diag(operator.explicit_k()), atol=1e-12)

    def test_adjoint_and_psd_identities(self):
        rng = np.random.default_rng(11)
        jacobian = rng.normal(size=(9, 6))
        operator = explicit_linear_operator(jacobian, bases(6)["random_orthogonal_seed_3"])
        u = rng.normal(size=6)
        y = rng.normal(size=9)
        np.testing.assert_allclose(operator.apply_a(u) @ y, u @ operator.apply_adjoint(y), atol=1e-12)
        np.testing.assert_allclose(u @ operator.k_mv(u), np.linalg.norm(operator.apply_a(u)) ** 2, atol=1e-11)
        self.assertGreaterEqual(u @ operator.k_mv(u), -1e-12)


if __name__ == "__main__":
    unittest.main()

