"""Unit tests for the standardized GDSA path energy."""

from __future__ import annotations

import numpy as np
import unittest

from energy import build_path


class EnergyTests(unittest.TestCase):
    def test_constant_metric_matches_analytic_energy(self) -> None:
        transform = np.eye(3)
        point = np.array([0.2, -0.4, 0.7])
        target = np.array([1.0, 0.0, 0.0])
        for steps in (1, 2, 5, 10, 50):
            result = build_path(point, target, transform, 0.0, 0.5, steps, "B1_euclidean")
            self.assertLessEqual(abs(result.midpoint_energy - 0.25), 1e-14)
            self.assertLessEqual(abs(result.target_progress - 0.5), 1e-14)

    def test_legacy_definition_is_preserved_and_named(self) -> None:
        result = build_path(
            np.zeros(2), np.array([1.0, 0.0]), np.eye(2), 0.0, 0.5, 5, "B1_euclidean"
        )
        self.assertLessEqual(abs(result.legacy_left_energy - 0.05), 1e-14)
        self.assertLessEqual(abs(result.midpoint_energy - 0.25), 1e-14)

    def test_midpoint_energy_converges_for_scalar_nonlinear_metric(self) -> None:
        alpha = 1.0
        start = 0.2
        tau = 0.5
        transform = np.eye(1)
        point = np.array([start])
        target = np.ones(1)
        antiderivative = lambda value: value / 2.0 + np.sinh(2.0 * alpha * value) / (4.0 * alpha)
        exact = 0.5 * (antiderivative(start + tau) - antiderivative(start))
        errors = []
        for steps in (2, 4, 8, 16, 32):
            result = build_path(point, target, transform, alpha, tau, steps, "B1_euclidean")
            errors.append(abs(result.midpoint_energy - exact))
        self.assertTrue(all(later < earlier for earlier, later in zip(errors, errors[1:])))
        self.assertLess(errors[-1], 1e-5)


if __name__ == "__main__":
    unittest.main()
