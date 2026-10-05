"""CPU-only unit tests for Hilbert and Riemannian geometry."""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np
import torch


EXPERIMENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from geometry import HilbertComplement, sphere_retraction, sphere_tangent_projection


class HilbertGeometryTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.known = rng.normal(size=(12, 3))
        self.geometry = HilbertComplement(self.known)

    def test_complement_constraints(self):
        diagnostics = self.geometry.diagnostics
        self.assertLess(diagnostics.max_known_complement_inner, 1e-12)
        self.assertLess(diagnostics.max_complement_gram_error, 1e-12)

    def test_candidate_is_feasible(self):
        rng = np.random.default_rng(3)
        z = self.geometry.normalize_reduced(
            rng.normal(size=self.geometry.complement_dimension)
        )
        errors = self.geometry.constraint_errors(self.geometry.direction(z))
        self.assertLess(errors["orthogonality_l2"], 1e-12)
        self.assertLess(errors["hilbert_norm_error"], 1e-12)

    def test_projection_is_feasible(self):
        rng = np.random.default_rng(5)
        projected = self.geometry.project_and_normalize(rng.normal(size=12))
        errors = self.geometry.constraint_errors(projected)
        self.assertLess(errors["orthogonality_l2"], 1e-12)
        self.assertLess(errors["hilbert_norm_error"], 1e-12)

    def test_sphere_gradient_and_retraction(self):
        torch.manual_seed(9)
        z = torch.randn(9, dtype=torch.float64)
        z = z / torch.linalg.vector_norm(z)
        ambient = torch.randn_like(z)
        tangent = sphere_tangent_projection(z, ambient)
        self.assertLess(abs(float(torch.dot(z, tangent))), 1e-12)
        updated = sphere_retraction(z, 0.2 * tangent)
        self.assertLess(abs(float(torch.dot(updated, updated)) - 1.0), 1e-12)

    def test_hilbert_gradient_represents_the_same_differential(self):
        rng = np.random.default_rng(23)
        gram = self.geometry.gram_diag
        coordinate_gradient = rng.normal(size=len(gram))
        hilbert_gradient = coordinate_gradient / gram
        perturbation = rng.normal(size=len(gram))
        coordinate_differential = float(coordinate_gradient @ perturbation)
        hilbert_differential = float(
            perturbation @ (gram * hilbert_gradient)
        )
        self.assertAlmostEqual(coordinate_differential, hilbert_differential, 12)

    def test_reduced_gradient_matches_function_coordinate_chain_rule(self):
        rng = np.random.default_rng(29)
        coordinate_gradient = rng.normal(size=self.geometry.dimension)
        expected = self.geometry.q.T @ coordinate_gradient
        q = torch.tensor(self.geometry.q, dtype=torch.float64)
        z = torch.randn(self.geometry.complement_dimension, dtype=torch.float64)
        z.requires_grad_(True)
        value = torch.tensor(coordinate_gradient, dtype=torch.float64) @ (q @ z)
        value.backward()
        np.testing.assert_allclose(z.grad.numpy(), expected, atol=1e-12, rtol=1e-12)

    def test_directional_derivative_and_taylor_remainder(self):
        torch.manual_seed(11)
        dimension = self.geometry.complement_dimension
        matrix = torch.randn(dimension, dimension, dtype=torch.float64)
        matrix = 0.5 * (matrix + matrix.T)
        z = torch.randn(dimension, dtype=torch.float64)
        z = (z / torch.linalg.vector_norm(z)).requires_grad_(True)
        value = z @ matrix @ z
        value.backward()
        gradient = sphere_tangent_projection(z.detach(), z.grad.detach())
        direction = torch.randn_like(z)
        direction = sphere_tangent_projection(z.detach(), direction)
        direction = direction / torch.linalg.vector_norm(direction)
        predicted = float(torch.dot(gradient, direction))
        errors = []
        for step in (1e-2, 5e-3, 2.5e-3):
            moved = sphere_retraction(z.detach(), step * direction)
            actual = float(moved @ matrix @ moved - value.detach())
            errors.append(abs(actual - step * predicted))
        self.assertLess(errors[1], errors[0] * 0.35)
        self.assertLess(errors[2], errors[1] * 0.35)


if __name__ == "__main__":
    unittest.main()

