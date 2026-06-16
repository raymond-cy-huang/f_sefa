"""Explicit nonlinear feature maps with numerical pre-image inversion."""

import numpy as np
from numpy.polynomial import chebyshev
from numpy.polynomial import hermite
from numpy.polynomial import laguerre
from numpy.polynomial import legendre


def make_mixing_matrix(latent_dim, num_projections=None, seed=0):
    """Builds a deterministic dense mixing matrix for feature pre-images."""

    num_projections = num_projections or latent_dim
    rng = np.random.RandomState(seed)
    gaussian = rng.randn(latent_dim, latent_dim)
    q, _ = np.linalg.qr(gaussian)
    if num_projections <= latent_dim:
        return q[:num_projections].astype(np.float64)

    extra = rng.randn(num_projections - latent_dim, latent_dim)
    extra /= np.maximum(np.linalg.norm(extra, axis=1, keepdims=True), 1e-12)
    return np.vstack([q, extra]).astype(np.float64)


def _coeff(order):
    value = np.zeros(order + 1, dtype=np.float64)
    value[order] = 1.0
    return value


def basis_values_and_derivatives(x, basis_name, orders):
    """Evaluates explicit basis values and derivatives at x."""

    basis_name = basis_name.lower()
    values = []
    derivatives = []
    for order in orders:
        coeff = _coeff(order)
        if basis_name == "legendre":
            values.append(legendre.legval(x, coeff))
            derivatives.append(legendre.legval(x, legendre.legder(coeff)))
        elif basis_name == "chebyshev":
            values.append(chebyshev.chebval(x, coeff))
            derivatives.append(chebyshev.chebval(x, chebyshev.chebder(coeff)))
        elif basis_name == "hermite":
            values.append(hermite.hermval(x, coeff))
            derivatives.append(hermite.hermval(x, hermite.hermder(coeff)))
        elif basis_name == "laguerre":
            values.append(laguerre.lagval(x, coeff))
            derivatives.append(laguerre.lagval(x, laguerre.lagder(coeff)))
        elif basis_name == "fourier":
            if order == 0:
                values.append(np.ones_like(x))
                derivatives.append(np.zeros_like(x))
            else:
                values.append(np.sin(np.pi * order * x))
                derivatives.append(np.pi * order * np.cos(np.pi * order * x))
                values.append(np.cos(np.pi * order * x))
                derivatives.append(-np.pi * order * np.sin(np.pi * order * x))
        else:
            raise ValueError(f"Unsupported basis `{basis_name}`.")
    return np.stack(values, axis=-1), np.stack(derivatives, axis=-1)


class MixedBasisPreimageMap:
    """Overcomplete explicit feature map Phi(z) = basis(R z / scale)."""

    def __init__(self, latent_dim, basis_name="legendre", degree=4,
                 input_scale=4.0, num_projections=None, seed=0,
                 include_constant=False):
        self.latent_dim = int(latent_dim)
        self.basis_name = basis_name.lower()
        self.degree = int(degree)
        self.input_scale = float(input_scale)
        first_order = 0 if include_constant else 1
        self.orders = list(range(first_order, self.degree + 1))
        if not self.orders:
            raise ValueError("At least one basis order is required.")
        self.mixing = make_mixing_matrix(
            self.latent_dim,
            num_projections=num_projections,
            seed=seed,
        )

    def encode_vector(self, vector):
        """Maps one latent vector to high-dimensional explicit features."""

        vector = np.asarray(vector, dtype=np.float64).reshape(self.latent_dim)
        projected = (self.mixing @ vector) / self.input_scale
        features, _ = basis_values_and_derivatives(
            projected,
            self.basis_name,
            self.orders,
        )
        return features.reshape(-1)

    def encode(self, latent):
        """Maps the last dimension of a latent array to explicit features."""

        latent = np.asarray(latent, dtype=np.float64)
        flat = latent.reshape(-1, self.latent_dim)
        encoded = np.stack([self.encode_vector(row) for row in flat], axis=0)
        return encoded.reshape(latent.shape[:-1] + (encoded.shape[-1],))

    def residual_and_jacobian(self, vector, target):
        """Returns Phi(vector)-target and its dense Jacobian."""

        vector = np.asarray(vector, dtype=np.float64).reshape(self.latent_dim)
        target = np.asarray(target, dtype=np.float64).reshape(-1)
        projected = (self.mixing @ vector) / self.input_scale
        features, derivatives = basis_values_and_derivatives(
            projected,
            self.basis_name,
            self.orders,
        )
        residual = features.reshape(-1) - target
        rows = []
        for proj_idx in range(self.mixing.shape[0]):
            for basis_idx in range(derivatives.shape[1]):
                rows.append((derivatives[proj_idx, basis_idx] / self.input_scale) *
                            self.mixing[proj_idx])
        return residual, np.vstack(rows)

    def inverse_vector(self, target, max_iterations=25, tolerance=1e-10,
                       step_clip=1.0, initial=None):
        """Solves min_z ||Phi(z)-target||^2 by Gauss-Newton/SVD lstsq."""

        if initial is None:
            current = np.zeros(self.latent_dim, dtype=np.float64)
        else:
            current = np.asarray(initial, dtype=np.float64).reshape(self.latent_dim).copy()

        last_residual_norm = None
        iterations = 0
        for iterations in range(1, max_iterations + 1):
            residual, jacobian = self.residual_and_jacobian(current, target)
            last_residual_norm = float(np.linalg.norm(residual))
            delta, _, _, _ = np.linalg.lstsq(jacobian, -residual, rcond=None)
            delta_norm = float(np.linalg.norm(delta))
            if step_clip is not None and delta_norm > step_clip:
                delta = delta * (step_clip / max(delta_norm, 1e-12))
                delta_norm = float(np.linalg.norm(delta))
            current = current + delta
            if delta_norm < tolerance:
                break

        final_residual, _ = self.residual_and_jacobian(current, target)
        return current, {
            "iterations": iterations,
            "initial_residual_norm": last_residual_norm,
            "final_residual_norm": float(np.linalg.norm(final_residual)),
        }

    def inverse(self, high_dim, max_iterations=25, tolerance=1e-10,
                step_clip=1.0, initial=None):
        """Inverts feature arrays slice-by-slice along the last dimension."""

        high_dim = np.asarray(high_dim, dtype=np.float64)
        flat_targets = high_dim.reshape(-1, high_dim.shape[-1])
        if initial is None:
            flat_initial = [None] * len(flat_targets)
        else:
            flat_initial = np.asarray(initial, dtype=np.float64).reshape(-1, self.latent_dim)

        vectors = []
        infos = []
        for idx, target in enumerate(flat_targets):
            init = None if initial is None else flat_initial[idx]
            vector, info = self.inverse_vector(
                target,
                max_iterations=max_iterations,
                tolerance=tolerance,
                step_clip=step_clip,
                initial=init,
            )
            vectors.append(vector)
            infos.append(info)
        return np.stack(vectors, axis=0).reshape(high_dim.shape[:-1] + (self.latent_dim,)), infos
