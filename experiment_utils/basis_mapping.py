"""Explicit orthogonal-basis mappings for latent round-trip experiments."""

from dataclasses import dataclass

import numpy as np
from numpy.polynomial import chebyshev
from numpy.polynomial import hermite
from numpy.polynomial import laguerre
from numpy.polynomial import legendre


@dataclass(frozen=True)
class BasisMapping:
    """Container for a basis mapping matrix."""

    name: str
    matrix: np.ndarray
    grid: np.ndarray


def available_basis_names():
    """Returns supported basis names."""

    return ["legendre", "chebyshev", "hermite", "laguerre", "fourier"]


def _normalize_rows(values, eps=1e-12):
    values = np.asarray(values, dtype=np.float64)
    scale = np.sqrt(np.mean(values**2, axis=1, keepdims=True))
    return values / np.maximum(scale, eps)


def _polynomial_values(name, degree, grid):
    orders = range(degree + 1)
    coeffs = []
    for order in orders:
        coeff = np.zeros(order + 1, dtype=np.float64)
        coeff[order] = 1.0
        coeffs.append(coeff)
    if name == "legendre":
        values = [legendre.legval(grid, coeff) for coeff in coeffs]
    elif name == "chebyshev":
        values = [chebyshev.chebval(grid, coeff) for coeff in coeffs]
    elif name == "hermite":
        values = [hermite.hermval(grid, coeff) for coeff in coeffs]
    elif name == "laguerre":
        laguerre_grid = (grid + 1.0) * 2.5
        values = [laguerre.lagval(laguerre_grid, coeff) for coeff in coeffs]
    else:
        raise ValueError(f"Unsupported polynomial basis `{name}`.")
    return _normalize_rows(np.stack(values, axis=0))


def _fourier_values(degree, grid):
    values = [np.ones_like(grid)]
    for order in range(1, degree + 1):
        values.append(np.sin(np.pi * order * grid))
        values.append(np.cos(np.pi * order * grid))
    return _normalize_rows(np.stack(values, axis=0))


def build_basis_matrix(name, latent_dim, degree, dtype=np.float64):
    """Builds A for high_dim_z = A @ z.

    The matrix is an explicit stack of diagonal basis evaluations. For a
    latent vector z in R^D and K basis terms, A has shape [K * D, D] and maps
    z to [diag(phi_0)z, ..., diag(phi_K)z]. The constant term keeps A full
    column rank, while the remaining terms lift the vector into a higher
    dimensional function-feature space.
    """

    if latent_dim <= 0:
        raise ValueError("latent_dim must be positive.")
    if degree < 0:
        raise ValueError("degree must be non-negative.")

    name = name.lower()
    grid = np.linspace(-1.0, 1.0, latent_dim, dtype=np.float64)
    if name == "fourier":
        values = _fourier_values(degree, grid)
    elif name in {"legendre", "chebyshev", "hermite", "laguerre"}:
        values = _polynomial_values(name, degree, grid)
    else:
        raise ValueError(f"Unsupported basis `{name}`.")

    blocks = [np.diag(row) for row in values]
    matrix = np.vstack(blocks).astype(dtype, copy=False)
    return BasisMapping(name=name, matrix=matrix, grid=grid)


def map_latent_slices(latent, matrix):
    """Applies A @ z to the last dimension of a latent array."""

    latent = np.asarray(latent)
    if latent.ndim == 0:
        raise ValueError("latent must have at least one dimension.")
    flat = latent.reshape(-1, latent.shape[-1]).T
    mapped = matrix @ flat
    return mapped.T.reshape(latent.shape[:-1] + (matrix.shape[0],))


def _scale_latent(latent, scale):
    return np.asarray(latent, dtype=np.float64) / float(scale)


def _poly_feature_and_derivative(name, x, degree):
    values = []
    derivatives = []
    for order in range(degree + 1):
        coeff = np.zeros(order + 1, dtype=np.float64)
        coeff[order] = 1.0
        if name == "legendre":
            values.append(legendre.legval(x, coeff))
            derivatives.append(legendre.legval(x, legendre.legder(coeff)))
        elif name == "chebyshev":
            values.append(chebyshev.chebval(x, coeff))
            derivatives.append(chebyshev.chebval(x, chebyshev.chebder(coeff)))
        elif name == "hermite":
            values.append(hermite.hermval(x, coeff))
            derivatives.append(hermite.hermval(x, hermite.hermder(coeff)))
        elif name == "laguerre":
            values.append(laguerre.lagval(x, coeff))
            derivatives.append(laguerre.lagval(x, laguerre.lagder(coeff)))
        else:
            raise ValueError(f"Unsupported polynomial basis `{name}`.")
    return np.stack(values, axis=-1), np.stack(derivatives, axis=-1)


def nonlinear_feature_map(latent, name, degree, input_scale=4.0):
    """Maps z_i to explicit nonlinear basis features.

    For polynomial bases this returns [phi_0(x_i), ..., phi_degree(x_i)] for
    each latent coordinate x_i = z_i / input_scale. For Fourier it returns
    [1, sin(pi*x_i), cos(pi*x_i), ...]. This is nonlinear in z and is the
    finite sampled function-space feature map used by the round-trip script.
    """

    name = name.lower()
    x = _scale_latent(latent, input_scale)
    if name == "fourier":
        values = [np.ones_like(x)]
        for order in range(1, degree + 1):
            values.append(np.sin(np.pi * order * x))
            values.append(np.cos(np.pi * order * x))
        return np.stack(values, axis=-1)

    if name == "laguerre":
        x = x + 2.0
    values, _ = _poly_feature_and_derivative(name, x, degree)
    return values


def nonlinear_feature_jacobian_diag(latent, name, degree, input_scale=4.0):
    """Returns d feature_i,k / d latent_i for the nonlinear feature map."""

    name = name.lower()
    scale = float(input_scale)
    x = _scale_latent(latent, scale)
    if name == "fourier":
        derivatives = [np.zeros_like(x)]
        for order in range(1, degree + 1):
            derivatives.append((np.pi * order / scale) * np.cos(np.pi * order * x))
            derivatives.append(-(np.pi * order / scale) * np.sin(np.pi * order * x))
        return np.stack(derivatives, axis=-1)

    if name == "laguerre":
        x = x + 2.0
    _, derivatives = _poly_feature_and_derivative(name, x, degree)
    return derivatives / scale
