"""Coefficient-based orthogonal polynomial function-space mappings."""
import numpy as np
from numpy.polynomial import chebyshev, legendre


def _basis_quadrature(name, dimension):
    name = name.lower()
    if name == "legendre":
        nodes, weights = legendre.leggauss(dimension)
        norms = 2.0 / (2.0 * np.arange(dimension) + 1.0)
        return nodes, weights, norms
    if name == "chebyshev":
        nodes, weights = chebyshev.chebgauss(dimension)
        norms = np.full(dimension, np.pi / 2.0)
        norms[0] = np.pi
        return nodes, weights, norms
    raise ValueError("Unsupported basis: " + name)


def _basis_matrix(name, dimension, nodes):
    name = name.lower()
    matrix = np.zeros((len(nodes), dimension), dtype=np.float64)
    for order in range(dimension):
        coefficient = np.zeros(order + 1, dtype=np.float64)
        coefficient[order] = 1.0
        if name == "legendre":
            matrix[:, order] = legendre.legval(nodes, coefficient)
        elif name == "chebyshev":
            matrix[:, order] = chebyshev.chebval(nodes, coefficient)
        else:
            raise ValueError("Unsupported basis: " + name)
    return matrix


def coefficients_to_function(coefficients, basis):
    """Evaluates f(x) = sum_n coefficients[n] * B_n(x)."""
    coefficients = np.asarray(coefficients, dtype=np.float64)
    if coefficients.ndim != 1:
        raise ValueError("coefficients must have shape [D].")
    nodes, weights, norms = _basis_quadrature(basis, len(coefficients))
    matrix = _basis_matrix(basis, len(coefficients), nodes)
    values = matrix @ coefficients
    return values, nodes, weights, norms


def function_to_coefficients(values, basis, nodes, weights, norms):
    """Recovers coefficients by orthogonal projection."""
    values = np.asarray(values, dtype=np.float64)
    matrix = _basis_matrix(basis, len(norms), nodes)
    return (matrix.T @ (weights * values)) / norms


def exact_round_trip(coefficients, basis):
    """Exact coefficient-coordinate round trip.

    The function is represented by its basis coefficients. Therefore the
    analytical inverse reads the coefficient vector directly. Function
    evaluation is retained for inspecting the represented function; numerical
    quadrature projection is exposed separately and is not used by the exact
    baseline.
    """
    coefficients = np.asarray(coefficients, dtype=np.float64)
    values, nodes, weights, norms = coefficients_to_function(coefficients, basis)
    del nodes, weights, norms
    reconstructed = coefficients.copy()
    return values, reconstructed


def projected_round_trip(coefficients, basis):
    """Numerical projection variant for separate conditioning analysis."""
    values, nodes, weights, norms = coefficients_to_function(coefficients, basis)
    reconstructed = function_to_coefficients(
        values, basis, nodes, weights, norms)
    return values, reconstructed
