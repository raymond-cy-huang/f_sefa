"""Numerical analysis helpers for raw orthogonal-basis Hilbert geometry."""

import math

import numpy as np

from experiment_utils.function_space_mapping import _basis_matrix
from experiment_utils.function_space_mapping import _basis_quadrature
from experiment_utils.function_space_mapping import coefficients_to_function
from experiment_utils.function_space_mapping import function_to_coefficients
from experiment_utils.metrics import latent_errors


def as_vector(value, dimension, name):
    vector = np.asarray(value, dtype=np.float64).reshape(-1)
    if vector.shape != (dimension,):
        raise ValueError(
            "{} must have shape ({},), got {}.".format(
                name, dimension, vector.shape))
    if not np.all(np.isfinite(vector)):
        raise ValueError("{} contains non-finite values.".format(name))
    return vector


def basis_geometry(basis, dimension):
    """Returns the exact Stage 01 quadrature, raw basis matrix, and diagonal G."""
    nodes, weights, norms = _basis_quadrature(basis, dimension)
    matrix = _basis_matrix(basis, dimension, nodes)
    gram = np.diag(norms)
    return {
        "nodes": nodes,
        "weights": weights,
        "norms": norms,
        "matrix": matrix,
        "gram": gram,
        "sqrt_gram": np.sqrt(norms),
    }


def gram_validation(geometry):
    matrix = geometry["matrix"]
    weights = geometry["weights"]
    gram = geometry["gram"]
    empirical = matrix.T @ (weights[:, None] * matrix)
    residual = empirical - gram
    off_diagonal = residual.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    return {
        "frobenius_error": float(np.linalg.norm(residual, ord="fro")),
        "relative_frobenius_error": float(
            np.linalg.norm(residual, ord="fro") /
            max(np.linalg.norm(gram, ord="fro"), np.finfo(np.float64).eps)),
        "max_absolute_error": float(np.max(np.abs(residual))),
        "max_off_diagonal_magnitude": float(
            np.max(np.abs(off_diagonal))),
    }


def hilbert_inner(left, right, norms):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(np.dot(left * norms, right))


def hilbert_norm(vector, norms):
    value = hilbert_inner(vector, vector, norms)
    if value < 0.0 and abs(value) < 1e-12:
        value = 0.0
    if value < 0.0:
        raise ValueError("Hilbert norm square is negative: {}".format(value))
    return float(math.sqrt(value))


def normalize_hilbert(vector, norms):
    norm = hilbert_norm(vector, norms)
    if norm <= np.finfo(np.float64).eps:
        raise ValueError("Cannot normalize a zero Hilbert-space direction.")
    return np.asarray(vector, dtype=np.float64) / norm, norm


def inverse_through_stage01(raw_coefficients, basis):
    """Applies c -> Bc -> G^-1 B^T W Bc using Stage 01 utilities."""
    values, nodes, weights, norms = coefficients_to_function(
        raw_coefficients, basis)
    reconstructed = function_to_coefficients(
        values, basis, nodes, weights, norms)
    return values, reconstructed


def direction_transport(w, direction, alpha, basis):
    direct, _, _, _ = coefficients_to_function(w + alpha * direction, basis)
    baseline, _, _, _ = coefficients_to_function(w, basis)
    transported, _, _, _ = coefficients_to_function(direction, basis)
    return latent_errors(direct, baseline + alpha * transported)


def orthonormal_coordinate_validation(left, right, norms):
    sqrt_gram = np.sqrt(norms)
    raw_inner = hilbert_inner(left, right, norms)
    orth_inner = float(np.dot(sqrt_gram * left, sqrt_gram * right))
    raw_square = hilbert_inner(left, left, norms)
    orth_square = float(np.dot(sqrt_gram * left, sqrt_gram * left))
    return {
        "raw_inner_product": raw_inner,
        "orthonormal_inner_product": orth_inner,
        "inner_product_absolute_error": abs(raw_inner - orth_inner),
        "raw_norm_square": raw_square,
        "orthonormal_norm_square": orth_square,
        "norm_square_absolute_error": abs(raw_square - orth_square),
    }


def image_equivalence_metrics(reference, candidate):
    reference = np.asarray(reference, dtype=np.uint8)
    candidate = np.asarray(candidate, dtype=np.uint8)
    difference = reference.astype(np.float64) - candidate.astype(np.float64)
    mse = float(np.mean(difference ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(difference)))
    psnr = float("inf") if mse == 0.0 else float(
        10.0 * np.log10((255.0 ** 2) / mse))

    from experiment_utils.metrics import image_errors
    ssim = image_errors(reference, candidate)["image_ssim"]
    return {
        "psnr": psnr,
        "ssim": float(ssim),
        "mae": mae,
        "rmse": rmse,
    }
