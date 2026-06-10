"""Inverse methods for explicit latent basis mappings."""

import numpy as np

try:
    from scipy.sparse.linalg import cg as scipy_cg
except ImportError:
    scipy_cg = None


def available_inverse_methods():
    """Returns supported inverse method names."""

    return ["svd_pinv", "ridge", "qr_lstsq", "lstsq", "cg"]


def _as_rhs(high_dim):
    high_dim = np.asarray(high_dim, dtype=np.float64)
    flat = high_dim.reshape(-1, high_dim.shape[-1]).T
    return high_dim, flat


def _restore_shape(solution, high_dim, latent_dim):
    return solution.T.reshape(high_dim.shape[:-1] + (latent_dim,))


def _numpy_cg(matrix, rhs, tol, maxiter):
    """Small dense conjugate-gradient fallback for SPD systems."""

    maxiter = maxiter or matrix.shape[0] * 10
    x = np.zeros_like(rhs)
    residual = rhs - matrix @ x
    direction = residual.copy()
    rs_old = float(residual @ residual)
    if np.sqrt(rs_old) <= tol:
        return x, 0

    for iteration in range(maxiter):
        mat_dir = matrix @ direction
        denom = float(direction @ mat_dir)
        if abs(denom) < 1e-30:
            return x, iteration + 1
        alpha = rs_old / denom
        x = x + alpha * direction
        residual = residual - alpha * mat_dir
        rs_new = float(residual @ residual)
        if np.sqrt(rs_new) <= tol:
            return x, 0
        direction = residual + (rs_new / rs_old) * direction
        rs_old = rs_new
    return x, maxiter


def inverse_map(high_dim, matrix, method, ridge_lambda=1e-6, cg_tol=1e-10, cg_maxiter=None):
    """Recovers latent slices from high-dimensional mapped vectors."""

    method = method.lower()
    matrix = np.asarray(matrix, dtype=np.float64)
    high_dim, rhs = _as_rhs(high_dim)
    latent_dim = matrix.shape[1]

    if method == "svd_pinv":
        solution = np.linalg.pinv(matrix) @ rhs
    elif method == "ridge":
        gram = matrix.T @ matrix
        reg = ridge_lambda * np.eye(latent_dim, dtype=np.float64)
        solution = np.linalg.solve(gram + reg, matrix.T @ rhs)
    elif method == "qr_lstsq":
        q, r = np.linalg.qr(matrix, mode="reduced")
        solution = np.linalg.solve(r, q.T @ rhs)
    elif method == "lstsq":
        solution, _, _, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    elif method == "cg":
        gram = matrix.T @ matrix
        right = matrix.T @ rhs
        columns = []
        for col in range(right.shape[1]):
            if scipy_cg is not None:
                try:
                    x, info = scipy_cg(
                        gram,
                        right[:, col],
                        rtol=cg_tol,
                        atol=0.0,
                        maxiter=cg_maxiter,
                    )
                except TypeError:
                    x, info = scipy_cg(
                        gram,
                        right[:, col],
                        tol=cg_tol,
                        maxiter=cg_maxiter,
                    )
            else:
                x, info = _numpy_cg(gram, right[:, col], cg_tol, cg_maxiter)
            if info != 0:
                raise RuntimeError(f"CG did not converge for RHS {col}; info={info}.")
            columns.append(x)
        solution = np.stack(columns, axis=1)
    else:
        raise ValueError(f"Unsupported inverse method `{method}`.")

    return _restore_shape(solution, high_dim, latent_dim)
