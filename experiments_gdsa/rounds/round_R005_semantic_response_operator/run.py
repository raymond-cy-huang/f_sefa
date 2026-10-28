"""Run exact 2D/16D controls for the GDSA semantic-response operator."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


ROUND_DIR = Path(__file__).resolve().parent
ROOT = ROUND_DIR.parents[2]
sys.path.insert(0, str(ROUND_DIR))

from operator_response import explicit_linear_operator  # noqa: E402


def load_r003():
    path = ROOT / "experiments_gdsa/rounds/round_R003_basis_validation/run.py"
    spec = importlib.util.spec_from_file_location("gdsa_r003_for_r005", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def random_orthogonal(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, r = np.linalg.qr(rng.normal(size=(dimension, dimension)))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1
    return q * signs


def bases(dimension: int) -> dict[str, np.ndarray]:
    result = {"identity": np.eye(dimension)}
    if dimension == 2:
        angle = np.pi / 7
        result["rotation_pi_over_7"] = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    else:
        r003 = load_r003()
        result["legendre"] = r003.legendre_transform(dimension)[0]
        result["chebyshev_first_kind"] = r003.chebyshev_transform(dimension)[0]
    for seed in range(5):
        result[f"random_orthogonal_seed_{seed}"] = random_orthogonal(dimension, seed)
    return result


def run_case(dimension: int, seed: int = 20261025) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed + dimension)
    output_dimension = dimension + 3
    jacobian = rng.normal(size=(output_dimension, dimension)) / np.sqrt(output_dimension)
    target = rng.normal(size=dimension)
    target /= np.linalg.norm(target)
    k_w = jacobian.T @ jacobian
    mu = 0.1 * max(float(np.trace(k_w) / dimension), 1e-12)
    w_operator = explicit_linear_operator(jacobian, np.eye(dimension))
    v_full = w_operator.solve_full(target, mu)
    y = rng.normal(size=output_dimension)
    rows = []
    for name, q in bases(dimension).items():
        operator = explicit_linear_operator(jacobian, q)
        k_h = operator.explicit_k()
        target_h = q @ target
        u_full = operator.solve_full(target_h, mu)
        v_from_h = q.T @ u_full
        diagonal = np.diag(k_h)
        u_diag = operator.solve_diag(target_h, diagonal, mu)
        v_diag = q.T @ u_diag
        probe = rng.normal(size=dimension)
        adjoint_error = abs(float(operator.apply_a(probe) @ y - probe @ operator.apply_adjoint(y)))
        off_diagonal = k_h - np.diag(diagonal)
        rows.append(
            {
                "dimension": dimension,
                "basis": name,
                "mu": mu,
                "orthogonality_spectral_error": float(np.linalg.norm(q.T @ q - np.eye(dimension), ord=2)),
                "similarity_frobenius_error": float(np.linalg.norm(k_h - q @ k_w @ q.T, ord="fro")),
                "adjoint_abs_error": adjoint_error,
                "minimum_eigenvalue_k": float(np.linalg.eigvalsh(k_h).min()),
                "full_w_direction_relative_error": float(np.linalg.norm(v_from_h - v_full) / np.linalg.norm(v_full)),
                "full_constraint_abs_error": abs(float(target @ v_from_h) - 1.0),
                "diagonal_off_fraction": float(np.linalg.norm(off_diagonal, ord="fro") / np.linalg.norm(k_h, ord="fro")),
                "diag_direction_relative_to_full": float(np.linalg.norm(v_diag - v_full) / np.linalg.norm(v_full)),
                "diag_true_response_energy": float(np.linalg.norm(jacobian @ v_diag) ** 2),
                "full_true_response_energy": float(np.linalg.norm(jacobian @ v_full) ** 2),
            }
        )
    return rows


def main() -> None:
    results_dir = ROUND_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    rows = run_case(2) + run_case(16)
    csv_path = results_dir / "toy_basis_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    maxima = {
        "orthogonality_spectral_error": max(float(row["orthogonality_spectral_error"]) for row in rows),
        "similarity_frobenius_error": max(float(row["similarity_frobenius_error"]) for row in rows),
        "adjoint_abs_error": max(float(row["adjoint_abs_error"]) for row in rows),
        "full_w_direction_relative_error": max(float(row["full_w_direction_relative_error"]) for row in rows),
        "full_constraint_abs_error": max(float(row["full_constraint_abs_error"]) for row in rows),
    }
    status = "PASS" if (
        maxima["orthogonality_spectral_error"] < 1e-10
        and maxima["similarity_frobenius_error"] < 1e-10
        and maxima["adjoint_abs_error"] < 1e-10
        and maxima["full_w_direction_relative_error"] < 1e-10
        and maxima["full_constraint_abs_error"] < 1e-10
    ) else "FAIL"
    summary = {
        "status": status,
        "seed": 20261025,
        "dimensions": [2, 16],
        "rows": len(rows),
        "maxima": maxima,
        "diagonal_off_fraction_range": [
            min(float(row["diagonal_off_fraction"]) for row in rows),
            max(float(row["diagonal_off_fraction"]) for row in rows),
        ],
        "interpretation": (
            "Exact full SRO is invariant under orthogonal coordinates; diagonal SRO is basis dependent. "
            "This is a mathematical control, not StyleGAN semantic evidence."
        ),
    }
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

