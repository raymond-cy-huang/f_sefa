"""Stage 02: paired boundary crossings constructed entirely in function space.

After the initial Phi/Phi_dir calls, every endpoint, hull, SVD, barycentric,
ray, boundary, and distance operation is performed on raw-Legendre function
coefficients. Psi is called only after all function-space samples are fixed.
"""

import csv
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np


EXPERIMENT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(EXPERIMENT_DIR, "../.."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
if EXPERIMENT_DIR not in sys.path:
    sys.path.insert(0, EXPERIMENT_DIR)

from config import parse_args
from experiment_utils.function_space_mapping import _basis_matrix
from experiment_utils.function_space_mapping import _basis_quadrature
from experiment_utils.function_space_mapping import function_to_coefficients
from experiment_utils.metrics import save_image


SEMANTICS = (
    ("gender", 1),
    ("age", 3),
    ("pose", 7),
    ("glasses", 10),
    ("smile", 18),
)

CSV_FIELDS = [
    "sample_id", "ray_id", "sample_type", "semantic", "eigen_rank",
    "barycentric_coefficients", "alpha_or_beta_1", "alpha_or_beta_2",
    "alpha_or_beta_3", "alpha_or_beta_4", "alpha_or_beta_5",
    "hull_membership", "t_boundary", "hilbert_distance_to_boundary",
    "hilbert_distance_to_center", "inverse_residual_hilbert",
    "inverse_residual_l2", "affine_residual_l2", "minimum_barycentric",
    "probe_distance_h", "image_path",
]


@dataclass(frozen=True)
class FunctionGeometry:
    """Raw-basis coefficient coordinates for elements of the Stage 01 H."""

    basis: str
    dimension: int
    nodes: np.ndarray
    weights: np.ndarray
    gram_diagonal: np.ndarray
    evaluation_matrix: np.ndarray

    @classmethod
    def build(cls, basis, dimension):
        nodes, weights, norms = _basis_quadrature(basis, dimension)
        matrix = _basis_matrix(basis, dimension, nodes)
        return cls(basis, dimension, nodes, weights, norms, matrix)

    def _coordinate_vector(self, value, name):
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (self.dimension,):
            raise ValueError(
                "{} must have shape ({},), got {}.".format(
                    name, self.dimension, vector.shape
                )
            )
        if not np.all(np.isfinite(vector)):
            raise ValueError("{} contains non-finite values.".format(name))
        return vector.copy()

    def phi(self, latent_source):
        """Maps one source point to raw-basis coordinates in H."""
        coefficients = self._coordinate_vector(latent_source, "latent source")
        # Evaluation makes the Stage 01 representation explicit and validates
        # that these coefficients define a finite function at all quadrature nodes.
        values = self.evaluation_matrix @ coefficients
        if not np.all(np.isfinite(values)):
            raise ValueError("Phi produced non-finite function values.")
        return coefficients

    def phi_direction(self, latent_direction_source):
        """Maps a source vector through the linear Stage 01 direction map."""
        coefficients = self._coordinate_vector(
            latent_direction_source, "latent direction source"
        )
        values = self.evaluation_matrix @ coefficients
        if not np.all(np.isfinite(values)):
            raise ValueError("Phi_dir produced non-finite function values.")
        return coefficients

    def psi(self, function_coefficients):
        """Stage 01 projection inverse, used only for final decoding."""
        coefficients = self._coordinate_vector(
            function_coefficients, "function-space point"
        )
        values = self.evaluation_matrix @ coefficients
        return function_to_coefficients(
            values,
            self.basis,
            self.nodes,
            self.weights,
            self.gram_diagonal,
        )

    def hilbert_inner(self, left, right):
        left = self._coordinate_vector(left, "left function vector")
        right = self._coordinate_vector(right, "right function vector")
        return float(np.dot(left * self.gram_diagonal, right))

    def hilbert_norm(self, vector):
        square = self.hilbert_inner(vector, vector)
        if square < 0.0 and abs(square) < 1e-14:
            square = 0.0
        if square < 0.0:
            raise ValueError("Negative Hilbert norm square: {}".format(square))
        return float(math.sqrt(square))

    def hilbert_distance(self, first, second):
        return self.hilbert_norm(np.asarray(first) - np.asarray(second))


class AffineSimplex:
    """SVD-based affine coordinates and membership in function space."""

    def __init__(self, vertices, tolerance):
        self.vertices = np.asarray(vertices, dtype=np.float64)
        if self.vertices.ndim != 2 or self.vertices.shape[1] != 5:
            raise ValueError("vertices must have shape [function_dim, 5].")
        self.tolerance = float(tolerance)
        self.reference = self.vertices[:, 4]
        self.affine_matrix = self.vertices[:, :4] - self.reference[:, None]
        self.u, self.singular_values, self.vt = np.linalg.svd(
            self.affine_matrix, full_matrices=False
        )
        rank_tolerance = (
            np.finfo(np.float64).eps
            * max(self.affine_matrix.shape)
            * self.singular_values[0]
        )
        self.rank_tolerance = float(rank_tolerance)
        self.rank = int(np.sum(self.singular_values > rank_tolerance))
        if self.rank != 4:
            raise ValueError(
                "Function-space vertices have affine rank {}, expected 4.".format(
                    self.rank
                )
            )

    def barycentric(self, point):
        point = np.asarray(point, dtype=np.float64).reshape(-1)
        offset = point - self.reference
        coordinates = self.vt.T @ (
            (self.u.T @ offset) / self.singular_values
        )
        alpha = np.concatenate((coordinates, [1.0 - np.sum(coordinates)]))
        reconstructed = self.vertices @ alpha
        residual = float(np.linalg.norm(reconstructed - point))
        scale = max(1.0, float(np.linalg.norm(point)))
        inside = bool(
            residual <= self.tolerance * scale
            and abs(float(np.sum(alpha)) - 1.0) <= self.tolerance
            and float(np.min(alpha)) >= -self.tolerance
        )
        return alpha, residual, inside


def validate_config(config):
    if config.basis != "legendre":
        raise ValueError("This experiment requires the raw Legendre basis.")
    if config.latent_dimension != 512:
        raise ValueError("This experiment requires dimension 512.")
    if config.num_rays <= 0:
        raise ValueError("num_rays must be positive.")
    if not np.isfinite(config.lambda_0) or config.lambda_0 <= 0.0:
        raise ValueError("lambda_0 must be finite and positive.")
    if not np.isfinite(config.epsilon_h) or config.epsilon_h <= 0.0:
        raise ValueError("epsilon_h must be finite and positive.")
    if config.membership_tolerance <= 0.0:
        raise ValueError("membership_tolerance must be positive.")
    probe_distances = np.asarray(config.probe_distances_h, dtype=np.float64)
    if (
        probe_distances.ndim != 1
        or len(probe_distances) == 0
        or not np.all(np.isfinite(probe_distances))
        or np.any(probe_distances < 0.0)
        or np.any(np.diff(probe_distances) <= 0.0)
        or probe_distances[0] != 0.0
    ):
        raise ValueError(
            "probe_distances_h must be strictly increasing, finite, "
            "non-negative, and begin at 0.0."
        )


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_source_w(generator, config, torch):
    """Creates the sole latent point source used before entering H."""
    device = next(generator.parameters()).device
    z = torch.randn(1, generator.z_space_dim, device=device)
    with torch.no_grad():
        w = generator.mapping(z)["w"]
        w = generator.truncation(
            w,
            trunc_psi=config.trunc_psi,
            trunc_layers=config.trunc_layers,
        )
    if w.ndim == 3:
        w = w[:, 0, :]
    vector = w[0].detach().cpu().numpy().astype(np.float64)
    if vector.shape != (config.latent_dimension,):
        raise ValueError("Expected W source shape (512,), got {}.".format(vector.shape))
    return vector


def source_sefa_directions(generator, factorize_weight):
    """Obtains the five source directions before the Phi_dir boundary."""
    layers, boundaries, eigenvalues = factorize_weight(generator, "all")
    boundaries = np.real(boundaries).astype(np.float64)
    eigenvalues = np.real(eigenvalues).astype(np.float64)
    ranked_indices = np.argsort(eigenvalues)[::-1]
    result = []
    for semantic, rank in SEMANTICS:
        source_index = int(ranked_indices[rank])
        direction = boundaries[source_index].reshape(-1)
        if direction.shape != (512,):
            raise ValueError("Unexpected SeFa direction shape {}.".format(direction.shape))
        result.append({
            "semantic": semantic,
            "eigen_rank": rank,
            "source_index": source_index,
            "eigenvalue": float(eigenvalues[source_index]),
            "source_direction": direction,
        })
    return layers, result


def decoder_code(generator, latent):
    vector = np.asarray(latent, dtype=np.float32).reshape(-1)
    if vector.shape != (generator.w_space_dim,):
        raise ValueError("Unexpected inverse output shape: {}.".format(vector.shape))
    return np.tile(vector[None, :], (generator.num_layers, 1))[None, ...]


def decode(generator, inverse_output, torch, postprocess):
    device = next(generator.parameters()).device
    code = torch.from_numpy(decoder_code(generator, inverse_output)).float().to(device)
    with torch.no_grad():
        image = generator.synthesis(code)["image"]
    return postprocess(image)[0]


def ensure_output_directories(output_dir):
    directories = {
        "vertices": os.path.join(output_dir, "images", "vertices"),
        "inside": os.path.join(output_dir, "images", "inside"),
        "boundary": os.path.join(output_dir, "images", "boundary"),
        "outside": os.path.join(output_dir, "images", "outside"),
        "probe": os.path.join(output_dir, "images", "probe"),
        "comparisons": os.path.join(output_dir, "comparison_figures"),
    }
    os.makedirs(output_dir, exist_ok=True)
    for path in directories.values():
        os.makedirs(path, exist_ok=True)
    return directories


def make_labeled_grid(items, path, columns=5, thumb=192):
    from PIL import Image, ImageDraw

    rows = int(math.ceil(len(items) / float(columns)))
    label_height = 28
    canvas = Image.new("RGB", (columns * thumb, rows * (thumb + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, array) in enumerate(items):
        row, column = divmod(index, columns)
        x = column * thumb
        y = row * (thumb + label_height)
        image = Image.fromarray(np.asarray(array, dtype=np.uint8)).resize(
            (thumb, thumb), Image.BILINEAR
        )
        canvas.paste(image, (x, y + label_height))
        draw.text((x + 5, y + 6), str(label), fill="black")
    canvas.save(path)


def make_crossing_grid(original, crossing_images, path, thumb=160):
    from PIL import Image, ImageDraw

    headers = ("Original", "Inside", "Boundary", "Outside")
    label_width = 64
    header_height = 30
    canvas = Image.new(
        "RGB",
        (label_width + 4 * thumb, header_height + len(crossing_images) * thumb),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((label_width + column * thumb + 5, 7), header, fill="black")
    original_pil = Image.fromarray(original).resize((thumb, thumb), Image.BILINEAR)
    for ray_id, images in enumerate(crossing_images):
        top = header_height + ray_id * thumb
        draw.text((7, top + thumb // 2), "{:02d}".format(ray_id), fill="black")
        row_images = (original_pil,) + tuple(
            Image.fromarray(image).resize((thumb, thumb), Image.BILINEAR)
            for image in images
        )
        for column, image in enumerate(row_images):
            canvas.paste(image, (label_width + column * thumb, top))
    canvas.save(path)


def make_pair_figure(original, images, ray_id, path, thumb=256):
    from PIL import Image, ImageDraw

    labels = ("Original", "Inside", "Boundary", "Outside")
    label_height = 32
    canvas = Image.new("RGB", (4 * thumb, thumb + label_height), "white")
    draw = ImageDraw.Draw(canvas)
    for column, (label, array) in enumerate(zip(labels, (original,) + tuple(images))):
        draw.text((column * thumb + 7, 7), label, fill="black")
        image = Image.fromarray(array).resize((thumb, thumb), Image.BILINEAR)
        canvas.paste(image, (column * thumb, label_height))
    draw.text((4 * thumb - 60, 7), "ray {:02d}".format(ray_id), fill="black")
    canvas.save(path)


def make_probe_grid(original, probe_images_by_ray, distances, path, thumb=160):
    """Shows the same function-space ray at increasing distances past the hull."""
    from PIL import Image, ImageDraw

    headers = ("Original",) + tuple(
        "Boundary" if distance == 0.0 else "+{:g} H".format(distance)
        for distance in distances
    )
    label_width = 64
    header_height = 30
    columns = len(headers)
    canvas = Image.new(
        "RGB",
        (label_width + columns * thumb, header_height + len(probe_images_by_ray) * thumb),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, header in enumerate(headers):
        draw.text((label_width + column * thumb + 5, 7), header, fill="black")
    original_pil = Image.fromarray(original).resize((thumb, thumb), Image.BILINEAR)
    for ray_id, images in enumerate(probe_images_by_ray):
        top = header_height + ray_id * thumb
        draw.text((7, top + thumb // 2), "{:02d}".format(ray_id), fill="black")
        row_images = (original_pil,) + tuple(
            Image.fromarray(image).resize((thumb, thumb), Image.BILINEAR)
            for image in images
        )
        for column, image in enumerate(row_images):
            canvas.paste(image, (label_width + column * thumb, top))
    canvas.save(path)


def make_csv_row(
    sample_id,
    ray_id,
    sample_type,
    coefficients,
    membership,
    t_boundary,
    distance_boundary,
    distance_center,
    inverse_h,
    inverse_l2,
    affine_residual,
    image_path,
    probe_distance_h="",
    semantic="",
    eigen_rank="",
):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    return {
        "sample_id": sample_id,
        "ray_id": ray_id,
        "sample_type": sample_type,
        "semantic": semantic,
        "eigen_rank": eigen_rank,
        "barycentric_coefficients": json.dumps(coefficients.tolist()),
        "alpha_or_beta_1": float(coefficients[0]),
        "alpha_or_beta_2": float(coefficients[1]),
        "alpha_or_beta_3": float(coefficients[2]),
        "alpha_or_beta_4": float(coefficients[3]),
        "alpha_or_beta_5": float(coefficients[4]),
        "hull_membership": bool(membership),
        "t_boundary": t_boundary,
        "hilbert_distance_to_boundary": distance_boundary,
        "hilbert_distance_to_center": distance_center,
        "inverse_residual_hilbert": inverse_h,
        "inverse_residual_l2": inverse_l2,
        "affine_residual_l2": affine_residual,
        "minimum_barycentric": float(np.min(coefficients)),
        "probe_distance_h": probe_distance_h,
        "image_path": image_path,
    }


def main():
    config = parse_args()
    validate_config(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_id
    directories = ensure_output_directories(config.output_dir)

    import torch
    from utils import factorize_weight, get_gan_type, load_generator, postprocess

    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    generator = load_generator(config.model)
    if get_gan_type(generator) not in ("stylegan", "stylegan2", "psp_stylegan2"):
        raise ValueError("Expected a StyleGAN-family generator.")
    if int(generator.w_space_dim) != config.latent_dimension:
        raise ValueError("Generator W dimension does not match configuration.")

    # The only latent-source phase. Everything is mapped immediately into H.
    w0_source = sample_source_w(generator, config, torch)
    layers, direction_sources = source_sefa_directions(generator, factorize_weight)
    geometry = FunctionGeometry.build(config.basis, config.latent_dimension)
    c0_h = geometry.phi(w0_source)
    mapped_directions = []
    for item in direction_sources:
        mapped = dict(item)
        mapped["u_h"] = geometry.phi_direction(item["source_direction"])
        del mapped["source_direction"]
        mapped_directions.append(mapped)

    # No latent-space semantic/geometric operation occurs below this line.
    vertices_h = np.column_stack([
        c0_h + config.lambda_0 * item["u_h"]
        for item in mapped_directions
    ])
    simplex = AffineSimplex(vertices_h, config.membership_tolerance)
    q = np.full(5, 0.2, dtype=np.float64)
    center_h = vertices_h @ q

    rng = np.random.default_rng(config.seed)
    crossings = []
    for ray_id in range(config.num_rays):
        anchor_alpha = rng.dirichlet(np.ones(5, dtype=np.float64))
        v = anchor_alpha - q
        negative = v < -np.finfo(np.float64).eps
        if not np.any(negative):
            raise RuntimeError("Dirichlet ray unexpectedly has no negative component.")
        t_boundary = float(np.min(-q[negative] / v[negative]))
        boundary_alpha = q + t_boundary * v
        boundary_h = vertices_h @ boundary_alpha
        ray_h = vertices_h @ v
        ray_norm_h = geometry.hilbert_norm(ray_h)
        if ray_norm_h <= np.finfo(np.float64).eps:
            raise RuntimeError("Degenerate function-space sampling ray.")
        delta_t = config.epsilon_h / ray_norm_h
        if delta_t >= t_boundary:
            raise ValueError(
                "epsilon_h={} is too large for ray {} (center-boundary distance {})."
                .format(config.epsilon_h, ray_id, t_boundary * ray_norm_h)
            )
        inside_alpha = q + (t_boundary - delta_t) * v
        outside_alpha = q + (t_boundary + delta_t) * v
        inside_h = vertices_h @ inside_alpha
        outside_h = vertices_h @ outside_alpha

        samples = []
        for sample_type, coefficients, point_h in (
            ("inside", inside_alpha, inside_h),
            ("boundary", boundary_alpha, boundary_h),
            ("outside", outside_alpha, outside_h),
        ):
            recovered, affine_residual, membership = simplex.barycentric(point_h)
            expected = sample_type != "outside"
            if membership != expected:
                raise RuntimeError(
                    "Unexpected membership for ray {} {}: {} (min alpha {}, residual {})."
                    .format(ray_id, sample_type, membership, np.min(recovered), affine_residual)
                )
            samples.append({
                "sample_type": sample_type,
                "coefficients": coefficients,
                "recovered_coefficients": recovered,
                "point_h": point_h,
                "membership": membership,
                "affine_residual": affine_residual,
                "distance_boundary": geometry.hilbert_distance(point_h, boundary_h),
                "distance_center": geometry.hilbert_distance(point_h, center_h),
            })
        crossings.append({
            "ray_id": ray_id,
            "anchor_alpha": anchor_alpha,
            "t_boundary": t_boundary,
            "ray_norm_h": ray_norm_h,
            "samples": samples,
            "probes": [],
        })
        for probe_distance_h in config.probe_distances_h:
            probe_t = t_boundary + probe_distance_h / ray_norm_h
            probe_alpha = q + probe_t * v
            probe_h = vertices_h @ probe_alpha
            recovered, affine_residual, membership = simplex.barycentric(probe_h)
            expected_membership = probe_distance_h == 0.0
            if membership != expected_membership:
                raise RuntimeError(
                    "Unexpected probe membership for ray {} at distance {}: {}."
                    .format(ray_id, probe_distance_h, membership)
                )
            crossings[-1]["probes"].append({
                "distance_h": float(probe_distance_h),
                "coefficients": recovered,
                "point_h": probe_h,
                "membership": membership,
                "affine_residual": affine_residual,
                "distance_center": geometry.hilbert_distance(probe_h, center_h),
            })

    # Every function-space sample is now immutable. Psi is only a decode step.
    original_image = decode(generator, geometry.psi(c0_h), torch, postprocess)
    vertex_items = [("Original", original_image)]
    rows = []
    for vertex_id, item in enumerate(mapped_directions):
        point_h = vertices_h[:, vertex_id]
        inverse = geometry.psi(point_h)
        reconstructed_h = geometry.phi(inverse)
        inverse_h = geometry.hilbert_distance(point_h, reconstructed_h)
        inverse_l2 = float(np.linalg.norm(point_h - reconstructed_h))
        recovered, affine_residual, membership = simplex.barycentric(point_h)
        image = decode(generator, inverse, torch, postprocess)
        image_path = os.path.join(
            directories["vertices"], "vertex_{:02d}_{}.png".format(
                vertex_id, item["semantic"]
            )
        )
        save_image(image, image_path)
        vertex_items.append((item["semantic"], image))
        rows.append(make_csv_row(
            "vertex_{:02d}".format(vertex_id),
            "",
            "vertex",
            recovered,
            membership,
            "",
            "",
            geometry.hilbert_distance(point_h, center_h),
            inverse_h,
            inverse_l2,
            affine_residual,
            image_path,
            semantic=item["semantic"],
            eigen_rank=item["eigen_rank"],
        ))
    make_labeled_grid(
        vertex_items,
        os.path.join(config.output_dir, "vertices.png"),
        columns=3,
    )

    images_by_type = {"inside": [], "boundary": [], "outside": []}
    crossing_images = []
    probe_images_by_ray = []
    for crossing in crossings:
        ray_id = crossing["ray_id"]
        triplet = []
        for sample in crossing["samples"]:
            sample_type = sample["sample_type"]
            point_h = sample["point_h"]
            inverse = geometry.psi(point_h)
            reconstructed_h = geometry.phi(inverse)
            inverse_h = geometry.hilbert_distance(point_h, reconstructed_h)
            inverse_l2 = float(np.linalg.norm(point_h - reconstructed_h))
            image = decode(generator, inverse, torch, postprocess)
            image_path = os.path.join(
                directories[sample_type],
                "ray_{:02d}_{}.png".format(ray_id, sample_type),
            )
            save_image(image, image_path)
            images_by_type[sample_type].append(("ray {:02d}".format(ray_id), image))
            triplet.append(image)
            rows.append(make_csv_row(
                "ray_{:02d}_{}".format(ray_id, sample_type),
                ray_id,
                sample_type,
                sample["recovered_coefficients"],
                sample["membership"],
                crossing["t_boundary"],
                sample["distance_boundary"],
                sample["distance_center"],
                inverse_h,
                inverse_l2,
                sample["affine_residual"],
                image_path,
            ))
        crossing_images.append(tuple(triplet))
        make_pair_figure(
            original_image,
            triplet,
            ray_id,
            os.path.join(
                directories["comparisons"], "ray_{:02d}_comparison.png".format(ray_id)
            ),
        )
        ray_probe_images = []
        ray_probe_items = []
        for probe_id, probe in enumerate(crossing["probes"]):
            point_h = probe["point_h"]
            inverse = geometry.psi(point_h)
            reconstructed_h = geometry.phi(inverse)
            inverse_h = geometry.hilbert_distance(point_h, reconstructed_h)
            inverse_l2 = float(np.linalg.norm(point_h - reconstructed_h))
            image = decode(generator, inverse, torch, postprocess)
            distance_label = "{:g}".format(probe["distance_h"]).replace(".", "p")
            image_path = os.path.join(
                directories["probe"],
                "ray_{:02d}_distance_{}.png".format(ray_id, distance_label),
            )
            save_image(image, image_path)
            ray_probe_images.append(image)
            ray_probe_items.append((
                "boundary" if probe["distance_h"] == 0.0
                else "+{:g} H".format(probe["distance_h"]),
                image,
            ))
            rows.append(make_csv_row(
                "ray_{:02d}_probe_{:02d}".format(ray_id, probe_id),
                ray_id,
                "probe_boundary" if probe["distance_h"] == 0.0 else "probe_outside",
                probe["coefficients"],
                probe["membership"],
                crossing["t_boundary"],
                probe["distance_h"],
                probe["distance_center"],
                inverse_h,
                inverse_l2,
                probe["affine_residual"],
                image_path,
                probe_distance_h=probe["distance_h"],
            ))
        probe_images_by_ray.append(tuple(ray_probe_images))
        make_labeled_grid(
            ray_probe_items,
            os.path.join(
                directories["comparisons"], "ray_{:02d}_boundary_probe.png".format(ray_id)
            ),
            columns=len(config.probe_distances_h),
            thumb=160,
        )

    for sample_type in ("inside", "boundary", "outside"):
        make_labeled_grid(
            images_by_type[sample_type],
            os.path.join(config.output_dir, "paired_{}.png".format(sample_type)),
            columns=5,
        )
    make_crossing_grid(
        original_image,
        crossing_images,
        os.path.join(config.output_dir, "paired_crossings.png"),
    )
    make_probe_grid(
        original_image,
        probe_images_by_ray,
        config.probe_distances_h,
        os.path.join(config.output_dir, "boundary_probe.png"),
    )

    csv_path = os.path.join(config.output_dir, "sampling.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", config.model + ".pth")
    payload = {
        "experiment": "stage02-convex-hull",
        "interpretation": (
            "Paired crossings of the boundary of C_known^H, not discovery "
            "of the unknown true semantic-valid boundary."
        ),
        "config": config.to_dict(),
        "pipeline": (
            "latent sources -> Phi/Phi_dir -> function-space endpoints/hull/"
            "SVD/rays/Hilbert geometry -> Psi -> generator"
        ),
        "function_space": {
            "coordinate_system": "raw Legendre coefficients",
            "basis_orders": [0, config.latent_dimension - 1],
            "gram_definition": "G[n,n] = 2 / (2*n + 1)",
            "mapping": "Phi(w)(x) = sum_n w[n] P_n(x)",
            "direction_mapping": "Phi_dir(d)(x) = sum_n d[n] P_n(x)",
            "inverse": "Stage 01 weighted orthogonal quadrature projection",
            "all_geometry_in_function_space": True,
        },
        "simplex": {
            "affine_rank": simplex.rank,
            "singular_values": simplex.singular_values.tolist(),
            "rank_tolerance": simplex.rank_tolerance,
            "membership_tolerance": config.membership_tolerance,
            "center_barycentric": q.tolist(),
        },
        "semantics": [{
            "name": item["semantic"],
            "eigen_rank": item["eigen_rank"],
            "source_eigen_index": item["source_index"],
            "eigenvalue": item["eigenvalue"],
            "direction_hilbert_norm": geometry.hilbert_norm(item["u_h"]),
        } for item in mapped_directions],
        "generator": {
            "model": config.model,
            "gan_type": get_gan_type(generator),
            "checkpoint_path": checkpoint_path,
            "checkpoint_sha256": sha256(checkpoint_path),
            "factorized_layers": [int(layer) for layer in layers],
        },
        "outputs": {
            "vertices": os.path.join(config.output_dir, "vertices.png"),
            "paired_inside": os.path.join(config.output_dir, "paired_inside.png"),
            "paired_boundary": os.path.join(config.output_dir, "paired_boundary.png"),
            "paired_outside": os.path.join(config.output_dir, "paired_outside.png"),
            "paired_crossings": os.path.join(config.output_dir, "paired_crossings.png"),
            "boundary_probe": os.path.join(config.output_dir, "boundary_probe.png"),
            "sampling_csv": csv_path,
            "comparison_directory": directories["comparisons"],
        },
    }
    config_path = os.path.join(config.output_dir, "experiment_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)

    print("Saved vertices: {}".format(payload["outputs"]["vertices"]))
    print("Saved paired crossings: {}".format(payload["outputs"]["paired_crossings"]))
    print("Saved sampling log: {}".format(csv_path))
    print("Saved experiment config: {}".format(config_path))


if __name__ == "__main__":
    main()

