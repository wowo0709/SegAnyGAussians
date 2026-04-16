import numpy as np


VALID_GAUSSIAN_BACKENDS = {"auto", "3dgs", "2dgs"}


def validate_gaussian_backend(gaussian_backend):
    if gaussian_backend not in VALID_GAUSSIAN_BACKENDS:
        supported = ", ".join(sorted(VALID_GAUSSIAN_BACKENDS))
        raise ValueError(f"Unsupported gaussian backend '{gaussian_backend}'. Expected one of: {supported}.")
    return gaussian_backend


def detect_gaussian_backend_from_scale_dims(source_scale_dims):
    if source_scale_dims == 3:
        return "3dgs"
    if source_scale_dims == 2:
        return "2dgs"
    raise ValueError(
        f"Unsupported number of Gaussian scale dimensions: {source_scale_dims}. "
        "Only 2DGS (2 scales) and 3DGS (3 scales) checkpoints are supported."
    )


def resolve_gaussian_backend(gaussian_backend, source_scale_dims, checkpoint_path):
    gaussian_backend = validate_gaussian_backend(gaussian_backend)
    detected_backend = detect_gaussian_backend_from_scale_dims(source_scale_dims)
    if gaussian_backend == "auto":
        return detected_backend
    if gaussian_backend != detected_backend:
        raise ValueError(
            f"Requested gaussian backend '{gaussian_backend}' is incompatible with '{checkpoint_path}'. "
            f"The checkpoint stores {source_scale_dims} scale dimensions, which maps to '{detected_backend}'."
        )
    return gaussian_backend


def ensure_rasterizer_scale_dims(scales):
    source_scale_dims = scales.shape[1]
    if source_scale_dims == 3:
        return scales
    if source_scale_dims == 2:
        thin_log_scale = np.maximum(np.min(scales, axis=1, keepdims=True) - 8.0, -12.0)
        return np.concatenate([scales, thin_log_scale], axis=1)
    raise ValueError(f"Unsupported number of Gaussian scale dimensions: {source_scale_dims}")


def extract_scales_from_ply_element(ply_element):
    scale_names = [p.name for p in ply_element.properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
    scales = np.zeros((ply_element.count, len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(ply_element[attr_name])
    return scales, len(scale_names)


def export_scaling_for_ply(scale_array, source_scale_dims):
    if source_scale_dims not in (2, 3):
        raise ValueError(f"Unsupported source scale dimensions for export: {source_scale_dims}")
    if scale_array.shape[1] < source_scale_dims:
        raise ValueError(
            f"Cannot export {source_scale_dims} scale dimensions from tensor with shape {scale_array.shape}."
        )
    return scale_array[:, :source_scale_dims]
