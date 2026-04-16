import torch


def resolve_active_mask(filtered_mask=None, active_mask=None):
    if active_mask is not None:
        return ~active_mask
    return filtered_mask


def resolve_gaussian_inputs(
    pc,
    scaling_modifier,
    pipe,
    filtered_mask=None,
    active_mask=None,
    xyz_override=None,
    rotation_override=None,
    opacity_override=None,
    prefer_2dgs_scales=False,
):
    filtered_mask = resolve_active_mask(filtered_mask=filtered_mask, active_mask=active_mask)

    means3D = pc.get_xyz if xyz_override is None else xyz_override
    opacity = pc.get_opacity if opacity_override is None else opacity_override
    if filtered_mask is not None:
        opacity = opacity.detach().clone()
        opacity[filtered_mask, :] = 0

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        if prefer_2dgs_scales:
            raise RuntimeError("The 2DGS contrastive backend does not support compute_cov3D_python.")
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        if prefer_2dgs_scales:
            source_scale_dims = getattr(pc, "source_scale_dims", scales.shape[1])
            if source_scale_dims != 2:
                raise RuntimeError(
                    f"2DGS contrastive rendering expected 2 source scale dimensions, but got {source_scale_dims}."
                )
            scales = scales[:, :2]
        rotations = pc.get_rotation if rotation_override is None else rotation_override

    return means3D, opacity, scales, rotations, cov3D_precomp


def make_screenspace_points(pc):
    screenspace_points = torch.zeros_like(
        pc.get_xyz,
        dtype=pc.get_xyz.dtype,
        requires_grad=True,
        device="cuda",
    ) + 0
    try:
        screenspace_points.retain_grad()
    except Exception:
        pass
    return screenspace_points


def build_point_feature_color_precomp(
    pc,
    norm_point_features=False,
    smooth_type=None,
    smooth_weights=None,
    smooth_K=16,
    point_feature_override=None,
):
    if point_feature_override is not None:
        colors_precomp = point_feature_override
    elif smooth_type is None:
        colors_precomp = pc.get_point_features
    elif smooth_type == "multi_res":
        colors_precomp = pc.get_multi_resolution_smoothed_point_features(smooth_weights=smooth_weights)
    elif smooth_type == "traditional":
        colors_precomp = pc.get_smoothed_point_features(K=smooth_K, dropout=0.5)
    else:
        raise ValueError(f"Unsupported smooth_type '{smooth_type}'")

    if norm_point_features:
        colors_precomp = colors_precomp / (colors_precomp.norm(dim=1, keepdim=True) + 1e-9)
    return colors_precomp.contiguous()


def infer_gaussian_backend(pc):
    backend = getattr(pc, "gaussian_backend", None)
    if backend in {"3dgs", "2dgs"}:
        return backend
    source_scale_dims = getattr(pc, "source_scale_dims", 3)
    if source_scale_dims == 2:
        return "2dgs"
    return "3dgs"
