import math
import os
import sys

from ._common import (
    build_point_feature_color_precomp,
    make_screenspace_points,
    resolve_gaussian_inputs,
)

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from diff_surfel_rasterization_contrastive_f import GaussianRasterizationSettings
    from diff_surfel_rasterization_contrastive_f import GaussianRasterizer
except ModuleNotFoundError:
    _submodule_path = os.path.join(_ROOT_DIR, "submodules", "diff-surfel-rasterization_contrastive_f")
    if _submodule_path not in sys.path:
        sys.path.insert(0, _submodule_path)
    from diff_surfel_rasterization_contrastive_f import GaussianRasterizationSettings
    from diff_surfel_rasterization_contrastive_f import GaussianRasterizer


def render_contrastive_feature_2dgs(
    viewpoint_camera,
    pc,
    pipe,
    bg_color,
    scaling_modifier=1.0,
    norm_point_features=False,
    smooth_type=None,
    smooth_weights=None,
    smooth_K=16,
    filtered_mask=None,
    active_mask=None,
    xyz_override=None,
    rotation_override=None,
    opacity_override=None,
    point_feature_override=None,
):
    screenspace_points = make_screenspace_points(pc)

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.feature_height),
        image_width=int(viewpoint_camera.feature_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D, opacity, scales, rotations, cov3D_precomp = resolve_gaussian_inputs(
        pc,
        scaling_modifier,
        pipe,
        filtered_mask=filtered_mask,
        active_mask=active_mask,
        xyz_override=xyz_override,
        rotation_override=rotation_override,
        opacity_override=opacity_override,
        prefer_2dgs_scales=True,
    )
    colors_precomp = build_point_feature_color_precomp(
        pc,
        norm_point_features=norm_point_features,
        smooth_type=smooth_type,
        smooth_weights=smooth_weights,
        smooth_K=smooth_K,
        point_feature_override=point_feature_override,
    )

    rendered_image, radii, _depth = rasterizer(
        means3D=means3D,
        means2D=screenspace_points,
        shs=None,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp,
    )

    return {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter": radii > 0,
        "radii": radii,
    }
