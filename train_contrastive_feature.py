#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import torch
from random import randint
from gaussian_renderer import render_contrastive_feature
import sys
from PIL import Image
from scene import Scene, GaussianModel, FeatureGaussianModel
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB
from utils.part_mask_utils import (
    apply_exclusive_positive_shell_dilation,
    classify_background_like_masks,
    compute_background_like_mask_thresholds,
    make_identity_gates,
    select_part_scale_masks,
)
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
from torch import nn
import torch.nn.functional as F
import pytorch3d.ops


import time

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


VIS_PALETTE = torch.tensor([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [0, 130, 200],
    [245, 130, 48],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [210, 245, 60],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [170, 110, 40],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
], dtype=torch.uint8)

def _save_png(array, path):
    Image.fromarray(array).save(path)

def _fit_kmeans(features, num_clusters, max_samples, num_iters=12):
    if features.shape[0] == 0:
        return None

    features = torch.nn.functional.normalize(features.float(), dim=-1)
    sample_count = min(features.shape[0], max_samples)
    perm = torch.randperm(features.shape[0])[:sample_count]
    fit_features = features[perm]

    init_perm = torch.randperm(sample_count)[:num_clusters]
    centroids = fit_features[init_perm].clone()

    for _ in range(num_iters):
        distances = torch.cdist(fit_features, centroids)
        labels = torch.argmin(distances, dim=1)

        new_centroids = centroids.clone()
        for cluster_idx in range(num_clusters):
            cluster_mask = labels == cluster_idx
            if cluster_mask.any():
                new_centroids[cluster_idx] = fit_features[cluster_mask].mean(dim=0)
        centroids = torch.nn.functional.normalize(new_centroids, dim=-1)

    return centroids

def _cluster_feature_map(rendered_features, original_image, num_clusters, max_samples):
    _, height, width = rendered_features.shape
    flat_features = rendered_features.permute(1, 2, 0).reshape(-1, rendered_features.shape[0]).cpu()
    valid_mask = flat_features.norm(dim=1) > 1e-6

    if valid_mask.sum().item() < num_clusters:
        return None, None

    centroids = _fit_kmeans(flat_features[valid_mask], num_clusters, max_samples)
    if centroids is None:
        return None, None

    labels = torch.full((flat_features.shape[0],), -1, dtype=torch.long)
    valid_features = flat_features[valid_mask]
    chunk = 65536
    all_valid_labels = []
    for start in range(0, valid_features.shape[0], chunk):
        batch = torch.nn.functional.normalize(valid_features[start:start + chunk], dim=-1)
        distances = torch.cdist(batch, centroids)
        all_valid_labels.append(torch.argmin(distances, dim=1))
    labels[valid_mask] = torch.cat(all_valid_labels, dim=0)

    palette = VIS_PALETTE
    if num_clusters > len(palette):
        repeats = int(np.ceil(num_clusters / len(palette)))
        palette = palette.repeat((repeats, 1))
    palette = palette[:num_clusters]

    cluster_rgb = torch.zeros((flat_features.shape[0], 3), dtype=torch.uint8)
    cluster_rgb[valid_mask] = palette[labels[valid_mask]]
    cluster_rgb = cluster_rgb.view(height, width, 3).numpy()

    original_rgb = (
        torch.clamp(original_image.detach().cpu(), 0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .byte()
        .numpy()
    )
    overlay = (0.35 * original_rgb.astype(np.float32) + 0.65 * cluster_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return cluster_rgb, overlay

def _compute_embedding_projection(features, max_samples):
    if features.shape[0] == 0:
        return None, None

    sample_count = min(features.shape[0], max_samples)
    perm = torch.randperm(features.shape[0])[:sample_count]
    sample_features = features[perm].float()
    mean = sample_features.mean(dim=0, keepdim=True)
    centered = sample_features - mean

    try:
        _, _, v = torch.pca_lowrank(centered, q=min(3, centered.shape[1]))
    except RuntimeError:
        return None, None

    basis = v[:, :3]
    if basis.shape[1] < 3:
        pad = torch.zeros((basis.shape[0], 3 - basis.shape[1]), dtype=basis.dtype)
        basis = torch.cat([basis, pad], dim=1)
    return mean, basis


def _embedding_feature_map(rendered_features, original_image, max_samples):
    _, height, width = rendered_features.shape
    flat_features = rendered_features.permute(1, 2, 0).reshape(-1, rendered_features.shape[0]).cpu()
    valid_mask = flat_features.norm(dim=1) > 1e-6

    if valid_mask.sum().item() < 3:
        return None, None

    mean, basis = _compute_embedding_projection(flat_features[valid_mask], max_samples)
    if mean is None or basis is None:
        return None, None

    projected = (flat_features[valid_mask].float() - mean) @ basis
    proj_min = projected.min(dim=0).values
    proj_max = projected.max(dim=0).values
    denom = torch.clamp(proj_max - proj_min, min=1e-6)
    projected = (projected - proj_min) / denom

    embedding_rgb = torch.zeros((flat_features.shape[0], 3), dtype=torch.uint8)
    embedding_rgb[valid_mask] = torch.clamp(projected * 255.0, 0.0, 255.0).byte()
    embedding_rgb = embedding_rgb.view(height, width, 3).numpy()

    original_rgb = (
        torch.clamp(original_image.detach().cpu(), 0.0, 1.0)
        .permute(1, 2, 0)
        .mul(255.0)
        .byte()
        .numpy()
    )
    overlay = (0.35 * original_rgb.astype(np.float32) + 0.65 * embedding_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)
    return embedding_rgb, overlay


def save_training_visualizations(scene, feature_gaussians, pipe, background, iteration, view_indices, viz_mode, num_clusters, max_samples, smooth_k, model_path):
    if not view_indices:
        return

    output_dir = os.path.join(model_path, 'training_viz', f'iteration_{iteration:06d}')
    os.makedirs(output_dir, exist_ok=True)

    train_cameras = scene.getTrainCameras()
    valid_indices = [idx for idx in view_indices if 0 <= idx < len(train_cameras)]
    if not valid_indices:
        return

    with torch.no_grad():
        for view_idx in valid_indices:
            view = train_cameras[view_idx]
            view.feature_height, view.feature_width = view.image_height, view.image_width
            render_pkg_feat = render_contrastive_feature(
                view,
                feature_gaussians,
                pipe,
                background,
                norm_point_features=True,
                smooth_type='traditional',
                smooth_K=smooth_k,
            )
            rendered_features = torch.nn.functional.interpolate(
                render_pkg_feat['render'].unsqueeze(0),
                view.original_image.shape[-2:],
                mode='bilinear',
                align_corners=False,
            ).squeeze(0)

            if viz_mode == "cluster":
                viz_rgb, overlay = _cluster_feature_map(
                    rendered_features,
                    view.original_image,
                    num_clusters=num_clusters,
                    max_samples=max_samples,
                )
                suffix = "clusters"
            else:
                viz_rgb, overlay = _embedding_feature_map(
                    rendered_features,
                    view.original_image,
                    max_samples=max_samples,
                )
                suffix = "embedding"

            if viz_rgb is None:
                continue

            original_rgb = (
                torch.clamp(view.original_image.detach().cpu(), 0.0, 1.0)
                .permute(1, 2, 0)
                .mul(255.0)
                .byte()
                .numpy()
            )

            prefix = os.path.join(output_dir, f'view_{view_idx:03d}')
            _save_png(original_rgb, prefix + '_rgb.png')
            _save_png(viz_rgb, prefix + f'_{suffix}.png')
            _save_png(overlay, prefix + f'_{suffix}_overlay.png')
# Borrowed from GARField but modified
def get_quantile_func(scales: torch.Tensor, distribution="normal"):
    """
    Use empirical quantiles to normalize scales without depending on sklearn.
    """
    reference = torch.sort(scales.flatten().detach().cpu()).values
    denom = max(reference.numel() - 1, 1)

    def quantile_transformer_func(values):
        original_shape = values.shape
        flat_values = values.reshape(-1).detach().cpu()
        ranks = torch.searchsorted(reference, flat_values, right=True).float()
        quantiles = torch.clamp(ranks / denom, 0.0, 1.0)

        if distribution == "normal":
            eps = 1e-4
            quantiles = torch.clamp(quantiles, eps, 1.0 - eps)
            transformed = torch.erfinv(2 * quantiles - 1) * np.sqrt(2.0)
        else:
            transformed = quantiles

        return transformed.reshape(original_shape).to(values.device)

    return quantile_transformer_func

def _safe_mean(values, reference_tensor):
    if values.numel() == 0:
        return reference_tensor.new_tensor(0.0)
    return values.mean()


def _compute_scale_gates(scale_values, scale_gate, feature_dim, scale_aware_dim, fixed_scale_gate=None):
    scale_values = scale_values.reshape(-1)
    if scale_aware_dim <= 0 or scale_aware_dim >= feature_dim:
        return scale_gate(scale_values.unsqueeze(-1))

    int_scales = ((1 - scale_values) * scale_aware_dim).long()
    int_scales = torch.clamp(int_scales, 0, scale_aware_dim)
    return fixed_scale_gate[int_scales].detach()


def _compute_effective_scale_gates(scale_values, scale_gate, feature_dim, scale_aware_dim, supervision_mode="multiscale", fixed_scale_gate=None):
    if supervision_mode == "single_scale_band":
        return make_identity_gates(scale_values, feature_dim)
    return _compute_scale_gates(scale_values, scale_gate, feature_dim, scale_aware_dim, fixed_scale_gate=fixed_scale_gate)


def _compute_per_pixel_weight(sam_masks, sampled_ray):
    sam_masks = sam_masks.float()
    per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:, None, None]
    per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim=0) / (sam_masks.sum(dim=0) + 1e-9)
    per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray]
    pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
    ptp_max_size = pixel_to_pixel_mask_size.max() if pixel_to_pixel_mask_size.numel() > 0 else sam_masks.new_tensor(1.0)
    pixel_to_pixel_mask_size = pixel_to_pixel_mask_size.clone()
    pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
    per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
    span = per_pixel_weight.max() - per_pixel_weight.min() if per_pixel_weight.numel() > 0 else sam_masks.new_tensor(0.0)
    if per_pixel_weight.numel() == 0 or float(span.item()) <= 1e-6:
        return torch.ones_like(per_pixel_weight)
    return (per_pixel_weight - per_pixel_weight.min()) / span * 9.0 + 1.0


def _prepare_multiscale_supervision_batch(viewpoint_cam, opt, q_trans, upper_bound_scale, background_like_thresholds=None):
    original_masks = viewpoint_cam.original_masks.cuda().float()
    sam_masks_shape = original_masks.shape[-2:]
    mask_scales = viewpoint_cam.mask_scales.cuda().float()
    dilation_kernel = int(getattr(opt, "multiscale_positive_dilate_kernel", 0))
    if mask_scales.numel() == 0 or original_masks.shape[0] == 0:
        return None, {"skip_reason": "empty_masks", "selected_count": 0, "mean_area_ratio": 0.0, "mean_iou_overlap": 0.0}

    mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
    original_masks = original_masks[sort_indices, :, :]
    positive_masks = original_masks
    background_like_count = 0
    dilated_mask_count = 0
    added_shell_pixel_ratio = 0.0
    if dilation_kernel >= 3 and background_like_thresholds is not None:
        background_like_flags, background_stats = classify_background_like_masks(
            original_masks.bool(),
            mask_scales,
            background_like_thresholds,
        )
        positive_masks, dilation_stats = apply_exclusive_positive_shell_dilation(
            original_masks.bool(),
            ~background_like_flags,
            dilation_kernel,
        )
        positive_masks = positive_masks.float()
        background_like_count = int(background_stats["background_like_count"])
        dilated_mask_count = int(dilation_stats["dilated_mask_count"])
        added_shell_pixel_ratio = float(dilation_stats["added_shell_pixel_ratio"])

    num_sampled_scales = min(8, len(mask_scales))
    sampled_scale_index = torch.randperm(len(mask_scales), device=mask_scales.device)[:num_sampled_scales]
    tmp = torch.zeros(num_sampled_scales + 2, device=mask_scales.device, dtype=torch.long)
    tmp[1:len(sampled_scale_index) + 1] = sampled_scale_index
    tmp[-1] = len(mask_scales) - 1
    tmp[0] = -1
    sampled_scale_index = tmp.long()
    sampled_scales = mask_scales[sampled_scale_index]

    smaller_scales = mask_scales[mask_scales < upper_bound_scale]
    second_big_scale = smaller_scales.max() if smaller_scales.numel() > 0 else mask_scales.max()

    ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (positive_masks.shape[-1] * positive_masks.shape[-2])
    sampled_ray = torch.rand(positive_masks.shape[-2], positive_masks.shape[-1], device=positive_masks.device) < ray_sample_rate
    non_mask_region = positive_masks.sum(dim=0) == 0
    sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)
    if not sampled_ray.any():
        return None, {
            "skip_reason": "no_sampled_rays",
            "selected_count": int(original_masks.shape[0]),
            "mean_area_ratio": float((positive_masks.sum(dim=(-1, -2)).mean() / max(positive_masks.shape[-1] * positive_masks.shape[-2], 1)).item()),
            "mean_iou_overlap": 0.0,
            "background_like_count": background_like_count,
            "dilated_mask_count": dilated_mask_count,
            "added_shell_pixel_ratio": added_shell_pixel_ratio,
        }

    per_pixel_weight = _compute_per_pixel_weight(positive_masks, sampled_ray)
    sam_masks_sampled_ray = positive_masks[:, sampled_ray]
    sampled_boundary = None
    if opt.boundary_negative_weight > 0:
        sampled_boundary = _build_boundary_band(original_masks, opt.boundary_band_kernel)[sampled_ray]

    gt_corrs = []
    sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1, device=mask_scales.device)[0]
    for idx, si in enumerate(sampled_scale_index):
        upper_bound = sampled_scales[idx] >= upper_bound_scale
        si_int = int(si.item())

        if si_int != len(mask_scales) - 1 and not upper_bound:
            sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si_int + 1]) * torch.rand(1, device=mask_scales.device)[0]
        elif upper_bound:
            sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1, device=mask_scales.device)[0]
        else:
            sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1, device=mask_scales.device)[0]

        if not upper_bound:
            gt_vec = torch.zeros_like(sam_masks_sampled_ray)
            gt_vec[:si_int + 1, :] = sam_masks_sampled_ray[:si_int + 1, :]
            for j in range(si_int, -1, -1):
                gt_vec[j, :] = torch.logical_and(torch.logical_not(gt_vec[j + 1:, :].any(dim=0)), gt_vec[j, :])
            gt_vec[si_int + 1:, :] = sam_masks_sampled_ray[si_int + 1:, :]
        else:
            gt_vec = sam_masks_sampled_ray

        gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
        gt_corr[gt_corr != 0] = 1
        gt_corrs.append(gt_corr)

    gt_corrs = torch.stack(gt_corrs, dim=0)
    sampled_scales = q_trans(sampled_scales).reshape(-1)

    stats = {
        "skip_reason": None,
        "selected_count": int(original_masks.shape[0]),
        "mean_area_ratio": float((positive_masks.sum(dim=(-1, -2)).mean() / max(positive_masks.shape[-1] * positive_masks.shape[-2], 1)).item()),
        "mean_iou_overlap": 0.0,
        "background_like_count": background_like_count,
        "dilated_mask_count": dilated_mask_count,
        "added_shell_pixel_ratio": added_shell_pixel_ratio,
    }
    return {
        "sam_masks_shape": sam_masks_shape,
        "sampled_ray": sampled_ray,
        "per_pixel_weight": per_pixel_weight,
        "sampled_boundary": sampled_boundary,
        "gt_corrs": gt_corrs,
        "sampled_scales": sampled_scales,
        "stats": stats,
    }, stats


def _prepare_single_scale_supervision_batch(viewpoint_cam, opt, q_trans):
    sam_masks = viewpoint_cam.original_masks.cuda().bool()
    sam_masks_shape = sam_masks.shape[-2:]
    mask_scales = viewpoint_cam.mask_scales.cuda().float()
    if mask_scales.numel() == 0 or sam_masks.shape[0] == 0:
        return None, {"skip_reason": "empty_masks", "selected_count": 0, "mean_area_ratio": 0.0, "mean_iou_overlap": 0.0}

    normalized_mask_scales = q_trans(mask_scales).reshape(-1)
    selected_masks, _, _, stats = select_part_scale_masks(
        sam_masks,
        normalized_mask_scales,
        opt.mask_scale_target,
        opt.mask_scale_tolerance,
        opt.mask_min_area,
        opt.mask_max_area_ratio,
        opt.mask_max_iou_overlap,
        opt.mask_boundary_erode_kernel,
    )
    if selected_masks.shape[0] < int(max(opt.min_valid_masks_per_view, 1)):
        stats = dict(stats)
        stats["skip_reason"] = stats.get("skip_reason") or "too_few_valid_masks"
        return None, stats

    ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (sam_masks_shape[-1] * sam_masks_shape[-2])
    sampled_ray = torch.rand(sam_masks_shape[-2], sam_masks_shape[-1], device=selected_masks.device) < ray_sample_rate
    non_mask_region = ~selected_masks.any(dim=0)
    sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)
    if not sampled_ray.any():
        stats = dict(stats)
        stats["skip_reason"] = "no_sampled_rays"
        return None, stats

    selected_masks_float = selected_masks.float()
    per_pixel_weight = _compute_per_pixel_weight(selected_masks_float, sampled_ray)
    sampled_masks = selected_masks_float[:, sampled_ray]
    gt_corr = torch.einsum('nh,nj->hj', sampled_masks, sampled_masks)
    gt_corr[gt_corr != 0] = 1

    sampled_boundary = None
    if opt.boundary_negative_weight > 0 and selected_masks.shape[0] >= int(max(opt.min_valid_masks_per_view, 1)):
        sampled_boundary = _build_boundary_band(selected_masks_float, opt.boundary_band_kernel)[sampled_ray]

    return {
        "sam_masks_shape": sam_masks_shape,
        "sampled_ray": sampled_ray,
        "per_pixel_weight": per_pixel_weight,
        "sampled_boundary": sampled_boundary,
        "gt_corrs": gt_corr.unsqueeze(0),
        "sampled_scales": torch.tensor([float(opt.mask_scale_target)], device=selected_masks.device, dtype=torch.float32),
        "stats": stats,
    }, stats


def _prepare_supervision_batch(viewpoint_cam, opt, q_trans, upper_bound_scale, background_like_thresholds=None):
    if opt.supervision_mode == "single_scale_band":
        return _prepare_single_scale_supervision_batch(viewpoint_cam, opt, q_trans)
    return _prepare_multiscale_supervision_batch(viewpoint_cam, opt, q_trans, upper_bound_scale, background_like_thresholds=background_like_thresholds)


def _build_boundary_band(sam_masks, kernel_size):
    if kernel_size <= 1:
        return sam_masks.any(dim=0)

    pad = kernel_size // 2
    mask_float = sam_masks.float().unsqueeze(1)
    dilated = F.max_pool2d(mask_float, kernel_size=kernel_size, stride=1, padding=pad)
    eroded = -F.max_pool2d(-mask_float, kernel_size=kernel_size, stride=1, padding=pad)
    boundary = (dilated - eroded) > 1e-6
    return boundary.any(dim=0).squeeze(0)


def _compute_graph_laplacian_loss(feature_gaussians, scale_values, scale_gate, feature_dim, scale_aware_dim, graph_k, anchor_samples, num_scales, visibility_filter=None, fixed_scale_gate=None, scene_sh0_rgb=None, spatial_weight=1.0, sh0_color_weight=0.0, sh0_color_sigma=0.25, supervision_mode="multiscale"):
    if graph_k <= 1 or anchor_samples <= 0 or num_scales <= 0:
        return feature_gaussians.get_xyz.new_tensor(0.0)

    neighbor_idx = feature_gaussians.get_feature_neighbor_idx(graph_k)
    if neighbor_idx is None or neighbor_idx.numel() == 0:
        return feature_gaussians.get_xyz.new_tensor(0.0)

    num_points = feature_gaussians.get_xyz.shape[0]
    if visibility_filter is not None and visibility_filter.any():
        candidate_indices = torch.nonzero(visibility_filter, as_tuple=False).squeeze(-1)
    else:
        candidate_indices = torch.arange(num_points, device=feature_gaussians.get_xyz.device)

    if candidate_indices.numel() == 0:
        return feature_gaussians.get_xyz.new_tensor(0.0)

    use_sh0_color = float(sh0_color_weight) > 0.0
    if use_sh0_color:
        if scene_sh0_rgb is None:
            raise RuntimeError("SH0 color weighting is enabled for the graph Laplacian, but no scene SH0 RGB was provided.")
        if scene_sh0_rgb.shape[0] != num_points:
            raise RuntimeError(
                f"Scene Gaussian count ({scene_sh0_rgb.shape[0]}) does not match feature Gaussian count ({num_points}) for SH0 color weighting."
            )

    anchor_count = min(int(anchor_samples), int(candidate_indices.numel()))
    anchor_perm = torch.randperm(candidate_indices.numel(), device=candidate_indices.device)[:anchor_count]
    anchor_indices = candidate_indices[anchor_perm]
    neighbor_indices = neighbor_idx[anchor_indices]

    raw_point_features = feature_gaussians.get_point_features
    anchor_base = F.normalize(raw_point_features[anchor_indices], dim=-1, p=2)
    neighbor_base = F.normalize(raw_point_features[neighbor_indices], dim=-1, p=2)

    anchor_xyz = feature_gaussians.get_xyz[anchor_indices].detach()
    neighbor_xyz = feature_gaussians.get_xyz[neighbor_indices].detach()
    dist2 = torch.sum((anchor_xyz.unsqueeze(1) - neighbor_xyz) ** 2, dim=-1)
    base_weight = torch.ones_like(dist2)
    if float(spatial_weight) > 0.0:
        local_k = min(4, dist2.shape[1])
        local_scale = torch.sqrt(torch.clamp(dist2[:, :local_k].mean(dim=1, keepdim=True), min=1e-6))
        spatial_denom = torch.clamp(2.0 * local_scale.pow(2), min=1e-6)
        base_weight = torch.exp(-float(spatial_weight) * dist2 / spatial_denom)
    if use_sh0_color:
        anchor_sh0_rgb = scene_sh0_rgb[anchor_indices].detach()
        neighbor_sh0_rgb = scene_sh0_rgb[neighbor_indices].detach()
        color_dist2 = torch.sum((anchor_sh0_rgb.unsqueeze(1) - neighbor_sh0_rgb) ** 2, dim=-1)
        color_denom = max(2.0 * float(sh0_color_sigma) * float(sh0_color_sigma), 1e-6)
        color_weight = torch.exp(-float(sh0_color_weight) * color_dist2 / color_denom)
        base_weight = base_weight * color_weight

    unique_scales = torch.unique(scale_values.detach().reshape(-1))
    if unique_scales.numel() == 0:
        return feature_gaussians.get_xyz.new_tensor(0.0)
    if unique_scales.numel() > num_scales:
        grid = torch.linspace(0, unique_scales.numel() - 1, steps=num_scales, device=unique_scales.device)
        unique_scales = unique_scales[grid.round().long()]

    laplacian_gates = _compute_effective_scale_gates(
        unique_scales,
        scale_gate,
        feature_dim,
        scale_aware_dim,
        supervision_mode=supervision_mode,
        fixed_scale_gate=fixed_scale_gate,
    )
    losses = []
    for gate in laplacian_gates:
        anchor_feature = F.normalize(anchor_base * gate.unsqueeze(0), dim=-1, p=2)
        neighbor_feature = F.normalize(neighbor_base * gate.view(1, 1, -1), dim=-1, p=2)
        affinity_weight = torch.clamp(((anchor_feature.unsqueeze(1) * neighbor_feature).sum(dim=-1).detach() + 1.0) * 0.5, min=0.0)
        weight = base_weight * affinity_weight
        weight = weight / torch.clamp(weight.sum(dim=1, keepdim=True), min=1e-6)
        diff2 = torch.sum((anchor_feature.unsqueeze(1) - neighbor_feature) ** 2, dim=-1)
        losses.append((weight * diff2).sum(dim=1).mean())

    return torch.stack(losses).mean() if losses else feature_gaussians.get_xyz.new_tensor(0.0)


def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from, viz_interval=0, viz_view_idx=None, viz_mode="embedding", viz_num_clusters=8, viz_max_samples=5000):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    print("Graph Laplacian weight:", opt.graph_laplacian_weight)
    print("Graph spatial weight:", opt.graph_spatial_weight)
    print("Graph SH0 color weight:", opt.graph_sh0_color_weight)
    print("Graph SH0 color sigma:", opt.graph_sh0_color_sigma)
    print("Boundary negative weight:", opt.boundary_negative_weight)
    print("Supervision mode:", opt.supervision_mode)
    print("Mask scale target:", opt.mask_scale_target)
    print("Mask scale tolerance:", opt.mask_scale_tolerance)
    print("Multiscale positive dilate kernel:", opt.multiscale_positive_dilate_kernel)
    assert opt.supervision_mode in {"multiscale", "single_scale_band"}
    if not (opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0):
        raise RuntimeError(
            "Contrastive feature training requires either --num_sampled_rays > 0 "
            "or --ray_sample_rate > 0. "
            f"Got num_sampled_rays={opt.num_sampled_rays}, ray_sample_rate={opt.ray_sample_rate}. "
            "For example, pass --num_sampled_rays 1000."
        )

    dataset.need_features = False
    dataset.need_masks = True
    dataset.feature_dim = getattr(dataset, "feature_dim", 32)
    feature_dim = dataset.feature_dim
    if feature_dim != 32:
        raise RuntimeError(
            f"Contrastive feature rendering currently requires feature_dim == 32, but got {feature_dim}."
        )

    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)
    print("Contrastive renderer backend:", scene.gaussian_backend)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    scene_sh0_rgb = None
    if opt.graph_sh0_color_weight > 0:
        scene_sh0_rgb = SH2RGB(scene.gaussians.get_features[:, 0, :].detach()).clamp(0.0, 1.0)
        if scene_sh0_rgb.shape[0] != feature_gaussians.get_xyz.shape[0]:
            raise RuntimeError(
                f"Scene Gaussian count ({scene_sh0_rgb.shape[0]}) does not match feature Gaussian count ({feature_gaussians.get_xyz.shape[0]}) for SH0 color weighting."
            )

    # 30030
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, feature_dim, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate = scale_gate.cuda()
    if opt.supervision_mode == "single_scale_band":
        with torch.no_grad():
            scale_gate[0].weight.zero_()
            scale_gate[0].bias.fill_(12.0)
        scale_gate.eval()
    else:
        scale_gate.train()
        param_group = {'params': scale_gate.parameters(), 'lr': opt.feature_lr, 'name': 'f'}
        feature_gaussians.optimizer.add_param_group(param_group)

    smooth_weights = None

    del gaussians
    torch.cuda.empty_cache()

    background = torch.ones([dataset.feature_dim], dtype=torch.float32, device="cuda") if dataset.white_background else torch.zeros([dataset.feature_dim], dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)
    
    first_iter = 0
    viewpoint_stack = None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    print("Preparing Quantile Transform...")
    # gather scales
    all_scales = [cam.mask_scales for cam in scene.getTrainCameras() if getattr(cam, "mask_scales", None) is not None and cam.mask_scales.numel() > 0]
    if len(all_scales) == 0:
        raise RuntimeError("No SAM mask scales were found on the training cameras.")
    all_scales = torch.cat(all_scales)

    upper_bound_scale = all_scales.max().item()
    # upper_bound_scale = np.percentile(all_scales.detach().cpu().numpy(), 75)

    # all_scales = []
    # for cam in scene.getTrainCameras():
    #     cam.mask_scales = torch.clamp(cam.mask_scales, 0, upper_bound_scale)
    #     all_scales.append(cam.mask_scales)
    # all_scales = torch.cat(all_scales)

    scale_aware_dim = opt.scale_aware_dim
    fixed_scale_gate = None
    background_like_thresholds = None
    dilation_enabled = opt.supervision_mode == "multiscale" and int(getattr(opt, "multiscale_positive_dilate_kernel", 0)) >= 3
    if dilation_enabled:
        background_like_thresholds = compute_background_like_mask_thresholds(scene.getTrainCameras())
        if background_like_thresholds is None:
            print("Background-like mask thresholds: unavailable (no valid masks); positive dilation disabled.")
        else:
            print(
                "Background-like mask thresholds:",
                {
                    "scale_q90": round(background_like_thresholds["scale_q90"], 4),
                    "area_q90": round(background_like_thresholds["area_q90"], 4),
                    "area_q95": round(background_like_thresholds["area_q95"], 4),
                    "border_cov_q90": round(background_like_thresholds["border_cov_q90"], 4),
                    "border_cov_gate": round(background_like_thresholds["border_cov_gate"], 4),
                    "area_gate": round(background_like_thresholds["area_gate"], 4),
                    "area_touch_gate": round(background_like_thresholds["area_touch_gate"], 4),
                },
            )
    else:
        print("Background-like mask thresholds: dilation disabled.")

    if scale_aware_dim <= 0 or scale_aware_dim >= feature_dim:
        print("Using adaptive scale gate.")
        q_trans = get_quantile_func(all_scales, "uniform")
    else:
        q_trans = get_quantile_func(all_scales, "uniform")
        fixed_scale_gate = torch.tensor([[1 for _ in range(feature_dim - scale_aware_dim + i)] + [0 for _ in range(scale_aware_dim - i)] for i in range(scale_aware_dim + 1)], device="cuda", dtype=torch.float32)

    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        
        if iteration < -1:
            viewpoint_cam = viewpoint_stack[0]
        else:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        with torch.no_grad():
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width
            supervision_batch, supervision_stats = _prepare_supervision_batch(
                viewpoint_cam,
                opt,
                q_trans,
                upper_bound_scale,
                background_like_thresholds=background_like_thresholds,
            )
            if supervision_batch is None:
                viewpoint_cam.release_auxiliary_data()
                continue
            sam_masks_shape = supervision_batch["sam_masks_shape"]
            sampled_ray = supervision_batch["sampled_ray"]
            per_pixel_weight = supervision_batch["per_pixel_weight"]
            sampled_boundary = supervision_batch["sampled_boundary"]
            gt_corrs = supervision_batch["gt_corrs"]
            sampled_scales = supervision_batch["sampled_scales"]

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features = render_pkg_feat["render"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), sam_masks_shape, mode='bilinear').squeeze(0)

        gates = _compute_effective_scale_gates(
            sampled_scales,
            scale_gate,
            feature_dim,
            scale_aware_dim,
            supervision_mode=opt.supervision_mode,
            fixed_scale_gate=fixed_scale_gate if scale_aware_dim > 0 and scale_aware_dim < feature_dim else None,
        )

        # First gather sampled rays to avoid materializing an N x C x H x W tensor.
        sampled_rendered_features = rendered_features[:, sampled_ray]
        sampled_feature_with_scale = sampled_rendered_features.unsqueeze(0) * gates.unsqueeze(-1)

        scale_conditioned_features_sam = sampled_feature_with_scale.permute([0,2,1])

        scale_conditioned_features_sam = torch.nn.functional.normalize(scale_conditioned_features_sam, dim=-1, p=2)
        corr = torch.einsum('nhc,njc->nhj', scale_conditioned_features_sam, scale_conditioned_features_sam)

        diag_mask = torch.eye(corr.shape[1], dtype=bool, device=corr.device)

        sum_0 = gt_corrs.sum(dim = 0)
        consistent_negative = sum_0 == 0
        consistent_positive = sum_0 == len(gt_corrs)
        inconsistent = torch.logical_not(torch.logical_or(consistent_negative, consistent_positive))
        inconsistent_num = inconsistent.count_nonzero()
        sampled_num = inconsistent_num / 2

        rand_num = torch.rand_like(sum_0)

        positive_count = torch.clamp(consistent_positive.count_nonzero(), min=1)
        negative_count = torch.clamp(consistent_negative.count_nonzero(), min=1)
        sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / positive_count)

        sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / negative_count)

        sampled_mask_positive = torch.logical_or(
            torch.logical_or(
                sampled_positive, torch.any(torch.logical_and(corr < 0.75, gt_corrs == 1), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_positive = torch.logical_and(sampled_mask_positive, ~diag_mask)
        sampled_mask_positive = torch.triu(sampled_mask_positive, diagonal=0)
        sampled_mask_positive = sampled_mask_positive.bool()

        sampled_mask_negative = torch.logical_or(
            torch.logical_or(
                sampled_negative, torch.any(torch.logical_and(corr > 0.5, gt_corrs == 0), dim = 0)
            ), 
            inconsistent
        )
        sampled_mask_negative = torch.logical_and(sampled_mask_negative, ~diag_mask)
        sampled_mask_negative = torch.triu(sampled_mask_negative, diagonal=0)
        sampled_mask_negative = sampled_mask_negative.bool()

        per_pixel_weight = per_pixel_weight.unsqueeze(0)
        positive_values = -per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]
        negative_values = per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])
        positive_term = _safe_mean(positive_values, corr)
        negative_term = _safe_mean(negative_values, corr)

        boundary_term = corr.new_tensor(0.0)
        if opt.boundary_negative_weight > 0 and sampled_boundary is not None and sampled_boundary.any():
            boundary_pair_mask = torch.logical_or(sampled_boundary.unsqueeze(0), sampled_boundary.unsqueeze(1))
            boundary_pair_mask = torch.logical_and(boundary_pair_mask, sampled_mask_negative)
            boundary_pair_mask = torch.logical_and(boundary_pair_mask, torch.triu(torch.ones_like(boundary_pair_mask, dtype=torch.bool), diagonal=1))
            boundary_values = per_pixel_weight[:, boundary_pair_mask] * torch.relu(corr[:, boundary_pair_mask] - opt.boundary_negative_margin)
            boundary_term = _safe_mean(boundary_values, corr)

        laplacian_term = corr.new_tensor(0.0)
        if opt.graph_laplacian_weight > 0:
            laplacian_k = opt.smooth_K if opt.graph_laplacian_k <= 0 else opt.graph_laplacian_k
            laplacian_term = _compute_graph_laplacian_loss(
                feature_gaussians,
                sampled_scales.detach(),
                scale_gate,
                feature_dim,
                scale_aware_dim,
                laplacian_k,
                opt.graph_laplacian_samples,
                opt.graph_laplacian_scales,
                visibility_filter=render_pkg_feat.get("visibility_filter"),
                fixed_scale_gate=fixed_scale_gate if scale_aware_dim > 0 and scale_aware_dim < feature_dim else None,
                scene_sh0_rgb=scene_sh0_rgb,
                spatial_weight=opt.graph_spatial_weight,
                sh0_color_weight=opt.graph_sh0_color_weight,
                sh0_color_sigma=opt.graph_sh0_color_sigma,
                supervision_mode=opt.supervision_mode,
            )

        loss = positive_term + negative_term + opt.boundary_negative_weight * boundary_term + opt.graph_laplacian_weight * laplacian_term + opt.rfn * rendered_feature_norm_reg

        with torch.no_grad():
            cosine_pos = _safe_mean(corr[gt_corrs == 1], corr)
            cosine_neg = _safe_mean(corr[gt_corrs == 0], corr)

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)
        viewpoint_cam.release_auxiliary_data()

        del gt_corrs, rendered_features, corr, sampled_feature_with_scale, sampled_rendered_features
        torch.cuda.empty_cache()

        iter_end.record()

        if iteration % 10 == 0:
            postfix = {
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
            }
            if opt.graph_laplacian_weight > 0:
                postfix["Lap"] = f"{laplacian_term.item():.{3}f}"
            if opt.boundary_negative_weight > 0:
                postfix["Bnd"] = f"{boundary_term.item():.{3}f}"
            if opt.supervision_mode == "single_scale_band":
                postfix["Masks"] = str(int(supervision_stats.get("selected_count", 0)))
                postfix["Area"] = f"{supervision_stats.get('mean_area_ratio', 0.0):.{3}f}"
                postfix["Ov"] = f"{supervision_stats.get('mean_iou_overlap', 0.0):.{3}f}"
            elif dilation_enabled:
                postfix["Bg"] = str(int(supervision_stats.get("background_like_count", 0)))
                postfix["Dil"] = str(int(supervision_stats.get("dilated_mask_count", 0)))
                postfix["Shell"] = f"{supervision_stats.get('added_shell_pixel_ratio', 0.0):.{3}f}"
            progress_bar.set_postfix(postfix)
            progress_bar.update(10)

        if viz_interval > 0 and viz_view_idx and (iteration % viz_interval == 0 or iteration == first_iter or iteration == opt.iterations):
            save_training_visualizations(
                scene,
                feature_gaussians,
                pipe,
                background,
                iteration,
                viz_view_idx,
                viz_mode,
                viz_num_clusters,
                viz_max_samples,
                opt.smooth_K,
                scene.feature_model_path,
            )

    
    scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = opt.smooth_K)
    feature_iteration_dir = os.path.join(scene.feature_model_path, "point_cloud", f"iteration_{iteration}")
    os.makedirs(feature_iteration_dir, exist_ok=True)
    torch.save(scale_gate.state_dict(), os.path.join(feature_iteration_dir, "scale_gate.pt"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
    args.model_path = os.path.abspath(args.model_path)
    exp_name = getattr(args, "exp_name", "")
    if getattr(args, "feature_model_path", ""):
        args.feature_model_path = os.path.abspath(args.feature_model_path)
    else:
        if exp_name:
            exp_name = os.path.normpath(exp_name)
            if os.path.isabs(exp_name) or exp_name.startswith(".."):
                raise ValueError(f"exp_name must be a relative subdirectory under model_path, got: {args.exp_name}")
            args.exp_name = exp_name
            args.feature_model_path = os.path.join(args.model_path, args.exp_name)
        else:
            args.exp_name = ""
            args.feature_model_path = args.model_path
    args.feature_model_path = os.path.abspath(args.feature_model_path)

    # Set up output folders
    print("Scene root: {}".format(args.model_path))
    if getattr(args, "exp_name", ""):
        print("Experiment name: {}".format(args.exp_name))
    print("Feature output root: {}".format(args.feature_model_path))
    os.makedirs(args.model_path, exist_ok = True)
    os.makedirs(args.feature_model_path, exist_ok = True)

    with open(os.path.join(args.feature_model_path, "feature_cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))
    with open(os.path.join(args.model_path, "feature_model_path.txt"), "w", encoding="utf-8") as handle:
        handle.write(args.feature_model_path + "\n")

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.feature_model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser, sentinel=True)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=np.random.randint(10000, 20000))
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--target', default='contrastive_feature', const='contrastive_feature', nargs='?', choices=['scene', 'seg', 'feature', 'coarse_seg_everything', 'contrastive_feature'])
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--viz_interval", type=int, default=0)
    parser.add_argument("--viz_view_idx", nargs='+', type=int, default=[])
    parser.add_argument("--viz_mode", type=str, default="embedding", choices=["embedding", "cluster"])
    parser.add_argument("--viz_num_clusters", type=int, default=8)
    parser.add_argument("--viz_max_samples", type=int, default=5000)
    
    # args = parser.parse_args(sys.argv[1:])
    args = get_combined_args(parser, target_cfg_file = 'cfg_args')
    args.save_iterations.append(args.iterations)
    prepare_output_and_logger(args)

    print("Optimizing scene root " + args.model_path)
    print("Saving feature outputs to " + args.feature_model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        args.iteration,
        args.save_iterations,
        args.checkpoint_iterations,
        args.debug_from,
        args.viz_interval,
        args.viz_view_idx,
        args.viz_mode,
        args.viz_num_clusters,
        args.viz_max_samples,
    )

    # All done
    print("\nTraining complete.")


"""
CUDA_VISIBLE_DEVICES=0 python train_contrastive_feature.py \
    -s /root/node1/data2/diyscene/mipnerf360/360_v2/garden \
    -m /root/node1/data2/diyscene/3dgs/fitted_gs/mipnerf360/360_v2/garden \
    --iterations 10000 --num_sampled_rays 1000 \
    --viz_interval 1000 --viz_view_idx 0 --viz_mode embedding --viz_max_samples 5000
"""
