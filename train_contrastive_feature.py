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
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args

import numpy as np


import torch
from torch import nn
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

def training(dataset, opt, pipe, iteration, saving_iterations, checkpoint_iterations, debug_from, viz_interval=0, viz_view_idx=None, viz_mode="embedding", viz_num_clusters=8, viz_max_samples=5000):
    print("RFN weight:", opt.rfn)
    print("Smooth K:", opt.smooth_K)
    print("Scale aware dim:", opt.scale_aware_dim)
    assert opt.ray_sample_rate > 0 or opt.num_sampled_rays > 0

    dataset.need_features = False
    dataset.need_masks = True

    gaussians = GaussianModel(dataset.sh_degree)

    feature_gaussians = FeatureGaussianModel(dataset.feature_dim)

    sample_rate = 0.2 if 'Replica' in dataset.source_path else 1.0
    scene = Scene(dataset, gaussians, feature_gaussians, load_iteration=iteration, shuffle=False, target='contrastive_feature', mode='train', sample_rate=sample_rate)

    feature_gaussians.change_to_segmentation_mode(opt, "contrastive_feature", fixed_feature=False)

    # 30030
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, 32, bias=True),
        torch.nn.Sigmoid()
    )
    scale_gate = scale_gate.cuda()
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
    all_scales = []
    for cam in scene.getTrainCameras():
        all_scales.append(cam.mask_scales)
    all_scales = torch.cat(all_scales)

    upper_bound_scale = all_scales.max().item()
    # upper_bound_scale = np.percentile(all_scales.detach().cpu().numpy(), 75)

    # all_scales = []
    # for cam in scene.getTrainCameras():
    #     cam.mask_scales = torch.clamp(cam.mask_scales, 0, upper_bound_scale)
    #     all_scales.append(cam.mask_scales)
    # all_scales = torch.cat(all_scales)

    scale_aware_dim = opt.scale_aware_dim

    if scale_aware_dim <= 0 or scale_aware_dim >= 32:
        print("Using adaptive scale gate.")
        q_trans = get_quantile_func(all_scales, "uniform")
    else:
        q_trans = get_quantile_func(all_scales, "uniform")
        fixed_scale_gate = torch.tensor([[1 for j in range(32 - scale_aware_dim + i)] + [0 for k in range(scale_aware_dim - i)] for i in range(scale_aware_dim+1)]).cuda()

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
            # N_mask, H, W
            sam_masks = viewpoint_cam.original_masks.cuda().float()
            sam_masks_shape = sam_masks.shape[-2:]
            viewpoint_cam.feature_height, viewpoint_cam.feature_width = viewpoint_cam.image_height, viewpoint_cam.image_width

            # N_mask
            mask_scales = viewpoint_cam.mask_scales.cuda()

            mask_scales, sort_indices = torch.sort(mask_scales, descending=True)
            sam_masks = sam_masks[sort_indices, :, :]

            num_sampled_scales = 8

            sampled_scale_index = torch.randperm(len(mask_scales))[:num_sampled_scales]

            tmp = torch.zeros(num_sampled_scales+2)

            tmp[1:len(sampled_scale_index)+1] = sampled_scale_index
            tmp[-1] = len(mask_scales) - 1
            tmp[0] = -1 # attach a bigger scale
            sampled_scale_index = tmp.long()
            

            sampled_scales = mask_scales[sampled_scale_index]

            second_big_scale = mask_scales[mask_scales < upper_bound_scale].max()

            ray_sample_rate = opt.ray_sample_rate if opt.ray_sample_rate > 0 else opt.num_sampled_rays / (sam_masks.shape[-1] * sam_masks.shape[-2])

            sampled_ray = torch.rand(sam_masks.shape[-2], sam_masks.shape[-1]).cuda() < ray_sample_rate
            non_mask_region = sam_masks.sum(dim = 0) == 0

            sampled_ray = torch.logical_and(sampled_ray, ~non_mask_region)

            # H W
            per_pixel_mask_size = sam_masks * sam_masks.sum(-1).sum(-1)[:,None,None]

            per_pixel_mean_mask_size = per_pixel_mask_size.sum(dim = 0) / (sam_masks.sum(dim = 0) + 1e-9)

            per_pixel_mean_mask_size = per_pixel_mean_mask_size[sampled_ray]


            pixel_to_pixel_mask_size = per_pixel_mean_mask_size.unsqueeze(0) * per_pixel_mean_mask_size.unsqueeze(1)
            ptp_max_size = pixel_to_pixel_mask_size.max()
            pixel_to_pixel_mask_size[pixel_to_pixel_mask_size == 0] = 1e10
            per_pixel_weight = torch.clamp(ptp_max_size / pixel_to_pixel_mask_size, 1.0, None)
            per_pixel_weight = (per_pixel_weight - per_pixel_weight.min()) / (per_pixel_weight.max() - per_pixel_weight.min()) * 9. + 1.
            
            sam_masks_sampled_ray = sam_masks[:, sampled_ray]

            gt_corrs = []

            sampled_scales[0] = upper_bound_scale + upper_bound_scale * torch.rand(1)[0]
            for idx, si in enumerate(sampled_scale_index):
                upper_bound = sampled_scales[idx] >= upper_bound_scale

                if si != len(mask_scales) - 1 and not upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - mask_scales[si+1]) * torch.rand(1)[0]
                elif upper_bound:
                    sampled_scales[idx] -= (sampled_scales[idx] - second_big_scale) * torch.rand(1)[0]
                else:
                    sampled_scales[idx] -= sampled_scales[idx] * torch.rand(1)[0]

                if not upper_bound:
                    gt_vec = torch.zeros_like(sam_masks_sampled_ray)
                    gt_vec[:si+1,:] = sam_masks_sampled_ray[:si+1,:]
                    for j in range(si, -1, -1):
                        gt_vec[j,:] = torch.logical_and(
                            torch.logical_not(gt_vec[j+1:,:].any(dim = 0)), gt_vec[j,:]
                        )
                    gt_vec[si+1:,:] = sam_masks_sampled_ray[si+1:,:]
                else:
                    gt_vec = sam_masks_sampled_ray

                gt_corr = torch.einsum('nh,nj->hj', gt_vec, gt_vec)
                gt_corr[gt_corr != 0] = 1
                gt_corrs.append(gt_corr)

            # N_scale S C_clip
            # gt_clip_features = torch.stack(gt_clip_features, dim = 0)
            # N_scale S S
            gt_corrs = torch.stack(gt_corrs, dim = 0)

            sampled_scales = q_trans(sampled_scales).squeeze()
            sampled_scales = sampled_scales.squeeze()

        render_pkg_feat = render_contrastive_feature(viewpoint_cam, feature_gaussians, pipe, background, norm_point_features=True, smooth_type = 'traditional', smooth_weights=torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_K = opt.smooth_K)
        rendered_features = render_pkg_feat["render"]

        rendered_feature_norm = rendered_features.norm(dim = 0, p=2).mean()
        rendered_feature_norm_reg = (1-rendered_feature_norm)**2

        rendered_features = torch.nn.functional.interpolate(rendered_features.unsqueeze(0), sam_masks_shape, mode='bilinear').squeeze(0)

        # N_sampled_scales 32
        if scale_aware_dim <= 0 or scale_aware_dim >= 32:
            gates = scale_gate(sampled_scales.unsqueeze(-1))
        else:
            int_sampled_scales = ((1 - sampled_scales.squeeze()) * scale_aware_dim).long()
            gates = fixed_scale_gate[int_sampled_scales].detach()

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

        sampled_positive = torch.logical_and(consistent_positive, rand_num < sampled_num / consistent_positive.count_nonzero())

        sampled_negative = torch.logical_and(consistent_negative, rand_num < sampled_num / consistent_negative.count_nonzero())

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
        loss = (- per_pixel_weight[:, sampled_mask_positive] * gt_corrs[:, sampled_mask_positive] * corr[:, sampled_mask_positive]).mean() \
                + (per_pixel_weight[:, sampled_mask_negative] * (1 - gt_corrs[:, sampled_mask_negative]) * torch.relu(corr[:, sampled_mask_negative])).mean() \
                + opt.rfn * rendered_feature_norm_reg

        with torch.no_grad():
            cosine_pos = corr[gt_corrs == 1].mean()
            cosine_neg = corr[gt_corrs == 0].mean()

        loss.backward()

        feature_gaussians.optimizer.step()
        feature_gaussians.optimizer.zero_grad(set_to_none = True)
        viewpoint_cam.release_auxiliary_data()

        del sam_masks, sam_masks_sampled_ray, gt_corrs, rendered_features, corr
        torch.cuda.empty_cache()

        iter_end.record()

        if iteration % 10 == 0:
            progress_bar.set_postfix({
                "RFN": f"{rendered_feature_norm.item():.{3}f}",
                "Pos cos": f"{cosine_pos.item():.{3}f}",
                "Neg cos": f"{cosine_neg.item():.{3}f}",
                "Loss": f"{loss.item():.{3}f}",
            })
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
                scene.model_path,
            )

    
    scene.save_feature(iteration, target = 'contrastive_feature', smooth_weights = torch.softmax(smooth_weights, dim = -1) if smooth_weights is not None else None, smooth_type = 'traditional', smooth_K = opt.smooth_K)
    torch.save(scale_gate.state_dict(), os.path.join(scene.model_path, "point_cloud/iteration_{}/".format(iteration) + "scale_gate.pt"))

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
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
    
    print("Optimizing " + args.model_path)

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