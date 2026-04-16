#!/usr/bin/env python3
import csv
import json
import os
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from tqdm import tqdm

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from gaussian_renderer import render_contrastive_feature
from scene import FeatureGaussianModel, GaussianModel, Scene
from utils.gaussian_graph_cluster import (
    cluster_gaussians_hdbscan,
    cluster_gaussians_hdbscan_refined,
    cluster_gaussians_normalized_cut,
    load_mesh_prior_for_points,
)
from utils.general_utils import safe_state
from utils.sh_utils import SH2RGB
from utils.part_mask_utils import make_identity_gates, select_part_scale_masks

VIS_PALETTE = np.array([
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
], dtype=np.uint8)


def _load_scale_gate_state_dict(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


def _quantile_transform_factory(scales, distribution='uniform'):
    reference = torch.sort(scales.flatten().detach().cpu()).values
    denom = max(reference.numel() - 1, 1)

    def transform(values):
        original_shape = values.shape
        flat_values = values.reshape(-1).detach().cpu()
        ranks = torch.searchsorted(reference, flat_values, right=True).float()
        quantiles = torch.clamp(ranks / denom, 0.0, 1.0)
        if distribution == 'normal':
            eps = 1e-4
            quantiles = torch.clamp(quantiles, eps, 1.0 - eps)
            transformed = torch.erfinv(2 * quantiles - 1) * np.sqrt(2.0)
        else:
            transformed = quantiles
        return transformed.reshape(original_shape).to(values.device)

    return transform


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


def _mean_or_zero(values):
    return float(sum(values) / len(values)) if values else 0.0


def _safe_int(value):
    return int(value) if value is not None else 0


def _sanitize_name(value):
    return ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)


def _ensure_parent(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _save_png(array, path):
    _ensure_parent(path)
    Image.fromarray(array).save(path)


def _generate_label_codes(count, feature_dim, seed=0):
    if count <= 0:
        return np.zeros((0, feature_dim), dtype=np.float32)
    rng = np.random.default_rng(seed)
    codes = rng.choice([-1.0, 1.0], size=(count, feature_dim)).astype(np.float32)
    norms = np.linalg.norm(codes, axis=1, keepdims=True) + 1e-6
    return codes / norms


def _build_palette(count):
    if count <= 0:
        return np.zeros((0, 3), dtype=np.uint8)
    if count <= VIS_PALETTE.shape[0]:
        return VIS_PALETTE[:count].copy()
    extra_count = count - VIS_PALETTE.shape[0]
    rng = np.random.default_rng(12345)
    extra = rng.integers(0, 256, size=(extra_count, 3), dtype=np.uint8)
    return np.concatenate([VIS_PALETTE, extra], axis=0)


def _prepare_cameras(scene, split, max_views, view_stride):
    train_cameras = list(scene.getTrainCameras())
    test_cameras = list(scene.getTestCameras())

    if split == 'test' and len(test_cameras) > 0:
        cameras = test_cameras
        used_split = 'test'
    elif split == 'test':
        cameras = train_cameras
        used_split = 'train_fallback'
    elif split == 'train':
        cameras = train_cameras
        used_split = 'train'
    else:
        cameras = train_cameras + test_cameras
        used_split = 'all'

    stride = max(int(view_stride), 1)
    cameras = cameras[::stride]
    if max_views > 0:
        cameras = cameras[:max_views]
    return cameras, used_split


def _cluster_full_features(full_features, xyz, sh0_rgb, mesh_prior, method, args, rng):
    num_points = full_features.shape[0]
    if num_points == 0:
        return {
            'labels': np.zeros((0,), dtype=np.int64),
            'confidence': np.zeros((0,), dtype=np.float32),
            'counts': np.zeros((0,), dtype=np.int64),
            'num_clusters': 0,
            'method': method,
        }

    method_key = str(method).lower()

    if method_key == 'normalizedcut':
        result = cluster_gaussians_normalized_cut(
            xyz,
            full_features,
            sample_size=args.cluster_sample_size,
            graph_k=args.cluster_graph_k,
            propagation_k=max(8, min(args.cluster_graph_k, 32)),
            max_clusters=args.cluster_max_clusters,
            min_cluster_size=args.cluster_min_cluster_size,
            cut_threshold=args.cluster_cut_threshold,
            spatial_weight=args.cluster_spatial_weight,
            sh0_rgb=sh0_rgb,
            sh0_color_weight=args.cluster_sh0_color_weight,
            sh0_color_sigma=args.cluster_sh0_color_sigma,
            point_mesh_vertex_idx=None if mesh_prior is None else mesh_prior['point_mesh_vertex_idx'],
            mesh_vertex_adjacency=None if mesh_prior is None else mesh_prior['mesh_vertex_adjacency'],
            mesh_weight=args.cluster_mesh_weight,
            random_state=args.seed,
        )
        result['method'] = 'NormalizedCut'
        return result

    if method_key == 'hdbscanrefined':
        result = cluster_gaussians_hdbscan_refined(
            xyz,
            full_features,
            sample_size=args.cluster_sample_size,
            hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
            hdbscan_epsilon=args.hdbscan_epsilon,
            graph_k=args.cluster_graph_k,
            min_cluster_size=args.cluster_min_cluster_size,
            spatial_scale=2.5,
            spatial_weight=args.cluster_spatial_weight,
            sh0_rgb=sh0_rgb,
            sh0_color_weight=args.cluster_sh0_color_weight,
            sh0_color_sigma=args.cluster_sh0_color_sigma,
            random_state=args.seed,
        )
        result['method'] = 'HDBSCANRefined'
        return result

    result = cluster_gaussians_hdbscan(
        full_features,
        sample_size=args.cluster_sample_size,
        hdbscan_min_cluster_size=args.hdbscan_min_cluster_size,
        hdbscan_epsilon=args.hdbscan_epsilon,
        random_state=args.seed,
    )
    result['method'] = 'HDBSCAN'
    return result


def _compute_connectivity_metrics(xyz, labels, connectivity_k):
    valid = labels >= 0
    valid_labels = np.unique(labels[valid])
    if valid_labels.size == 0:
        return {
            'cluster_size_min': 0,
            'cluster_size_median': 0.0,
            'cluster_size_max': 0,
            'connectivity_mean': 0.0,
            'connectivity_weighted_mean': 0.0,
            'cluster_sizes': [],
        }

    ratios = []
    cluster_sizes = []
    for cluster_id in valid_labels.tolist():
        cluster_idx = np.nonzero(labels == int(cluster_id))[0]
        size = int(cluster_idx.size)
        cluster_sizes.append(size)
        if size <= 1:
            ratios.append(1.0)
            continue

        points = xyz[cluster_idx]
        query_k = min(int(connectivity_k) + 1, size)
        distances, neighbors = cKDTree(points).query(points, k=query_k, workers=-1)
        if distances.ndim == 1:
            neighbors = neighbors[:, None]
        if neighbors.shape[1] <= 1:
            ratios.append(1.0)
            continue

        rows = np.repeat(np.arange(size, dtype=np.int32), neighbors.shape[1] - 1)
        cols = neighbors[:, 1:].reshape(-1).astype(np.int32, copy=False)
        values = np.ones(rows.shape[0], dtype=np.uint8)
        graph = csr_matrix((values, (rows, cols)), shape=(size, size), dtype=np.uint8)
        graph = graph.maximum(graph.transpose())
        _, component_labels = connected_components(graph, directed=False, return_labels=True)
        component_sizes = np.bincount(component_labels)
        ratios.append(float(component_sizes.max() / max(size, 1)))

    total_points = max(sum(cluster_sizes), 1)
    weighted_ratio = float(sum(r * n for r, n in zip(ratios, cluster_sizes)) / total_points)
    return {
        'cluster_size_min': int(min(cluster_sizes)),
        'cluster_size_median': float(np.median(cluster_sizes)),
        'cluster_size_max': int(max(cluster_sizes)),
        'connectivity_mean': float(sum(ratios) / len(ratios)),
        'connectivity_weighted_mean': weighted_ratio,
        'cluster_sizes': cluster_sizes,
    }


def _decode_cluster_labels(rendered_codes, cluster_codes):
    if cluster_codes.shape[0] == 0:
        height, width = rendered_codes.shape[1], rendered_codes.shape[2]
        empty_labels = np.full((height, width), -1, dtype=np.int64)
        empty_conf = np.zeros((height, width), dtype=np.float32)
        empty_valid = np.zeros((height, width), dtype=bool)
        return empty_labels, empty_conf, empty_valid

    codebook = cluster_codes.to(rendered_codes.device)
    channels, height, width = rendered_codes.shape
    flat = rendered_codes.permute(1, 2, 0).reshape(-1, channels)
    norms = flat.norm(dim=-1)
    valid = norms > 1e-6

    labels = torch.full((flat.shape[0],), -1, device=flat.device, dtype=torch.long)
    confidence = torch.zeros((flat.shape[0],), device=flat.device, dtype=torch.float32)
    if valid.any():
        flat_valid = F.normalize(flat[valid], dim=-1, p=2)
        similarity = flat_valid @ codebook.t()
        confidence_valid, labels_valid = similarity.max(dim=-1)
        labels[valid] = labels_valid
        confidence[valid] = confidence_valid

    return (
        labels.view(height, width).detach().cpu().numpy().astype(np.int64, copy=False),
        confidence.view(height, width).detach().cpu().numpy().astype(np.float32, copy=False),
        valid.view(height, width).detach().cpu().numpy().astype(bool, copy=False),
    )


def _best_cluster_overlap(label_map, mask, num_clusters):
    mask = mask.astype(bool, copy=False)
    mask_area = int(mask.sum())
    if mask_area == 0 or num_clusters <= 0:
        return {'best_cluster': -1, 'best_iou': 0.0, 'best_precision': 0.0, 'best_recall': 0.0, 'mask_area': mask_area}

    flat_labels = label_map.reshape(-1)
    valid = flat_labels >= 0
    cluster_area = np.bincount(flat_labels[valid], minlength=num_clusters).astype(np.int64, copy=False)
    mask_flat = mask.reshape(-1)
    masked_labels = flat_labels[mask_flat & valid]
    if masked_labels.size == 0:
        return {'best_cluster': -1, 'best_iou': 0.0, 'best_precision': 0.0, 'best_recall': 0.0, 'mask_area': mask_area}

    intersections = np.bincount(masked_labels, minlength=num_clusters).astype(np.int64, copy=False)
    candidate_ids = np.nonzero(intersections > 0)[0]
    if candidate_ids.size == 0:
        return {'best_cluster': -1, 'best_iou': 0.0, 'best_precision': 0.0, 'best_recall': 0.0, 'mask_area': mask_area}

    candidate_intersections = intersections[candidate_ids].astype(np.float32, copy=False)
    candidate_cluster_area = np.maximum(cluster_area[candidate_ids].astype(np.float32, copy=False), 1.0)
    unions = mask_area + candidate_cluster_area - candidate_intersections
    ious = candidate_intersections / np.maximum(unions, 1.0)
    best_idx = int(np.argmax(ious))
    best_cluster = int(candidate_ids[best_idx])
    best_intersection = float(candidate_intersections[best_idx])
    best_cluster_area = float(candidate_cluster_area[best_idx])

    return {
        'best_cluster': best_cluster,
        'best_iou': float(ious[best_idx]),
        'best_precision': float(best_intersection / max(best_cluster_area, 1.0)),
        'best_recall': float(best_intersection / max(mask_area, 1.0)),
        'mask_area': mask_area,
    }


def _colorize_labels(label_map, num_clusters):
    palette = _build_palette(num_clusters)
    rgb = np.zeros(label_map.shape + (3,), dtype=np.uint8)
    valid = label_map >= 0
    if num_clusters > 0 and np.any(valid):
        rgb[valid] = palette[label_map[valid]]
    return rgb


def _overlay_image(image_rgb, label_rgb):
    return (0.35 * image_rgb.astype(np.float32) + 0.65 * label_rgb.astype(np.float32)).clip(0, 255).astype(np.uint8)


def _write_scale_csv(path, metrics):
    fieldnames = [
        'scale',
        'num_clusters',
        'cluster_confidence_mean',
        'cluster_size_min',
        'cluster_size_median',
        'cluster_size_max',
        'connectivity_mean',
        'connectivity_weighted_mean',
        'selected_masks',
        'mask_best_iou_mean',
        'mask_best_precision_mean',
        'mask_best_recall_mean',
        'pixel_confidence_mean',
        'evaluated_views',
    ]
    _ensure_parent(path)
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key) for key in fieldnames})


def evaluate(args):
    safe_state(args.quiet)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = ModelParams(ArgumentParser(add_help=False), sentinel=True).extract(args)
    dataset.need_features = False
    dataset.need_masks = True
    dataset.feature_dim = getattr(dataset, 'feature_dim', 32)
    feature_dim = dataset.feature_dim

    pipe = PipelineParams(ArgumentParser(add_help=False)).extract(args)

    gaussians = GaussianModel(dataset.sh_degree)
    feature_gaussians = FeatureGaussianModel(feature_dim)
    scene = Scene(
        dataset,
        gaussians=gaussians,
        feature_gaussians=feature_gaussians,
        load_iteration=args.scene_iteration,
        feature_load_iteration=args.iteration,
        shuffle=False,
        target='contrastive_feature',
        mode='eval',
    )

    if scene.feature_loaded_iter is None:
        raise RuntimeError(f'Could not resolve a contrastive feature iteration under {dataset.model_path}')

    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, feature_dim, bias=True),
        torch.nn.Sigmoid(),
    ).cuda()
    feature_root = scene.feature_artifact_root or dataset.feature_model_path or dataset.model_path
    scale_gate_path = os.path.join(feature_root, 'point_cloud', f'iteration_{scene.feature_loaded_iter}', 'scale_gate.pt')
    if not os.path.exists(scale_gate_path):
        fallback_scale_gate_path = os.path.join(dataset.model_path, 'point_cloud', f'iteration_{scene.feature_loaded_iter}', 'scale_gate.pt')
        if feature_root != dataset.model_path and os.path.exists(fallback_scale_gate_path):
            scale_gate_path = fallback_scale_gate_path
        else:
            raise FileNotFoundError(f'Scale gate checkpoint not found: {scale_gate_path}')
    scale_gate.load_state_dict(_load_scale_gate_state_dict(scale_gate_path))
    scale_gate.eval()

    cameras, used_split = _prepare_cameras(scene, args.camera_split, args.max_views, args.view_stride)
    if len(cameras) == 0:
        raise RuntimeError('No cameras available for evaluation.')

    quantile_cameras = list(scene.getTrainCameras())
    if len(quantile_cameras) == 0:
        quantile_cameras = cameras
    all_scales = torch.cat([cam.mask_scales for cam in quantile_cameras])
    q_trans = _quantile_transform_factory(all_scales, distribution='uniform')

    scale_aware_dim = args.scale_aware_dim
    fixed_scale_gate = None
    if 0 < scale_aware_dim < feature_dim:
        fixed_scale_gate = torch.tensor(
            [[1 for _ in range(feature_dim - scale_aware_dim + i)] + [0 for _ in range(scale_aware_dim - i)] for i in range(scale_aware_dim + 1)],
            device='cuda',
            dtype=torch.float32,
        )

    point_features = feature_gaussians.get_point_features.detach()
    normed_point_features = F.normalize(point_features, dim=-1, p=2)
    xyz = feature_gaussians.get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
    scene_sh0_rgb = None
    if args.cluster_sh0_color_weight > 0:
        scene_sh0_rgb_torch = SH2RGB(scene.gaussians.get_features[:, 0, :].detach()).clamp(0.0, 1.0)
        if scene_sh0_rgb_torch.shape[0] != feature_gaussians.get_xyz.shape[0]:
            raise RuntimeError(
                f"Scene Gaussian count ({scene_sh0_rgb_torch.shape[0]}) does not match feature Gaussian count ({feature_gaussians.get_xyz.shape[0]}) for SH0 color weighting."
            )
        scene_sh0_rgb = scene_sh0_rgb_torch.cpu().numpy().astype(np.float32, copy=False)

    mesh_prior = None
    if args.cluster_mesh_weight > 0:
        mesh_prior = load_mesh_prior_for_points(
            xyz,
            model_path=dataset.model_path,
            mesh_path=args.cluster_mesh_path if args.cluster_mesh_path else None,
        )

    background = torch.ones([feature_dim], dtype=torch.float32, device='cuda') if dataset.white_background else torch.zeros([feature_dim], dtype=torch.float32, device='cuda')

    eval_root = Path(args.output_root)
    eval_root.mkdir(parents=True, exist_ok=True)
    if args.save_viz_views > 0:
        (eval_root / 'viz').mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    scale_metrics = []

    for scale_idx, scale in enumerate(args.eval_scales):
        scale_tensor = torch.tensor([scale], device=normed_point_features.device, dtype=torch.float32)
        gates = _compute_effective_scale_gates(
            scale_tensor,
            scale_gate,
            feature_dim,
            scale_aware_dim,
            supervision_mode=args.supervision_mode,
            fixed_scale_gate=fixed_scale_gate,
        ).squeeze(0)
        scale_conditioned = F.normalize(normed_point_features * gates.unsqueeze(0), dim=-1, p=2)
        scale_conditioned_np = scale_conditioned.detach().cpu().numpy().astype(np.float32, copy=False)

        cluster_result = _cluster_full_features(
            scale_conditioned_np,
            xyz,
            scene_sh0_rgb,
            mesh_prior,
            args.cluster_method,
            args,
            rng,
        )
        labels = cluster_result['labels']
        confidence = cluster_result['confidence']
        num_clusters = int(cluster_result['num_clusters'])

        connectivity = _compute_connectivity_metrics(xyz, labels, args.connectivity_k)
        codes = _generate_label_codes(num_clusters, feature_dim, seed=args.seed + scale_idx * 1009)
        point_codes = np.zeros((labels.shape[0], feature_dim), dtype=np.float32)
        valid_point_mask = labels >= 0
        if num_clusters > 0 and np.any(valid_point_mask):
            point_codes[valid_point_mask] = codes[labels[valid_point_mask]]
        point_codes_torch = torch.from_numpy(point_codes).cuda().float()
        cluster_codes_torch = torch.from_numpy(codes).cuda().float() if num_clusters > 0 else torch.zeros((0, feature_dim), device='cuda')

        mask_iou_scores = []
        mask_precision_scores = []
        mask_recall_scores = []
        pixel_confidence_scores = []
        selected_masks = 0
        evaluated_views = 0
        saved_viz = 0

        view_iter = tqdm(cameras, desc=f'Eval scale {scale:.2f}', leave=False) if not args.quiet else cameras
        for view_idx, camera in enumerate(view_iter):
            normalized_mask_scales = q_trans(camera.mask_scales.float()).detach().cpu()
            selected_masks_tensor, _, _, selection_stats = select_part_scale_masks(
                camera.original_masks.bool(),
                normalized_mask_scales,
                scale,
                args.mask_scale_tolerance,
                args.mask_min_area,
                args.mask_max_area_ratio,
                args.mask_max_iou_overlap,
                args.mask_boundary_erode_kernel,
            )
            if selected_masks_tensor.shape[0] < int(max(args.min_valid_masks_per_view, 1)):
                camera.release_auxiliary_data()
                continue

            target_height = max(1, camera.image_height // max(int(args.render_downsample), 1))
            target_width = max(1, camera.image_width // max(int(args.render_downsample), 1))
            camera.feature_height = target_height
            camera.feature_width = target_width

            with torch.no_grad():
                rendered_codes = render_contrastive_feature(
                    camera,
                    feature_gaussians,
                    pipe,
                    background,
                    point_feature_override=point_codes_torch,
                )['render']

            label_map, pixel_confidence_map, valid_map = _decode_cluster_labels(rendered_codes, cluster_codes_torch)
            if np.any(valid_map):
                pixel_confidence_scores.append(float(pixel_confidence_map[valid_map].mean()))

            selected_masks_tensor = selected_masks_tensor.float().unsqueeze(1)
            resized_masks = F.interpolate(selected_masks_tensor, size=(target_height, target_width), mode='nearest').squeeze(1).cpu().numpy() > 0.5

            for mask in resized_masks:
                overlap = _best_cluster_overlap(label_map, mask, num_clusters)
                mask_iou_scores.append(overlap['best_iou'])
                mask_precision_scores.append(overlap['best_precision'])
                mask_recall_scores.append(overlap['best_recall'])
                selected_masks += 1

            if args.save_viz_views > 0 and saved_viz < args.save_viz_views:
                image_resized = F.interpolate(
                    camera.original_image.unsqueeze(0),
                    size=(target_height, target_width),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)
                image_rgb = torch.clamp(image_resized.detach().cpu(), 0.0, 1.0).permute(1, 2, 0).mul(255.0).byte().numpy()
                label_rgb = _colorize_labels(label_map, num_clusters)
                overlay = _overlay_image(image_rgb, label_rgb)
                prefix = eval_root / 'viz' / f'scale_{scale:.2f}_view_{view_idx:03d}'
                _save_png(image_rgb, str(prefix) + '_rgb.png')
                _save_png(label_rgb, str(prefix) + '_clusters.png')
                _save_png(overlay, str(prefix) + '_overlay.png')
                saved_viz += 1

            evaluated_views += 1
            camera.release_auxiliary_data()
            torch.cuda.empty_cache()

        scale_metrics.append({
            'scale': float(scale),
            'num_clusters': num_clusters,
            'cluster_confidence_mean': float(confidence.mean()) if confidence.size > 0 else 0.0,
            'cluster_size_min': connectivity['cluster_size_min'],
            'cluster_size_median': connectivity['cluster_size_median'],
            'cluster_size_max': connectivity['cluster_size_max'],
            'connectivity_mean': connectivity['connectivity_mean'],
            'connectivity_weighted_mean': connectivity['connectivity_weighted_mean'],
            'selected_masks': int(selected_masks),
            'mask_best_iou_mean': _mean_or_zero(mask_iou_scores),
            'mask_best_precision_mean': _mean_or_zero(mask_precision_scores),
            'mask_best_recall_mean': _mean_or_zero(mask_recall_scores),
            'pixel_confidence_mean': _mean_or_zero(pixel_confidence_scores),
            'evaluated_views': int(evaluated_views),
        })

        torch.cuda.empty_cache()

    summary = {
        'mean_num_clusters': _mean_or_zero([row['num_clusters'] for row in scale_metrics]),
        'mean_cluster_confidence': _mean_or_zero([row['cluster_confidence_mean'] for row in scale_metrics]),
        'mean_connectivity_weighted': _mean_or_zero([row['connectivity_weighted_mean'] for row in scale_metrics]),
        'mean_mask_best_iou': _mean_or_zero([row['mask_best_iou_mean'] for row in scale_metrics if row['selected_masks'] > 0]),
        'mean_mask_best_precision': _mean_or_zero([row['mask_best_precision_mean'] for row in scale_metrics if row['selected_masks'] > 0]),
        'mean_mask_best_recall': _mean_or_zero([row['mask_best_recall_mean'] for row in scale_metrics if row['selected_masks'] > 0]),
        'mean_pixel_confidence': _mean_or_zero([row['pixel_confidence_mean'] for row in scale_metrics if row['evaluated_views'] > 0]),
        'total_selected_masks': int(sum(row['selected_masks'] for row in scale_metrics)),
        'total_evaluated_views': int(sum(row['evaluated_views'] for row in scale_metrics)),
    }

    result = {
        'label': args.label,
        'model_path': dataset.model_path,
        'source_path': dataset.source_path,
        'feature_iteration': int(scene.feature_loaded_iter),
        'cluster_method': args.cluster_method,
        'cluster_spatial_weight': float(args.cluster_spatial_weight),
        'cluster_sh0_color_weight': float(args.cluster_sh0_color_weight),
        'cluster_sh0_color_sigma': float(args.cluster_sh0_color_sigma),
        'cluster_mesh_weight': float(args.cluster_mesh_weight),
        'cluster_mesh_path_used': None if mesh_prior is None else mesh_prior['mesh_path'],
        'supervision_mode': args.supervision_mode,
        'mask_scale_target': float(args.mask_scale_target),
        'mask_scale_tolerance': float(args.mask_scale_tolerance),
        'mask_min_area': int(args.mask_min_area),
        'mask_max_area_ratio': float(args.mask_max_area_ratio),
        'mask_max_iou_overlap': float(args.mask_max_iou_overlap),
        'mask_boundary_erode_kernel': int(args.mask_boundary_erode_kernel),
        'min_valid_masks_per_view': int(args.min_valid_masks_per_view),
        'camera_split_requested': args.camera_split,
        'camera_split_used': used_split,
        'num_views': len(cameras),
        'render_downsample': int(args.render_downsample),
        'mask_scale_tolerance': float(args.mask_scale_tolerance),
        'eval_scales': [float(scale) for scale in args.eval_scales],
        'scales': scale_metrics,
        'summary': summary,
    }

    _ensure_parent(args.output_json)
    with open(args.output_json, 'w') as handle:
        json.dump(result, handle, indent=2)
    _write_scale_csv(args.output_csv, scale_metrics)
    print(json.dumps({'output_json': args.output_json, 'output_csv': args.output_csv, 'summary': summary}, indent=2))


def build_parser():
    parser = ArgumentParser(description='Evaluate SAGA part-cluster quality for ablation studies.')
    ModelParams(parser, sentinel=True)
    OptimizationParams(parser)
    PipelineParams(parser)
    parser.add_argument('--iteration', type=int, default=-1, help='Feature iteration to evaluate. -1 uses the latest contrastive feature checkpoint.')
    parser.add_argument('--scene_iteration', type=int, default=-1, help='Optional scene iteration override. -1 uses the latest scene checkpoint if needed.')
    parser.add_argument('--label', type=str, default=None, help='Human-readable run label stored in the output JSON.')
    parser.add_argument('--camera_split', type=str, default='test', choices=['train', 'test', 'all'])
    parser.add_argument('--max_views', type=int, default=16)
    parser.add_argument('--view_stride', type=int, default=1)
    parser.add_argument('--eval_scales', nargs='+', type=float, default=[0.2, 0.5, 0.8])
    parser.add_argument('--render_downsample', type=int, default=4)
    parser.add_argument('--cluster_method', type=str, default='NormalizedCut', choices=['HDBSCAN', 'HDBSCANRefined', 'NormalizedCut'])
    parser.add_argument('--cluster_sample_size', type=int, default=20000)
    parser.add_argument('--cluster_graph_k', type=int, default=16)
    parser.add_argument('--cluster_max_clusters', type=int, default=24)
    parser.add_argument('--cluster_min_cluster_size', type=int, default=128)
    parser.add_argument('--cluster_cut_threshold', type=float, default=0.12)
    parser.add_argument('--cluster_spatial_weight', type=float, default=1.0)
    parser.add_argument('--cluster_sh0_color_weight', type=float, default=0.0)
    parser.add_argument('--cluster_sh0_color_sigma', type=float, default=0.25)
    parser.add_argument('--cluster_mesh_weight', type=float, default=0.0)
    parser.add_argument('--cluster_mesh_path', type=str, default=None)
    parser.add_argument('--hdbscan_min_cluster_size', type=int, default=10)
    parser.add_argument('--hdbscan_epsilon', type=float, default=0.01)
    parser.add_argument('--connectivity_k', type=int, default=12)
    parser.add_argument('--save_viz_views', type=int, default=0)
    parser.add_argument('--output_root', type=str, default=None)
    parser.add_argument('--output_json', type=str, default=None)
    parser.add_argument('--output_csv', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--quiet', action='store_true')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = get_combined_args(parser, target_cfg_file='cfg_args')
    if not hasattr(args, 'label'):
        args.label = None
    if not hasattr(args, 'output_root'):
        args.output_root = None
    if not hasattr(args, 'output_json'):
        args.output_json = None
    if not hasattr(args, 'output_csv'):
        args.output_csv = None
    if args.label is None:
        args.label = os.path.basename(os.path.normpath(args.model_path))
    method_slug = _sanitize_name(args.cluster_method.lower())
    label_slug = _sanitize_name(args.label)
    if args.output_root is None:
        args.output_root = os.path.join(args.model_path, 'ablation_eval')
    if args.output_json is None:
        args.output_json = os.path.join(args.output_root, f'{label_slug}_{method_slug}.json')
    if args.output_csv is None:
        args.output_csv = os.path.join(args.output_root, f'{label_slug}_{method_slug}.csv')
    evaluate(args)
