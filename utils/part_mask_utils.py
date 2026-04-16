import torch
import torch.nn.functional as F


def make_identity_gates(scale_values, feature_dim):
    flat = scale_values.reshape(-1)
    return torch.ones((flat.shape[0], feature_dim), device=scale_values.device, dtype=torch.float32)


def erode_binary_masks(masks, kernel_size):
    masks = masks.bool()
    if masks.numel() == 0 or kernel_size <= 1:
        return masks
    kernel_size = int(max(kernel_size, 1))
    mask_float = masks.float().unsqueeze(1)
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=masks.device, dtype=mask_float.dtype)
    convolved = F.conv2d(mask_float, kernel, padding=kernel_size // 2)
    return (convolved >= float(kernel_size * kernel_size) - 1e-6).squeeze(1)


def _compute_mask_area_and_border_stats(masks):
    masks = masks.bool()
    if masks.numel() == 0:
        device = masks.device
        empty_float = torch.empty((0,), device=device, dtype=torch.float32)
        empty_long = torch.empty((0,), device=device, dtype=torch.long)
        return empty_float, empty_float, empty_long

    height, width = masks.shape[-2:]
    image_area = float(max(height * width, 1))
    flat_masks = masks.reshape(masks.shape[0], -1)
    area_ratios = flat_masks.sum(dim=1).float() / image_area

    top = masks[:, 0, :].any(dim=1)
    bottom = masks[:, -1, :].any(dim=1)
    left = masks[:, :, 0].any(dim=1)
    right = masks[:, :, -1].any(dim=1)
    border_touch_count = torch.stack([top, bottom, left, right], dim=1).sum(dim=1)

    border_pixels = (
        masks[:, 0, :].sum(dim=1)
        + masks[:, -1, :].sum(dim=1)
        + masks[:, :, 0].sum(dim=1)
        + masks[:, :, -1].sum(dim=1)
    )
    border_cov = border_pixels.float() / float(max(2 * height + 2 * width, 1))
    return area_ratios, border_cov, border_touch_count


@torch.no_grad()
def compute_background_like_mask_thresholds(train_cameras):
    all_scales = []
    all_area_ratios = []
    all_border_covs = []

    for cam in train_cameras:
        sam_masks = getattr(cam, "original_masks", None)
        mask_scales = getattr(cam, "mask_scales", None)
        if sam_masks is None or mask_scales is None:
            continue
        sam_masks = sam_masks.bool()
        mask_scales = mask_scales.reshape(-1).float()
        if sam_masks.numel() == 0 or mask_scales.numel() == 0:
            cam.release_auxiliary_data()
            continue
        area_ratios, border_cov, _ = _compute_mask_area_and_border_stats(sam_masks)
        all_scales.append(mask_scales.cpu())
        all_area_ratios.append(area_ratios.cpu())
        all_border_covs.append(border_cov.cpu())
        cam.release_auxiliary_data()

    if not all_scales:
        return None

    all_scales = torch.cat(all_scales).float()
    all_area_ratios = torch.cat(all_area_ratios).float()
    all_border_covs = torch.cat(all_border_covs).float()

    scale_q90 = float(torch.quantile(all_scales, torch.tensor(0.9)))
    area_q90 = float(torch.quantile(all_area_ratios, torch.tensor(0.9)))
    area_q95 = float(torch.quantile(all_area_ratios, torch.tensor(0.95)))
    border_cov_q90 = float(torch.quantile(all_border_covs, torch.tensor(0.9)))

    thresholds = {
        "scale_q90": scale_q90,
        "area_q90": area_q90,
        "area_q95": area_q95,
        "border_cov_q90": border_cov_q90,
        "border_cov_gate": max(0.08, border_cov_q90),
        "area_gate": max(0.05, area_q90),
        "area_touch_gate": max(0.05, 0.5 * area_q90),
    }
    return thresholds


@torch.no_grad()
def classify_background_like_masks(sam_masks, mask_scales, thresholds):
    sam_masks = sam_masks.bool()
    mask_scales = mask_scales.reshape(-1).float()
    if sam_masks.numel() == 0 or mask_scales.numel() == 0 or thresholds is None:
        device = sam_masks.device if sam_masks.numel() > 0 else mask_scales.device
        empty = torch.zeros((sam_masks.shape[0],), device=device, dtype=torch.bool)
        return empty, {
            "background_like_count": 0,
            "mean_area_ratio": 0.0,
            "mean_border_cov": 0.0,
        }

    area_ratios, border_cov, border_touch_count = _compute_mask_area_and_border_stats(sam_masks)
    background_like = area_ratios >= float(thresholds["area_q95"])
    background_like = torch.logical_or(
        background_like,
        torch.logical_and(
            border_touch_count >= 2,
            torch.logical_or(
                area_ratios >= float(thresholds["area_touch_gate"]),
                border_cov >= float(thresholds["border_cov_gate"]),
            ),
        ),
    )
    background_like = torch.logical_or(
        background_like,
        torch.logical_and(
            mask_scales >= float(thresholds["scale_q90"]),
            torch.logical_or(
                torch.logical_or(
                    area_ratios >= float(thresholds["area_gate"]),
                    border_touch_count >= 2,
                ),
                border_cov >= float(thresholds["border_cov_gate"]),
            ),
        ),
    )
    stats = {
        "background_like_count": int(background_like.sum().item()),
        "mean_area_ratio": float(area_ratios.mean().item()) if area_ratios.numel() > 0 else 0.0,
        "mean_border_cov": float(border_cov.mean().item()) if border_cov.numel() > 0 else 0.0,
    }
    return background_like, stats


@torch.no_grad()
def apply_exclusive_positive_shell_dilation(original_masks, eligible_masks, kernel_size):
    original_masks = original_masks.bool()
    eligible_masks = eligible_masks.bool().reshape(-1)
    if original_masks.numel() == 0 or kernel_size <= 1 or not eligible_masks.any():
        return original_masks, {
            "dilated_mask_count": 0,
            "added_shell_pixel_ratio": 0.0,
        }

    kernel_size = int(max(kernel_size, 1))
    if kernel_size % 2 == 0:
        kernel_size += 1
    pad = kernel_size // 2

    eligible_masks_expanded = original_masks[eligible_masks]
    dilated = F.max_pool2d(eligible_masks_expanded.float().unsqueeze(1), kernel_size=kernel_size, stride=1, padding=pad).squeeze(1) > 0.5
    shell = torch.logical_and(dilated, ~eligible_masks_expanded)

    all_original_union = original_masks.any(dim=0, keepdim=True)
    shell = torch.logical_and(shell, ~all_original_union)
    shell = torch.logical_and(shell, shell.sum(dim=0, keepdim=True) == 1)

    positive_masks = original_masks.clone()
    positive_masks[eligible_masks] = torch.logical_or(positive_masks[eligible_masks], shell)

    added_shell_pixels = int(shell.sum().item())
    image_area = float(max(original_masks.shape[-1] * original_masks.shape[-2], 1))
    stats = {
        "dilated_mask_count": int(shell.reshape(shell.shape[0], -1).any(dim=1).sum().item()),
        "added_shell_pixel_ratio": added_shell_pixels / image_area,
    }
    return positive_masks, stats


def _pairwise_mean_iou(masks):
    if masks.shape[0] < 2:
        return 0.0
    flat = masks.reshape(masks.shape[0], -1).float()
    areas = flat.sum(dim=1)
    ious = []
    for idx in range(masks.shape[0]):
        inter = (flat[idx:idx + 1] * flat[idx + 1:]).sum(dim=1)
        union = areas[idx] + areas[idx + 1:] - inter
        valid = union > 0
        if valid.any():
            ious.append((inter[valid] / union[valid]).mean())
    if not ious:
        return 0.0
    return torch.stack(ious).mean().item()


@torch.no_grad()
def select_part_scale_masks(
    sam_masks,
    normalized_mask_scales,
    target_scale,
    tolerance,
    min_area,
    max_area_ratio,
    max_iou_overlap,
    erode_kernel,
):
    sam_masks = sam_masks.bool()
    normalized_mask_scales = normalized_mask_scales.reshape(-1)
    height, width = sam_masks.shape[-2:]
    image_area = float(height * width)
    flat_masks = sam_masks.reshape(sam_masks.shape[0], -1)
    areas = flat_masks.sum(dim=1).float()

    valid = (normalized_mask_scales - float(target_scale)).abs() <= float(tolerance)
    valid = torch.logical_and(valid, areas >= float(max(min_area, 1)))
    if max_area_ratio > 0:
        valid = torch.logical_and(valid, areas <= float(max_area_ratio) * image_area)

    candidate_idx = torch.nonzero(valid, as_tuple=False).squeeze(-1)
    if candidate_idx.numel() == 0:
        empty_masks = sam_masks[:0]
        empty_scales = normalized_mask_scales[:0]
        return empty_masks, empty_scales, candidate_idx, {
            "selected_count": 0,
            "mean_area": 0.0,
            "mean_area_ratio": 0.0,
            "mean_iou_overlap": 0.0,
            "skip_reason": "no_masks_in_scale_band",
        }

    order = torch.argsort((normalized_mask_scales[candidate_idx] - float(target_scale)).abs(), descending=False)
    candidate_idx = candidate_idx[order]

    kept = []
    for idx in candidate_idx.tolist():
        candidate_mask = flat_masks[idx]
        candidate_area = areas[idx]
        if candidate_area <= 0:
            continue
        keep = True
        for kept_idx in kept:
            inter = torch.logical_and(candidate_mask, flat_masks[kept_idx]).sum().float()
            union = candidate_area + areas[kept_idx] - inter
            if union > 0 and (inter / union) > float(max_iou_overlap):
                keep = False
                break
        if keep:
            kept.append(idx)

    if not kept:
        empty_masks = sam_masks[:0]
        empty_scales = normalized_mask_scales[:0]
        return empty_masks, empty_scales, normalized_mask_scales[:0], {
            "selected_count": 0,
            "mean_area": 0.0,
            "mean_area_ratio": 0.0,
            "mean_iou_overlap": 0.0,
            "skip_reason": "all_masks_suppressed",
        }

    keep_idx = torch.tensor(kept, device=sam_masks.device, dtype=torch.long)
    selected_masks = sam_masks[keep_idx]
    selected_scales = normalized_mask_scales[keep_idx]

    if erode_kernel > 1:
        selected_masks = erode_binary_masks(selected_masks, erode_kernel)
        non_empty = selected_masks.reshape(selected_masks.shape[0], -1).any(dim=1)
        selected_masks = selected_masks[non_empty]
        selected_scales = selected_scales[non_empty]
        keep_idx = keep_idx[non_empty]

    stats = {
        "selected_count": int(selected_masks.shape[0]),
        "mean_area": 0.0,
        "mean_area_ratio": 0.0,
        "mean_iou_overlap": 0.0,
        "skip_reason": None,
    }
    if selected_masks.shape[0] == 0:
        stats["skip_reason"] = "masks_empty_after_erosion"
        return selected_masks, selected_scales, keep_idx, stats

    selected_areas = selected_masks.reshape(selected_masks.shape[0], -1).sum(dim=1).float()
    stats["mean_area"] = float(selected_areas.mean().item())
    stats["mean_area_ratio"] = float((selected_areas.mean() / image_area).item())
    stats["mean_iou_overlap"] = float(_pairwise_mean_iou(selected_masks))
    return selected_masks, selected_scales, keep_idx, stats
