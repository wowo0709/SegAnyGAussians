#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from gaussian_renderer import render_with_depth
from scene.cameras import MiniCam
from scene.gaussian_model import GaussianModel
from utils.graphics_utils import getProjectionMatrix, getWorld2View2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Gaussian export views with the actual SegAnyGaussians rasterizer.")
    parser.add_argument("--job-json", type=Path, required=True, help="Path to a render job JSON file.")
    return parser.parse_args()


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def focal_to_fov(focal: float, pixels: int) -> float:
    return 2.0 * math.atan(float(pixels) / (2.0 * float(focal)))


def build_minicam(view: dict, device: torch.device) -> MiniCam:
    width = int(view["width"])
    height = int(view["height"])
    fx = float(view["fx"])
    fy = float(view["fy"])
    cx = float(view["cx"])
    cy = float(view["cy"])
    znear = float(view.get("znear", 0.01))
    zfar = float(view.get("zfar", 100.0))

    rotation = np.asarray(view["camera_rotation"], dtype=np.float32)
    position = np.asarray(view["camera_position"], dtype=np.float32)
    translation = -(rotation.T @ position)

    world_view = torch.tensor(getWorld2View2(rotation, translation), dtype=torch.float32, device=device).transpose(0, 1)
    projection = getProjectionMatrix(
        znear=znear,
        zfar=zfar,
        fovX=focal_to_fov(fx, width),
        fovY=focal_to_fov(fy, height),
        cx=cx,
        cy=cy,
        w=width,
        h=height,
    ).transpose(0, 1).to(device=device, dtype=torch.float32)
    full_proj = world_view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0)

    return MiniCam(
        width=width,
        height=height,
        fovy=focal_to_fov(fy, height),
        fovx=focal_to_fov(fx, width),
        znear=znear,
        zfar=zfar,
        world_view_transform=world_view,
        full_proj_transform=full_proj,
    )


def tensor_to_image_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3 and array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.clip(np.round(array * 255.0), 0.0, 255.0).astype(np.uint8)


def tensor_to_mask_uint8(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3:
        array = array.max(axis=-1)
    return (array > 0.5).astype(np.uint8) * 255


def tensor_to_depth_float(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if array.ndim == 3 and array.shape[0] in (1, 3):
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32)


def save_depth_preview(depth: np.ndarray, path: str | Path) -> None:
    valid = depth > 0
    preview = np.full((*depth.shape, 3), 255, dtype=np.uint8)
    if np.any(valid):
        lo = float(depth[valid].min())
        hi = float(depth[valid].max())
        norm = np.zeros_like(depth, dtype=np.float32)
        if hi > lo:
            norm[valid] = (depth[valid] - lo) / (hi - lo)
        gray = np.clip(np.round(norm * 255.0), 0.0, 255.0).astype(np.uint8)
        preview[valid] = np.stack([gray[valid]] * 3, axis=-1)
    ensure_dir(Path(path).parent)
    Image.fromarray(preview).save(path)


def render_job(job_path: Path) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for SegAnyGaussians export rendering, but no CUDA device is available.")

    payload = json.loads(job_path.read_text())
    ply_path = Path(payload["ply_path"])
    if not ply_path.exists():
        raise FileNotFoundError(f"Gaussian export not found: {ply_path}")

    device = torch.device("cuda")
    background_rgb = payload.get("background_rgb", [1.0, 1.0, 1.0])
    bg = torch.tensor(background_rgb, dtype=torch.float32, device=device)

    gaussian = GaussianModel(3)
    gaussian.load_ply(str(ply_path))
    pipe = SimpleNamespace(debug=False, compute_cov3D_python=False, convert_SHs_python=False)

    views = payload.get("views", [])
    if not views:
        raise ValueError(f"No views found in render job: {job_path}")

    for view in views:
        camera = build_minicam(view, device)
        render_pack = render_with_depth(camera, gaussian, pipe, bg)

        rgb = tensor_to_image_uint8(render_pack["render"])
        mask = tensor_to_mask_uint8(render_pack["mask"])
        depth = tensor_to_depth_float(render_pack["depth"])

        output_rgb = Path(view["output_rgb"])
        output_mask = Path(view["output_mask"])
        output_depth_npy = Path(view["output_depth_npy"])
        output_depth_png = Path(view["output_depth_png"])

        ensure_dir(output_rgb.parent)
        ensure_dir(output_mask.parent)
        ensure_dir(output_depth_npy.parent)
        ensure_dir(output_depth_png.parent)

        Image.fromarray(rgb).save(output_rgb)
        Image.fromarray(mask).save(output_mask)
        np.save(output_depth_npy, depth.astype(np.float32))
        save_depth_preview(depth, output_depth_png)

        if not np.any(depth > 0):
            raise RuntimeError(f"Rendered depth is empty for view {view.get('view_id', '<unknown>')} in {job_path}")


def main() -> None:
    args = parse_args()
    render_job(args.job_json)


if __name__ == "__main__":
    main()
