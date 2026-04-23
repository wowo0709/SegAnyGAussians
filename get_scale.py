import torch


import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from argparse import ArgumentParser, Namespace, SUPPRESS
import cv2

from arguments import ModelParams, PipelineParams
from scene import Scene, GaussianModel, FeatureGaussianModel

import gaussian_renderer
import importlib
importlib.reload(gaussian_renderer)

import os
FEATURE_DIM = 32

DATA_ROOT = './data/nerf_llff_data_for_3dgs/'
# MODEL_PATH = './output/figurines_lerf_poses/'
# MODEL_PATH = './output/figurines/'

ALLOW_PRINCIPLE_POINT_SHIFT = False
SCALE_DEFINITION_CHOICES = ("legacy_3d_std", "area_only", "hybrid_area_3d_extent")

def resolve_mask_root(image_root):
    candidates = [
        os.path.join(image_root, 'sam_masks'),
        os.path.join(image_root, 'images', 'sam_masks'),
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    raise FileNotFoundError(f"Could not find sam_masks under image_root: {image_root}")


def get_combined_args(parser : ArgumentParser):
    # cmdlne_string = ['--model_path', model_path]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args()
    
    target_cfg_file = "cfg_args"

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, target_cfg_file)
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file found: {}".format(cfgfilepath))
        pass
    args_cfgfile = eval(cfgfile_string)

    # for k in args_cfgfile.__dict__.keys():
        # print(k, args_cfgfile.__dict__[k], "?")

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v

    # for k in merged_dict.keys():
        # print(k, merged_dict[k])
    return Namespace(**merged_dict)

def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


def _compute_mask_area_ratio(mask):
    return mask.float().mean()


def _compute_legacy_3d_std(points_in_3d):
    if points_in_3d.shape[0] < 2:
        return points_in_3d.new_tensor(0.0)
    return (points_in_3d.std(dim=0) * 2).norm()


def _compute_robust_3d_extent(points_in_3d):
    if points_in_3d.shape[0] < 2:
        return points_in_3d.new_tensor(0.0)
    q10 = torch.quantile(points_in_3d, 0.1, dim=0)
    q90 = torch.quantile(points_in_3d, 0.9, dim=0)
    return (q90 - q10).norm()


def compute_mask_scale(mask, points_in_3d, scale_definition):
    mask = mask.bool()
    if mask.numel() == 0 or not mask.any():
        return 0.0

    area_ratio = _compute_mask_area_ratio(mask)
    if scale_definition == "area_only":
        return float(area_ratio.item())

    points_in_mask = points_in_3d[mask]
    if scale_definition == "legacy_3d_std":
        return float(_compute_legacy_3d_std(points_in_mask).item())
    if scale_definition == "hybrid_area_3d_extent":
        robust_extent = _compute_robust_3d_extent(points_in_mask)
        hybrid_scale = torch.sqrt(torch.clamp(area_ratio, min=0.0)) * robust_extent
        return float(hybrid_scale.item())

    raise ValueError(f"Unknown scale definition: {scale_definition}")


if __name__ == '__main__':
    
    parser = ArgumentParser(description="Get scales for SAM masks")

    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--segment", action="store_true")
    parser.add_argument('--idx', default=0, type=int)
    parser.add_argument('--precomputed_mask', default=None, type=str)

    parser.add_argument("--image_root", default='/datasets/nerf_data/360_v2/garden/', type=str)
    parser.add_argument(
        "--scale-definition",
        default="legacy_3d_std",
        choices=SCALE_DEFINITION_CHOICES,
        type=str,
    )
    parser.add_argument(
        "--mask-scales-dir-name",
        dest="mask_scales_dir_name",
        default=SUPPRESS,
        type=str,
    )

    args = get_combined_args(parser)

    dataset = model.extract(args)
    dataset.need_features = False
    dataset.need_masks = False

    # ALLOW_PRINCIPLE_POINT_SHIFT = 'lerf' in args.model_path
    dataset.allow_principle_point_shift = ALLOW_PRINCIPLE_POINT_SHIFT

    feature_gaussians = None
    scene_gaussians = GaussianModel(dataset.sh_degree)

    scene = Scene(dataset, scene_gaussians, feature_gaussians, load_iteration=-1, feature_load_iteration=-1, shuffle=False, mode='eval', target='scene')


    mask_root = resolve_mask_root(args.image_root)

    from tqdm import tqdm
    OUTPUT_DIR = os.path.join(args.image_root, args.mask_scales_dir_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("Scale definition:", args.scale_definition)
    print("Mask scales output dir:", OUTPUT_DIR)

    cameras = scene.getTrainCameras()
    background = torch.zeros(scene_gaussians.get_mask.shape[0], 3, device='cuda')
    render_pipe = pipeline.extract(args)

    with torch.no_grad():
        for it, view in tqdm(enumerate(cameras), total=len(cameras)):
            rendered_pkg = gaussian_renderer.render_with_depth(view, scene_gaussians, render_pipe, background)
            depth = rendered_pkg['depth'].squeeze().cpu()

            mask_path = os.path.join(mask_root, view.image_name + '.pt')
            corresponding_masks = torch.load(mask_path, map_location='cpu').float()

            grid_index = generate_grid_index(depth)
            points_in_3D = torch.zeros(depth.shape[0], depth.shape[1], 3)
            points_in_3D[:, :, -1] = depth

            # caluculate cx cy fx fy with FoVx FoVy
            cx = depth.shape[1] / 2
            cy = depth.shape[0] / 2
            fx = cx / np.tan(cameras[0].FoVx / 2)
            fy = cy / np.tan(cameras[0].FoVy / 2)

            points_in_3D[:, :, 0] = (grid_index[:, :, 0] - cx) * depth / fx
            points_in_3D[:, :, 1] = (grid_index[:, :, 1] - cy) * depth / fy

            upsampled_mask = torch.nn.functional.interpolate(
                corresponding_masks.unsqueeze(1),
                mode='bilinear',
                size=(depth.shape[0], depth.shape[1]),
                align_corners=False,
            )

            eroded_masks = torch.conv2d(
                upsampled_mask.float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3),
                padding=1,
            )
            eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

            scale = torch.zeros(len(corresponding_masks))
            for mask_id in range(len(corresponding_masks)):
                scale[mask_id] = compute_mask_scale(
                    eroded_masks[mask_id],
                    points_in_3D,
                    args.scale_definition,
                )

            torch.save(scale, os.path.join(OUTPUT_DIR, view.image_name + '.pt'))
