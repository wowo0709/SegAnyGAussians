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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import torch
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    features: torch.tensor
    masks: torch.tensor
    mask_scales: torch.tensor
    image_path: str
    image_name: str
    width: int
    height: int
    cx: float = None
    cy: float = None
    features_path: str = None
    masks_path: str = None
    mask_scales_path: str = None

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def _dedupe_preserve_order(items):
    deduped = []
    seen = set()
    for item in items:
        if item is None:
            continue
        normalized = os.path.abspath(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return deduped

def _directory_has_images(path):
    if not path or not os.path.isdir(path):
        return False
    for entry in os.listdir(path):
        if Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
            return True
    return False

def _path_is_within(child, parent):
    try:
        Path(child).resolve().relative_to(Path(parent).resolve())
        return True
    except ValueError:
        return False

def _find_split_render_root(scene_path):
    scene_path = Path(scene_path).resolve()
    instance_name = scene_path.name
    for parent in [scene_path] + list(scene_path.parents):
        candidate = parent / "renders" / instance_name
        if candidate.is_dir():
            return str(candidate)
    return None

def _resolve_image_and_aux_roots(scene_path, images_hint=None):
    scene_path = os.path.abspath(scene_path)
    candidate_dirs = []
    if images_hint:
        if os.path.isabs(images_hint):
            candidate_dirs.append(images_hint)
        else:
            candidate_dirs.append(os.path.join(scene_path, images_hint))
    candidate_dirs.extend([
        scene_path,
        os.path.join(scene_path, "images"),
        _find_split_render_root(scene_path),
    ])
    candidate_dirs = _dedupe_preserve_order(candidate_dirs)

    image_root = None
    for candidate in candidate_dirs:
        if _directory_has_images(candidate):
            image_root = candidate
            break

    if image_root is None:
        for candidate in candidate_dirs:
            if os.path.isdir(candidate):
                image_root = candidate
                break

    if image_root is None:
        image_root = scene_path

    auxiliary_root = scene_path if _path_is_within(image_root, scene_path) else image_root
    return image_root, auxiliary_root

def _build_frame_path_candidates(root, file_path, extension):
    normalized_rel = file_path[2:] if file_path.startswith("./") else file_path
    stem, suffix = os.path.splitext(normalized_rel)
    basename = os.path.basename(normalized_rel)
    candidates = [
        os.path.join(root, normalized_rel),
        os.path.join(root, basename),
        os.path.join(root, "images", basename),
    ]
    if not suffix and extension:
        with_extension = stem + extension
        basename_with_extension = os.path.basename(with_extension)
        candidates.extend([
            os.path.join(root, with_extension),
            os.path.join(root, basename_with_extension),
            os.path.join(root, "images", basename_with_extension),
        ])
    return _dedupe_preserve_order(candidates)

def _resolve_frame_image_path(scene_path, file_path, extension, image_root=None):
    candidate_roots = _dedupe_preserve_order([scene_path, image_root])
    for root in candidate_roots:
        for candidate in _build_frame_path_candidates(root, file_path, extension):
            if os.path.exists(candidate):
                return candidate
    raise FileNotFoundError(
        f"Could not resolve frame path '{file_path}' under scene '{scene_path}'"
        + (f" or image root '{image_root}'." if image_root else ".")
    )

def _resolve_auxiliary_tensor_paths(auxiliary_root, image_name, need_features=False, need_masks=False, mask_scales_dir_name="mask_scales"):
    features_path = None
    masks_path = None
    mask_scales_path = None

    if need_features:
        candidate = os.path.join(auxiliary_root, "clip_features", image_name + ".pt")
        if os.path.exists(candidate):
            features_path = candidate

    if need_masks:
        candidate = os.path.join(auxiliary_root, "sam_masks", image_name + ".pt")
        if os.path.exists(candidate):
            masks_path = candidate

        candidate = os.path.join(auxiliary_root, mask_scales_dir_name, image_name + ".pt")
        if os.path.exists(candidate):
            mask_scales_path = candidate

    return features_path, masks_path, mask_scales_path

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder, features_folder = None, masks_folder = None, mask_scale_folder = None, sample_rate = 1.0, allow_principle_point_shift = False):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        if idx % 10 >= sample_rate * 10:
            continue
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write(f"Reading camera {idx+1}/{len(cam_extrinsics)}")
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width
        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "SIMPLE_RADIAL":
            focal_length = intr.params[0]
            FovY = focal2fov(focal_length, height)
            FovX = focal2fov(focal_length, width)
        else:
            assert False, f"Colmap camera model {intr.model} not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        features_path = os.path.join(features_folder, image_name.split('.')[0] + ".pt") if features_folder is not None else None
        masks_path = os.path.join(masks_folder, image_name.split('.')[0] + ".pt") if masks_folder is not None else None
        mask_scales_path = os.path.join(mask_scale_folder, image_name.split('.')[0] + ".pt") if mask_scale_folder is not None else None

        features = torch.load(features_path, map_location='cpu') if features_path is not None else None
        masks = None
        mask_scales = torch.load(mask_scales_path, map_location='cpu') if mask_scales_path is not None else None

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image, features=features, masks=masks, mask_scales = mask_scales,
                              image_path=image_path, image_name=image_name, width=width, height=height, cx=intr.params[2] if len(intr.params) > 3 and allow_principle_point_shift else None, cy=intr.params[3] if len(intr.params) >3 and allow_principle_point_shift else None,
                              features_path=features_path, masks_path=masks_path, mask_scales_path=mask_scales_path)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path, only_xyz=False):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors, normals = None, None
    if not only_xyz:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def createRandomPointCloudFromCameraHull(train_cam_infos, num_pts=100_000, radius_scale=1.0, min_radius=0.05):
    cam_centers = []
    for cam in train_cam_infos:
        world_to_view = getWorld2View2(cam.R, cam.T)
        cam_to_world = np.linalg.inv(world_to_view)
        cam_centers.append(cam_to_world[:3, 3])

    cam_centers = np.stack(cam_centers, axis=0)
    center = cam_centers.mean(axis=0)
    base_radius = np.linalg.norm(cam_centers - center, axis=1).max()
    init_radius = max(base_radius * radius_scale, min_radius)

    xyz = (np.random.random((num_pts, 3)) * 2.0 - 1.0) * init_radius + center
    shs = np.random.random((num_pts, 3)) / 255.0
    return xyz, SH2RGB(shs) * 255

def readColmapSceneInfo(path, images, eval, llffhold=8, need_features=False, need_masks=False, sample_rate = 1.0, allow_principle_point_shift = False, replica=False, mask_scales_dir_name="mask_scales"):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    feature_dir = "clip_features"
    mask_dir = "sam_masks"
    mask_scale_dir = mask_scales_dir_name

    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir), features_folder=os.path.join(path, feature_dir) if need_features else None, masks_folder=os.path.join(path, mask_dir) if need_masks else None, mask_scale_folder=os.path.join(path, mask_scale_dir) if need_masks else None, sample_rate=sample_rate, allow_principle_point_shift = allow_principle_point_shift)

    if not replica:
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    else:
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : int(x.image_name.split("_")[-1]))

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", image_root=None, auxiliary_root=None, need_features=False, need_masks=False, mask_scales_dir_name="mask_scales"):
    cam_infos = []
    auxiliary_root = auxiliary_root or path

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            image_path = _resolve_frame_image_path(path, frame["file_path"], extension, image_root=image_root)
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_name = Path(image_path).stem
            image = Image.open(image_path)
            features_path, masks_path, mask_scales_path = _resolve_auxiliary_tensor_paths(
                auxiliary_root,
                image_name,
                need_features=need_features,
                need_masks=need_masks,
                mask_scales_dir_name=mask_scales_dir_name,
            )

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            features=None, masks=None, mask_scales=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            features_path=features_path, masks_path=masks_path, mask_scales_path=mask_scales_path))
            
    return cam_infos

def _resolve_lerf_image_path(scene_path, file_path, extension, image_root=None):
    return _resolve_frame_image_path(scene_path, file_path, extension, image_root=image_root)


def readCamerasFromLerfTransforms(path, transformsfile, white_background, extension=".jpg", image_root=None, auxiliary_root=None, need_features=False, need_masks=False, mask_scales_dir_name="mask_scales"):
    cam_infos = []
    auxiliary_root = auxiliary_root or path

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            tmp = np.array(frame["transform_matrix"])
            tmp_R = tmp[:3,:3]
            tmp_R = -tmp_R
            tmp_R[:,0] = -tmp_R[:,0]
            tmp[:3,:3] = tmp_R
            matrix = np.linalg.inv(tmp)

            R = np.transpose(matrix[:3,:3])
            T = matrix[:3, 3]

            image_path = _resolve_lerf_image_path(path, frame["file_path"], extension, image_root=image_root)
            image_name = Path(image_path).stem
            image = Image.open(image_path)
            features_path, masks_path, mask_scales_path = _resolve_auxiliary_tensor_paths(
                auxiliary_root,
                image_name,
                need_features=need_features,
                need_masks=need_masks,
                mask_scales_dir_name=mask_scales_dir_name,
            )

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")

            fovx = 2 * np.arctan(frame['w'] / (2 * frame['fl_x']))
            fovy = 2 * np.arctan(frame['h'] / (2 * frame['fl_y']))

            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            features=None, masks=None, mask_scales=None,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            cx=frame.get("cx"), cy=frame.get("cy"),
                            features_path=features_path, masks_path=masks_path, mask_scales_path=mask_scales_path))

    return cam_infos

def readNerfSyntheticInfo(path, images, white_background, eval, need_features=False, need_masks=False, extension=".png", init_radius_scale=1.0, init_min_radius=0.05, mask_scales_dir_name="mask_scales"):
    image_root, auxiliary_root = _resolve_image_and_aux_roots(path, images)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_train.json",
        white_background,
        extension,
        image_root=image_root,
        auxiliary_root=auxiliary_root,
        need_features=need_features,
        need_masks=need_masks,
        mask_scales_dir_name=mask_scales_dir_name,
    )
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(
        path,
        "transforms_test.json",
        white_background,
        extension,
        image_root=image_root,
        auxiliary_root=auxiliary_root,
        need_features=need_features,
        need_masks=need_masks,
        mask_scales_dir_name=mask_scales_dir_name,
    )
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz, rgb = createRandomPointCloudFromCameraHull(train_cam_infos, num_pts=num_pts, radius_scale=init_radius_scale, min_radius=init_min_radius)
        print(f"Random init radius scale: {init_radius_scale}, min radius: {init_min_radius}")
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readLerfInfo(path, images, white_background, eval, need_features=False, need_masks=False, extension=".jpg", init_radius_scale=1.0, init_min_radius=0.05, mask_scales_dir_name="mask_scales"):
    image_root, auxiliary_root = _resolve_image_and_aux_roots(path, images)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromLerfTransforms(
        path,
        "transforms.json",
        white_background,
        extension,
        image_root=image_root,
        auxiliary_root=auxiliary_root,
        need_features=need_features,
        need_masks=need_masks,
        mask_scales_dir_name=mask_scales_dir_name,
    )
    test_cam_infos = []
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        xyz, rgb = createRandomPointCloudFromCameraHull(train_cam_infos, num_pts=num_pts, radius_scale=init_radius_scale, min_radius=init_min_radius)
        print(f"Random init radius scale: {init_radius_scale}, min radius: {init_min_radius}")
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Lerf" : readLerfInfo
}
