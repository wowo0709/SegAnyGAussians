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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, fetchPly
from scene.gaussian_model import GaussianModel
try:
    from scene.gaussian_model_ff import FeatureGaussianModel
except Exception:
    FeatureGaussianModel = None
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


def _resolve_scene_checkpoint_path(model_path, iteration):
    iteration_root = os.path.join(model_path, "point_cloud", f"iteration_{iteration}")
    for filename in ("scene_point_cloud.ply", "point_cloud.ply"):
        candidate = os.path.join(iteration_root, filename)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"Could not find a scene checkpoint under {iteration_root}. Expected one of scene_point_cloud.ply or point_cloud.ply."
    )


def _candidate_feature_model_paths(model_path, feature_model_path):
    roots = []
    for candidate in (feature_model_path, model_path):
        if candidate and candidate not in roots:
            roots.append(candidate)
    return roots


def _search_for_feature_iteration(model_paths, target):
    for root in model_paths:
        point_cloud_root = os.path.join(root, "point_cloud")
        if not os.path.isdir(point_cloud_root):
            continue
        iteration = searchForMaxIteration(point_cloud_root, target=target)
        if iteration is not None:
            return iteration
    return None


def _resolve_feature_artifact(model_paths, iteration, filename):
    checked_paths = []
    for root in model_paths:
        candidate = os.path.join(root, "point_cloud", f"iteration_{iteration}", filename)
        checked_paths.append(candidate)
        if os.path.exists(candidate):
            return root, candidate
    checked_preview = "\n  ".join(checked_paths[:8]) if checked_paths else "(no feature roots checked)"
    if len(checked_paths) > 8:
        checked_preview += "\n  ..."
    raise FileNotFoundError(
        f"Could not find feature artifact {filename} at iteration {iteration}. Checked:\n  {checked_preview}"
    )


def _resolve_scene_gaussian_backend(requested_backend, *models):
    resolved_backend = None
    for model in models:
        if model is None:
            continue
        current_backend = model.resolve_gaussian_backend(requested_backend)
        if resolved_backend is None:
            resolved_backend = current_backend
        elif current_backend != resolved_backend:
            raise RuntimeError(
                f"Loaded Gaussian models resolved to different backends: '{resolved_backend}' vs '{current_backend}'."
            )
    if resolved_backend is not None:
        return resolved_backend
    return "3dgs" if requested_backend == "auto" else requested_backend


class Scene:

    gaussians : GaussianModel
    feature_gaussians : FeatureGaussianModel

    # target: feature, seg, scene
    def __init__(self, args : ModelParams, gaussians : GaussianModel=None, feature_gaussians: FeatureGaussianModel=None, load_iteration=None, feature_load_iteration=None, shuffle=True, resolution_scales=[1.0], init_from_3dgs_pcd=False, target='scene', mode='train', sample_rate = 1.0):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.feature_model_path = getattr(args, "feature_model_path", None) or self.model_path
        self.feature_model_search_roots = _candidate_feature_model_paths(self.model_path, self.feature_model_path)
        self.feature_artifact_root = None
        self.loaded_iter = None
        self.feature_loaded_iter = None
        self.gaussians = gaussians
        self.feature_gaussians = feature_gaussians

        if load_iteration:
            if load_iteration == -1:
                if mode == 'train':
                    # only load feature gaussians when doing segmentation
                    if target == 'seg' or target == 'coarse_seg_everything':
                        self.feature_loaded_iter = _search_for_feature_iteration(self.feature_model_search_roots, target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'scene':
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target == 'feature' or target == 'contrastive_feature':
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'seg':
                        self.feature_loaded_iter = None
                        self.feature_gaussians = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="seg")
                    elif target == 'scene':
                        self.feature_gaussians = None
                        self.feature_loaded_iter = None
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    elif target in ['feature', 'contrastive_feature']:
                        self.feature_loaded_iter = _search_for_feature_iteration(self.feature_model_search_roots, target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        self.loaded_iter = -1 if gaussians is None else searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target='scene')
                    elif target == 'coarse_seg_everything':
                        self.feature_loaded_iter = _search_for_feature_iteration(self.feature_model_search_roots, target=target)
                        self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene")
                    else:
                        assert False and "Unknown target!"
            else:
                self.loaded_iter = load_iteration
                if mode == 'train':
                    if target == 'seg' or target == 'coarse_seg_everything':
                        self.feature_loaded_iter = _search_for_feature_iteration(self.feature_model_search_roots, target="feature") if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                    elif target == 'scene' or 'feature' in target:
                        self.feature_loaded_iter = None
                    else:
                        assert False and "Unknown target!"
                elif mode == 'eval':
                    if target == 'seg':
                        self.feature_loaded_iter = None
                        self.feature_gaussians = None
                    elif target == 'scene':
                        self.feature_gaussians = None
                        self.feature_loaded_iter = None
                    elif target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature':
                        self.feature_loaded_iter = _search_for_feature_iteration(self.feature_model_search_roots, target=target) if (feature_load_iteration is None or feature_load_iteration == -1) else feature_load_iteration
                        # Respect an explicit scene iteration override during evaluation.
                        # Contrastive-feature eval still needs the scene checkpoint for SH0,
                        # camera metadata, and renderer-side geometry.
                        self.loaded_iter = load_iteration
                    else:
                        assert False and "Unknown target!"

            print("Loading trained model at iteration {}, {}".format(self.loaded_iter, self.feature_loaded_iter))
            
        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")) and not os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print(f"Allow Camera Principle Point Shift: {args.allow_principle_point_shift}")
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval, need_features = args.need_features, need_masks = args.need_masks, sample_rate = sample_rate, allow_principle_point_shift = args.allow_principle_point_shift, replica = 'replica' in args.model_path)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval, init_radius_scale=args.init_radius_scale, init_min_radius=args.init_min_radius)
        elif os.path.exists(os.path.join(args.source_path, "transforms.json")):
            print("Found transforms.json file, assuming Lerf data set!")
            scene_info = sceneLoadTypeCallbacks["Lerf"](args.source_path, args.white_background, args.eval, init_radius_scale=args.init_radius_scale, init_min_radius=args.init_min_radius)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(self.model_path, "input.ply") , 'wb') as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)

        # Load or initialize scene / seg gaussians
        if self.loaded_iter and self.gaussians is not None:
            if mode == 'train':
                self.gaussians.load_ply(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))
            else:
                if target == 'coarse_seg_everything':
                    self.gaussians.load_ply(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))
                elif 'feature' not in target:
                    self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            target+"_point_cloud.ply"))
                else:
                    self.gaussians.load_ply(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))

        elif self.gaussians is not None:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        # Load or initialize feature gaussians
        if self.feature_loaded_iter and self.feature_gaussians is not None:
            if target == 'feature' or target == 'seg':
                self.feature_artifact_root, feature_path = _resolve_feature_artifact(
                    self.feature_model_search_roots,
                    self.feature_loaded_iter,
                    "feature_point_cloud.ply",
                )
                self.feature_gaussians.load_ply(feature_path)
            elif target == 'coarse_seg_everything':
                if mode == 'train':
                    self.feature_gaussians.load_ply_from_3dgs(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))
                elif mode == 'eval':
                    self.feature_artifact_root, feature_path = _resolve_feature_artifact(
                        self.feature_model_search_roots,
                        self.feature_loaded_iter,
                        "coarse_seg_everything_point_cloud.ply",
                    )
                    self.feature_gaussians.load_ply(feature_path)
            elif target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_gaussians.load_ply_from_3dgs(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))
                elif mode == 'eval':
                    self.feature_artifact_root, feature_path = _resolve_feature_artifact(
                        self.feature_model_search_roots,
                        self.feature_loaded_iter,
                        "contrastive_feature_point_cloud.ply",
                    )
                    self.feature_gaussians.load_ply(feature_path)


        elif self.feature_gaussians is not None:
            if target=='feature' and init_from_3dgs_pcd:
                print("Initialize feature gaussians from 3DGS point cloud")
                resolved_scene_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"), target="scene") if (self.loaded_iter is None or self.loaded_iter == -1) else self.loaded_iter
                self.feature_gaussians.create_from_pcd(
                    fetchPly(
                        _resolve_scene_checkpoint_path(self.model_path, resolved_scene_iter),
                        only_xyz=True,
                    ),
                    self.cameras_extent
                )
            elif target == 'contrastive_feature':
                if mode == 'train':
                    self.feature_gaussians.load_ply_from_3dgs(_resolve_scene_checkpoint_path(self.model_path, self.loaded_iter))
                elif mode == 'eval':
                    self.feature_artifact_root, feature_path = _resolve_feature_artifact(
                        self.feature_model_search_roots,
                        self.feature_loaded_iter,
                        "contrastive_feature_point_cloud.ply",
                    )
                    self.feature_gaussians.load_ply(feature_path)
            else:
                print("Initialize feature gaussians from Colmap point cloud")
                self.feature_gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        self.gaussian_backend = _resolve_scene_gaussian_backend(
            getattr(args, "gaussian_backend", "auto"),
            self.gaussians,
            self.feature_gaussians,
        )
        print(f"Resolved Gaussian backend: {self.gaussian_backend} (requested: {getattr(args, 'gaussian_backend', 'auto')})")


    def save(self, iteration, target='scene'):
        assert target != 'feature' and "Please use save_feature() to save feature gaussians!"
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, target+"_point_cloud.ply"))

    def save_mask(self, iteration, id = 0):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_mask(os.path.join(point_cloud_path, f"seg_point_cloud_{id}.npy"))

    def save_feature(self, iteration, target = 'coarse_seg_everything', smooth_weights = None, smooth_type = None, smooth_K = 16):
        assert self.feature_gaussians is not None and (target == 'feature' or target == 'coarse_seg_everything' or target == 'contrastive_feature')
        point_cloud_path = os.path.join(self.feature_model_path, "point_cloud/iteration_{}".format(iteration))
        os.makedirs(point_cloud_path, exist_ok=True)
        self.feature_gaussians.save_ply(os.path.join(point_cloud_path, f"{target}_point_cloud.ply"), smooth_weights, smooth_type, smooth_K)


    # def save_coarse_seg_everything(self, iteration):
    #     assert self.feature_gaussians is not None
    #     point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
    #     self.feature_gaussians.save_ply(os.path.join(point_cloud_path, "coarse_seg_everything_point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
