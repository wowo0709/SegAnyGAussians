# Borrowed from OmniSeg3D-GS (https://github.com/OceanYing/OmniSeg3D-GS)
import torch
from scene import Scene
import hashlib
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render, render_contrastive_feature
import torchvision
from utils.general_utils import safe_state, build_rotation
from argparse import ArgumentParser, ArgumentTypeError
from arguments import ModelParams, PipelineParams, get_combined_args
# from gaussian_renderer import GaussianModel
import numpy as np
from PIL import Image
import colorsys
import cv2
from plyfile import PlyData, PlyElement
import re

# from scene.gaussian_model import GaussianModel
from scene import Scene, GaussianModel, FeatureGaussianModel
import dearpygui.dearpygui as dpg
import math
import threading
import time
from scene.cameras import Camera
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.system_utils import searchForMaxIteration

from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial.transform import Rotation as R

from utils.gaussian_graph_cluster import (
    cluster_gaussians_hdbscan,
    cluster_gaussians_hdbscan_refined,
    cluster_gaussians_normalized_cut,
    load_mesh_prior_for_points,
)
from utils.sh_utils import SH2RGB

def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min() + 1e-7)
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)
    return depth_img

class CONFIG:
    r = 2   # scale ratio
    window_width = int(2160/r)
    window_height = int(1200/r)

    width = int(2160/r)
    height = int(1200/r)

    radius = 2

    debug = False
    dt_gamma = 0.2

    # gaussian model
    sh_degree = 3

    convert_SHs_python = False
    compute_cov3D_python = False

    white_background = False

    FEATURE_DIM = 32
    MODEL_PATH = './output/figurines' # 30000
    FEATURE_MODEL_PATH = ""

    FEATURE_GAUSSIAN_ITERATION = 10000
    SCENE_GAUSSIAN_ITERATION = 30000

    SCALE_GATE_PATH = ""
    FEATURE_PCD_PATH = ""
    SCENE_PCD_PATH = os.path.join(MODEL_PATH, f'point_cloud/iteration_{str(SCENE_GAUSSIAN_ITERATION)}/scene_point_cloud.ply')

    CLUSTER_METHOD = "HDBSCANRefined"
    HIDE_SMALL_RESIDUE = True
    RESIDUE_SCOPE = "Both"
    CUBOID_PERCENTILE = 100.0


def _list_available_iterations(point_cloud_root):
    iterations = []
    if not os.path.isdir(point_cloud_root):
        return iterations

    for entry in os.listdir(point_cloud_root):
        if not entry.startswith("iteration_"):
            continue
        try:
            iterations.append(int(entry.split("_")[-1]))
        except ValueError:
            continue

    iterations.sort(reverse=True)
    return iterations


def _iteration_candidates(model_path, requested_iteration=None, search_target=None):
    point_cloud_root = os.path.join(model_path, "point_cloud")
    candidates = []
    seen = set()

    def add_candidate(iteration):
        if iteration is None:
            return
        try:
            iteration = int(iteration)
        except (TypeError, ValueError):
            return
        if iteration not in seen:
            seen.add(iteration)
            candidates.append(iteration)

    add_candidate(requested_iteration)
    if search_target is not None:
        add_candidate(searchForMaxIteration(point_cloud_root, target=search_target))

    for iteration in _list_available_iterations(point_cloud_root):
        add_candidate(iteration)

    return candidates


def _resolve_scene_pcd(model_path, requested_iteration=None):
    point_cloud_root = os.path.join(model_path, "point_cloud")
    if not os.path.isdir(point_cloud_root):
        raise FileNotFoundError(f"Point cloud directory does not exist: {point_cloud_root}")

    checked_paths = []
    for iteration in _iteration_candidates(model_path, requested_iteration=requested_iteration, search_target="scene"):
        iteration_root = os.path.join(point_cloud_root, f"iteration_{iteration}")
        for filename in ("scene_point_cloud.ply", "point_cloud.ply"):
            scene_path = os.path.join(iteration_root, filename)
            checked_paths.append(scene_path)
            if os.path.exists(scene_path):
                return iteration, scene_path

    checked_preview = "\n  ".join(checked_paths[:8]) if checked_paths else "(no iteration directories found)"
    if len(checked_paths) > 8:
        checked_preview += "\n  ..."
    raise FileNotFoundError(
        f"Could not find a scene point cloud under {point_cloud_root}. Checked:\n  {checked_preview}"
    )


def _read_feature_model_path_hint(model_path):
    hint_path = os.path.join(model_path, "feature_model_path.txt")
    if not os.path.isfile(hint_path):
        return ""
    try:
        with open(hint_path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return ""


def _read_namespace_path_hint(model_path, cfg_filename, field_name):
    cfg_path = os.path.join(model_path, cfg_filename)
    if not os.path.isfile(cfg_path):
        return ""
    try:
        with open(cfg_path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return ""

    patterns = (
        rf"{re.escape(field_name)}='([^']+)'",
        rf'{re.escape(field_name)}="([^"]+)"',
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def _candidate_scene_model_paths(model_path, feature_model_path=""):
    roots = []

    def add_candidate(candidate):
        if not candidate:
            return
        candidate = os.path.abspath(candidate)
        if candidate not in roots:
            roots.append(candidate)

    def add_parent_chain(candidate, max_depth=4):
        current = os.path.abspath(candidate)
        for _ in range(max_depth):
            parent = os.path.dirname(current)
            if not parent or parent == current:
                break
            add_candidate(parent)
            current = parent

    for seed in (model_path, feature_model_path):
        if not seed:
            continue
        add_candidate(seed)
        add_candidate(_read_namespace_path_hint(seed, "feature_cfg_args", "model_path"))
        add_candidate(_read_namespace_path_hint(seed, "cfg_args", "model_path"))
        add_parent_chain(seed)

    return roots


def _resolve_scene_bundle(model_path, requested_iteration=None, feature_model_path=""):
    checked_paths = []
    checked_roots = []
    for scene_root in _candidate_scene_model_paths(model_path, feature_model_path):
        point_cloud_root = os.path.join(scene_root, "point_cloud")
        if not os.path.isdir(point_cloud_root):
            checked_roots.append(point_cloud_root)
            continue

        for iteration in _iteration_candidates(scene_root, requested_iteration=requested_iteration, search_target="scene"):
            iteration_root = os.path.join(point_cloud_root, f"iteration_{iteration}")
            for filename in ("scene_point_cloud.ply", "point_cloud.ply"):
                scene_path = os.path.join(iteration_root, filename)
                checked_paths.append(scene_path)
                if os.path.exists(scene_path):
                    return scene_root, iteration, scene_path

    checked_preview = "\n  ".join((checked_paths + checked_roots)[:8]) if (checked_paths or checked_roots) else "(no scene root candidates found)"
    if len(checked_paths) + len(checked_roots) > 8:
        checked_preview += "\n  ..."
    raise FileNotFoundError(
        "Could not find a scene point cloud from the provided model_path / feature root candidates. "
        f"Checked:\n  {checked_preview}"
    )


def _candidate_feature_model_paths(model_path, feature_model_path=""):
    roots = []
    for candidate in (feature_model_path, _read_feature_model_path_hint(model_path), model_path, os.path.join(model_path, "feature_model")):
        if candidate:
            candidate = os.path.abspath(candidate)
            if candidate not in roots:
                roots.append(candidate)
    return roots


def _resolve_feature_bundle(model_path, requested_iteration=None, feature_model_path=""):
    checked_pairs = []
    for feature_root in _candidate_feature_model_paths(model_path, feature_model_path):
        point_cloud_root = os.path.join(feature_root, "point_cloud")
        if not os.path.isdir(point_cloud_root):
            continue
        for iteration in _iteration_candidates(feature_root, requested_iteration=requested_iteration, search_target="feature"):
            iteration_root = os.path.join(point_cloud_root, f"iteration_{iteration}")
            feature_path = os.path.join(iteration_root, "contrastive_feature_point_cloud.ply")
            scale_gate_path = os.path.join(iteration_root, "scale_gate.pt")
            checked_pairs.append((feature_path, scale_gate_path))
            if os.path.exists(feature_path) and os.path.exists(scale_gate_path):
                return feature_root, iteration, feature_path, scale_gate_path

    checked_preview = "\n  ".join(
        f"{feature_path} | {scale_gate_path}" for feature_path, scale_gate_path in checked_pairs[:8]
    ) if checked_pairs else "(no iteration directories found)"
    if len(checked_pairs) > 8:
        checked_preview += "\n  ..."
    raise FileNotFoundError(
        f"Could not find a matching feature point cloud and scale gate under {point_cloud_root}. Checked:\n  {checked_preview}"
    )


def _resolve_model_artifacts(opt):
    requested_model_path = os.path.abspath(opt.MODEL_PATH)
    requested_feature_model_path = os.path.abspath(opt.FEATURE_MODEL_PATH) if getattr(opt, "FEATURE_MODEL_PATH", "") else ""

    scene_root, scene_iteration, scene_pcd_path = _resolve_scene_bundle(
        requested_model_path,
        requested_iteration=opt.SCENE_GAUSSIAN_ITERATION,
        feature_model_path=requested_feature_model_path,
    )
    feature_search_root = requested_feature_model_path or requested_model_path
    feature_root, feature_iteration, feature_pcd_path, scale_gate_path = _resolve_feature_bundle(
        scene_root,
        requested_iteration=opt.FEATURE_GAUSSIAN_ITERATION,
        feature_model_path=feature_search_root,
    )

    opt.MODEL_PATH = scene_root
    opt.SCENE_GAUSSIAN_ITERATION = scene_iteration
    opt.FEATURE_GAUSSIAN_ITERATION = feature_iteration
    opt.FEATURE_MODEL_PATH = feature_root
    opt.SCENE_PCD_PATH = scene_pcd_path
    opt.FEATURE_PCD_PATH = feature_pcd_path
    opt.SCALE_GATE_PATH = scale_gate_path

    print(f"Resolved scene root: {scene_root}")
    print(f"Using scene iteration {scene_iteration}: {scene_pcd_path}")
    print(f"Using feature root: {feature_root}")
    print(f"Using feature iteration {feature_iteration}: {feature_pcd_path}")
    print(f"Using scale gate: {scale_gate_path}")


def _load_scale_gate_state_dict(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class OrbitCamera:
    def __init__(self, W, H, r=2, fovy=60):
        self.W = W
        self.H = H
        self.radius = r  # camera distance from center
        self.center = np.array([0, 0, 0], dtype=np.float32)  # look at this point
        self.rot = R.from_quat(
            [0, 0, 0, 1]
        )  # init camera matrix: [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

        self.up = np.array([0, 1, 0], dtype=np.float32)  # need to be normalized!
        self.right = np.array([1, 0, 0], dtype=np.float32)  # need to be normalized!
        self.fovy = fovy
        self.translate = np.array([0, 0, self.radius])
        self.scale_f = 1.0


        self.rot_mode = 1   # rotation mode (1: self.pose_movecenter (movable rotation center), 0: self.pose_objcenter (fixed scene center))
        # self.rot_mode = 0


    @property
    def pose_movecenter(self):
        # --- first move camera to radius : in world coordinate--- #
        res = np.eye(4, dtype=np.float32)
        res[2, 3] -= self.radius
        
        # --- rotate: Rc --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tc --- #
        res[:3, 3] -= self.center
        
        # --- Convention Transform --- #
        # now we have got matrix res=c2w=[Rc|tc], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]
        res[:3, 3] = -rot[:3, :3].transpose() @ res[:3, 3]
        
        return res
    
    @property
    def pose_objcenter(self):
        res = np.eye(4, dtype=np.float32)
        
        # --- rotate: Rw --- #
        rot = np.eye(4, dtype=np.float32)
        rot[:3, :3] = self.rot.as_matrix()
        res = rot @ res

        # --- translate: tw --- #
        res[2, 3] += self.radius    # camera coordinate z-axis
        res[:3, 3] -= self.center   # camera coordinate x,y-axis
        
        # --- Convention Transform --- #
        # now we have got matrix res=w2c=[Rw|tw], but gaussian-splatting requires convention as [Rc|-Rc.T@tc]=[Rw.T|tw]
        res[:3, :3] = rot[:3, :3].transpose()
        
        return res

    @property
    def opt_pose(self):
        # --- deprecated ! Not intuitive implementation --- #
        res = np.eye(4, dtype=np.float32)

        res[:3, :3] = self.rot.as_matrix()

        scale_mat = np.eye(4)
        scale_mat[0, 0] = self.scale_f      # why apply scale ratio to rotation matrix? It's confusing.
        scale_mat[1, 1] = self.scale_f
        scale_mat[2, 2] = self.scale_f

        transl = self.translate - self.center
        transl_mat = np.eye(4)
        transl_mat[:3, 3] = transl

        return transl_mat @ scale_mat @ res

    # intrinsics
    @property
    def intrinsics(self):
        focal = self.H / (2 * np.tan(np.radians(self.fovy) / 2))
        return np.array([focal, focal, self.W // 2, self.H // 2])

    def orbit(self, dx, dy):
        if self.rot_mode == 1:    # rotate the camera axis, in world coordinate system
            up = self.rot.as_matrix()[:3, 1]
            side = self.rot.as_matrix()[:3, 0]
        elif self.rot_mode == 0:    # rotate in camera coordinate system
            up = -self.up
            side = -self.right
        rotvec_x = up * np.radians(0.01 * dx)
        rotvec_y = side * np.radians(0.01 * dy)

        self.rot = R.from_rotvec(rotvec_x) * R.from_rotvec(rotvec_y) * self.rot

    def scale(self, delta):
        # self.radius *= 1.1 ** (-delta)    # non-linear version
        self.radius -= 0.1 * delta      # linear version

    def pan(self, dx, dy, dz=0):
        
        if self.rot_mode == 1:
            # pan in camera coordinate system: project from [Coord_c] to [Coord_w]
            self.center += 0.0005 * self.rot.as_matrix()[:3, :3] @ np.array([dx, -dy, dz])
        elif self.rot_mode == 0:
            # pan in world coordinate system: at [Coord_w]
            self.center += 0.0005 * np.array([-dx, dy, dz])


class GaussianSplattingGUI:
    def __init__(self, opt, gaussian_model:GaussianModel, feature_gaussian_model:FeatureGaussianModel, scale_gate: torch.nn.modules.container.Sequential) -> None:
        self.opt = opt

        self.width = opt.width
        self.height = opt.height
        self.window_width = opt.window_width
        self.window_height = opt.window_height
        self.camera = OrbitCamera(opt.width, opt.height, r=opt.radius)

        bg_color = [1, 1, 1] if opt.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        bg_feature = [0 for i in range(opt.FEATURE_DIM)]
        bg_feature = torch.tensor(bg_feature, dtype=torch.float32, device="cuda")

        self.bg_color = background
        self.bg_feature = bg_feature
        self.render_buffer = np.zeros((self.width, self.height, 3), dtype=np.float32)
        self.update_camera = True
        self.dynamic_resolution = True
        self.debug = opt.debug
        self.engine = {
            'scene': gaussian_model,
            'feature': feature_gaussian_model,
            'scale_gate': scale_gate
        }

        self.cluster_point_colors = None
        self.cluster_method = getattr(opt, "CLUSTER_METHOD", "HDBSCANRefined")
        self.cluster_sample_size = int(getattr(opt, "CLUSTER_SAMPLE_SIZE", 20000))
        self.cluster_graph_k = int(getattr(opt, "CLUSTER_GRAPH_K", 16))
        self.cluster_max_clusters = int(getattr(opt, "CLUSTER_MAX_CLUSTERS", 24))
        self.cluster_min_cluster_size = int(getattr(opt, "CLUSTER_MIN_CLUSTER_SIZE", 128))
        self.cluster_cut_threshold = float(getattr(opt, "CLUSTER_CUT_THRESHOLD", 0.12))
        self.cluster_spatial_weight = float(getattr(opt, "CLUSTER_SPATIAL_WEIGHT", 1.0))
        self.cluster_sh0_color_weight = float(getattr(opt, "CLUSTER_SH0_COLOR_WEIGHT", 0.0))
        self.cluster_sh0_color_sigma = float(getattr(opt, "CLUSTER_SH0_COLOR_SIGMA", 0.25))
        self.cluster_mesh_weight = float(getattr(opt, "CLUSTER_MESH_WEIGHT", 0.0))
        self.cluster_mesh_path = getattr(opt, "CLUSTER_MESH_PATH", None)
        self.hide_small_residue = bool(getattr(opt, "HIDE_SMALL_RESIDUE", True))
        self.residue_max_size = int(getattr(opt, "RESIDUE_MAX_SIZE", 1000))
        self.residue_scope = str(getattr(opt, "RESIDUE_SCOPE", "Both"))
        self.exclude_residue_on_active_export = bool(getattr(opt, "EXCLUDE_RESIDUE_ON_ACTIVE_EXPORT", True))
        self.cluster_mesh_prior = None
        self.label_to_color = np.random.rand(1000, 3)
        self.label_to_code = self._generate_label_codes(1000)
        self.seg_score = None

        self.proj_mat = None

        self.load_model = False
        print("loading model file...")
        self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
        self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
        self.engine['scale_gate'].load_state_dict(_load_scale_gate_state_dict(self.opt.SCALE_GATE_PATH))
        self._validate_scene_feature_alignment(context="initial load")
        self.do_pca()   # calculate self.proj_mat
        self.load_model = True

        print("loading model file done.")

        self.mode = "image"  # choose from ['image', 'depth']
        self.cuboid_manipulation = False
        self.show_active = True
        self.preview_max_gaussians = 200000
        self.cuboid_percentile = float(getattr(opt, "CUBOID_PERCENTILE", 100.0))
        self.transform_mode = "Rotate"
        self.extend_handle_tolerance_px = 12.0
        self.face_handle_tolerance_px = 18.0
        self.extend_handle = None
        self.render_mode_rgb = True
        self.render_mode_pca = False
        self.render_mode_cluster = False
        self.edit_click_select = True
        self.edit_multiclick = True

        dpg.create_context()
        self.register_dpg()

        self.frame_id = 0

        # --- for better operation --- #
        self.moving = False
        self.moving_middle = False
        self.mouse_pos = (0, 0)

        # --- for interactive segmentation --- #
        self.img_mode = 0
        self.clickmode_button = False
        self.clickmode_multi_button = False     # choose multiple object 
        self.new_click = False
        self.prompt_num = 0
        self.new_click_xy = []
        self.clear_edit = False                 # clear all the click prompts
        self.roll_back = False
        self.preview = False    # binary segmentation mode
        self.segment3d_flag = False
        self.reload_flag = False        # reload the whole scene / point cloud
        self.object_seg_id = 0          # to store the segmented object with increasing index order (path at: ./)
        self.cluster_in_3D_flag = False

        self.render_mode_rgb = True
        self.render_mode_similarity = False
        self.render_mode_pca = False
        self.render_mode_cluster = False

        self.save_flag = False

        # --- cluster editing phase 1 state --- #
        self.cluster_cache = None
        self.cluster_cache_state = "stale"
        self.cluster_cache_scale = None
        self.cluster_request_generation = 0
        self.cluster_pending_result = None
        self.cluster_job_error = None
        self.cluster_thread = None
        self.recluster_requested = False
        self.last_scale_value = None
        self.cluster_status_message = "stale: click Recluster"
        self.edit_click_select = True
        self.edit_marquee_select = False
        self.edit_multiclick = True
        self.pending_cluster_ids = []
        self.pending_cluster_seed_indices = {}
        self.pending_parts = []
        self.pending_part_counter = 0
        self.component_cache = {}
        self.marquee_request = None
        self.right_dragging = False
        self.right_drag_start_local_xy = None
        self.right_drag_current_local_xy = None
        self.marquee_overlay_rect = None
        self.marquee_threshold_px = 8.0
        self.active_group = None
        self.cluster_pick_request = None
        self.pick_tolerance_px = 8.0
        self.last_scene_outputs = None
        self.last_feature_outputs = None
        self.last_view_camera = None
        self.selection_debug_render = None
        self.selection_highlight_enabled = False
        self.last_score_threshold = None
        self.last_picked_cluster_id = None
        self.last_picked_seed_index = None
        self.last_pick_result = "none"
        self.last_pick_confidence = None
        self.cluster_pick_feature_render = None
        self.cluster_pick_patch_radius = 2
        self.hidden_opacity_logit = -20.0
        self.cuboid_manipulation = False
        self.show_active = True
        self.preview_max_gaussians = 200000
        self.cuboid_percentile = float(getattr(opt, "CUBOID_PERCENTILE", 100.0))
        self.connectivity_knn = 24
        self.connectivity_local_scale_neighbors = 4
        self.connectivity_scale = 2.5
        self.connectivity_timeout_sec = 10.0
        self.apply_active_requested = False
        self.cancel_active_requested = False
        self.left_dragging = False
        self.middle_dragging = False
        self.drag_mode = None
        self.drag_last_local_xy = None
        self.transform_mode = "Rotate"
        self.extend_handle_tolerance_px = 12.0
        self.face_handle_tolerance_px = 18.0
        self.extend_handle = None
        self.save_active_flag = False
        self.save_full_scene_flag = False
        self.apply_undo_stack = []
        self.apply_redo_stack = []
        self.max_apply_history = 20
        self.part_list_dirty = True
        self.part_row_signature = None

    def __del__(self):
        dpg.destroy_context()

    def prepare_buffer(self, outputs):
        if self.model == "images":
            return outputs["render"]
        else:
            return np.expand_dims(outputs["depth"], -1).repeat(3, -1)
    
    def grayscale_to_colormap(self, gray):
        """Convert a grayscale value to Jet colormap RGB values."""
        # Ensure the grayscale values are in the range [0, 1]
        # gray = np.clip(gray, 0, 1)

        # Jet colormap ranges (these are normalized to [0, 1])
        jet_colormap = np.array([
            [0, 0, 0.5],
            [0, 0, 1],
            [0, 0.5, 1],
            [0, 1, 1],
            [0.5, 1, 0.5],
            [1, 1, 0],
            [1, 0.5, 0],
            [1, 0, 0],
            [0.5, 0, 0]
        ])

        # Corresponding positions for the colors in the colormap
        positions = np.linspace(0, 1, jet_colormap.shape[0])

        # Interpolate the RGB values based on the grayscale value
        r = np.interp(gray, positions, jet_colormap[:, 0])
        g = np.interp(gray, positions, jet_colormap[:, 1])
        b = np.interp(gray, positions, jet_colormap[:, 2])

        return np.stack((r, g, b), axis=-1)


    def _selection_cluster_ids(self):
        if self._active_group_has_cuboid():
            return sorted({int(part["cluster_id"]) for part in self.active_group.get("parts", [])})
        return sorted({int(part["cluster_id"]) for part in self.pending_parts})

    def _part_display_color(self, part_index, active=False):
        hue = (0.1618 * float(part_index)) % 1.0
        saturation = 0.95 if not active else 0.75
        value = 1.0
        color = colorsys.hsv_to_rgb(hue, saturation, value)
        return np.asarray(color, dtype=np.float32)

    def _clone_part(self, part):
        cloned = {}
        for key, value in part.items():
            if torch.is_tensor(value):
                cloned[key] = value.detach().clone()
            else:
                cloned[key] = value
        return cloned

    def _filter_part_indices(self, part, score_thres):
        indices = part.get("source_indices")
        if indices is None:
            indices = part.get("global_indices")
        if indices is None or indices.numel() == 0:
            return None
        if self.cluster_cache is None or score_thres <= 0.0:
            return indices
        confidence = self.cluster_cache["confidence"][indices]
        filtered = indices[confidence >= float(score_thres)]
        if filtered.numel() == 0:
            return None
        global_hidden_mask = self._current_global_residue_hidden_mask()
        if global_hidden_mask is not None:
            filtered = filtered[~global_hidden_mask[filtered]]
            if filtered.numel() == 0:
                return None
        return filtered

    def _pending_selection_mask(self, score_thres):
        if self.cluster_cache is None or len(self.pending_parts) == 0:
            return None
        labels = self.cluster_cache["labels"]
        mask = torch.zeros_like(labels, dtype=torch.bool)
        for part in self.pending_parts:
            indices = self._filter_part_indices(part, score_thres)
            if indices is not None and indices.numel() > 0:
                mask[indices] = True
        if int(mask.sum().item()) == 0:
            return None
        return mask

    def _part_fingerprint(self, indices):
        arr = indices.detach().cpu().numpy().astype(np.int64, copy=False)
        return f"{arr.shape[0]}:{hashlib.blake2b(arr.tobytes(), digest_size=16).hexdigest()}"

    def _part_exists_in_pending(self, fingerprint):
        return any(part.get("fingerprint") == fingerprint for part in self.pending_parts)

    def _find_pending_part_index(self, fingerprint):
        for idx, part in enumerate(self.pending_parts):
            if part.get("fingerprint") == fingerprint:
                return idx
        return None

    def _find_active_source_part_index(self, fingerprint):
        if not self._active_group_has_cuboid():
            return None
        for idx, part in enumerate(self.active_group.get("source_parts", [])):
            if part.get("fingerprint") == fingerprint:
                return idx
        return None

    def _add_pending_part(self, part, clear_existing=False):
        if part is None:
            return "noop"
        fingerprint = part.get("fingerprint")
        if fingerprint is None:
            fingerprint = self._part_fingerprint(part["source_indices"])
            part["fingerprint"] = fingerprint

        if clear_existing:
            same_single_pending = (
                len(self.pending_parts) == 1
                and self.pending_parts[0].get("fingerprint") == fingerprint
            )
            self.pending_parts = []
            self._mark_part_list_dirty()
            if same_single_pending:
                self._update_cluster_status_widget()
                return "removed"

        existing_idx = self._find_pending_part_index(fingerprint)
        if existing_idx is not None:
            del self.pending_parts[existing_idx]
            self._mark_part_list_dirty()
            self._update_cluster_status_widget()
            return "removed"

        self.pending_part_counter += 1
        part["part_id"] = self.pending_part_counter
        self.pending_parts.append(part)
        self._mark_part_list_dirty()
        self._update_cluster_status_widget()
        return "added"

    def _clear_pending_selection(self):
        self.pending_cluster_ids = []
        self.pending_cluster_seed_indices = {}
        self.pending_parts = []
        self._mark_part_list_dirty()
        self._update_cluster_status_widget()

    def _part_display_count(self, part, active=False):
        score_thres = 0.0 if self.last_score_threshold is None else float(self.last_score_threshold)
        if active:
            indices = self._part_visible_indices(part, active=True)
        else:
            indices = self._filter_part_indices(part, score_thres)
        return 0 if indices is None else int(indices.numel())

    def _format_part_row_label(self, part, index, active=False):
        prefix = "A" if active else "P"
        cluster_id = int(part.get("cluster_id", -1))
        count = self._part_display_count(part, active=active)
        return f"{prefix}{index}: cluster {cluster_id} ({count})"

    def _mark_part_list_dirty(self):
        self.part_list_dirty = True

    def _pending_active_signature(self):
        active_parts = [] if self.active_group is None else self.active_group.get("parts", [])
        pending_parts = self.pending_parts
        pending_sig = tuple(part.get("fingerprint") for part in pending_parts)
        active_sig = tuple(part.get("fingerprint") for part in active_parts)
        return pending_sig, active_sig

    def _callback_pending_row_remove(self, sender, app_data, user_data):
        self._remove_pending_part_by_fingerprint(user_data["fingerprint"])

    def _callback_pending_row_move_to_active(self, sender, app_data, user_data):
        self._move_pending_part_to_active(user_data["fingerprint"])

    def _callback_active_row_remove(self, sender, app_data, user_data):
        self._remove_active_part_by_fingerprint(user_data["fingerprint"], move_to_pending=False)

    def _callback_active_row_move_to_pending(self, sender, app_data, user_data):
        self._remove_active_part_by_fingerprint(user_data["fingerprint"], move_to_pending=True)

    def _rebuild_part_row_list(self, container_tag, parts, active=False):
        if not dpg.does_item_exist(container_tag):
            return
        dpg.delete_item(container_tag, children_only=True)
        if len(parts) == 0:
            dpg.add_text("(none)", parent=container_tag)
            return
        for idx, part in enumerate(parts):
            fingerprint = part.get("fingerprint")
            label = self._format_part_row_label(part, idx, active=active)
            row_tag = dpg.add_button(label=label, parent=container_tag, width=-1)
            user_data = {"fingerprint": fingerprint, "row_type": "active" if active else "pending"}
            with dpg.popup(parent=row_tag, mousebutton=dpg.mvMouseButton_Right):
                dpg.add_text(label)
                if active:
                    dpg.add_menu_item(label="Remove", callback=self._callback_active_row_remove, user_data=user_data)
                    dpg.add_menu_item(label="Move to Pending", callback=self._callback_active_row_move_to_pending, user_data=user_data)
                    dpg.add_menu_item(label="Move to Active", enabled=False)
                else:
                    dpg.add_menu_item(label="Remove", callback=self._callback_pending_row_remove, user_data=user_data)
                    dpg.add_menu_item(label="Move to Pending", enabled=False)
                    dpg.add_menu_item(label="Move to Active", callback=self._callback_pending_row_move_to_active, user_data=user_data)

    def _refresh_part_row_lists_if_needed(self):
        signature = self._pending_active_signature()
        if (not self.part_list_dirty) and signature == self.part_row_signature:
            return
        if not dpg.does_item_exist("_cluster_pending_rows") or not dpg.does_item_exist("_cluster_active_rows"):
            return
        active_parts = [] if self.active_group is None else self.active_group.get("parts", [])
        pending_parts = self.pending_parts
        self._rebuild_part_row_list("_cluster_pending_rows", pending_parts, active=False)
        self._rebuild_part_row_list("_cluster_active_rows", active_parts, active=True)
        self.part_row_signature = signature
        self.part_list_dirty = False

    def _update_show_active_button_theme(self):
        if not dpg.does_item_exist("_ShowActiveButton"):
            return
        theme_tag = "_ShowActiveButtonOnTheme" if self.show_active else "_ShowActiveButtonOffTheme"
        if dpg.does_item_exist(theme_tag):
            dpg.bind_item_theme("_ShowActiveButton", theme_tag)

    def _format_part_list_text(self, title, parts, active=False):
        prefix = "A" if active else "P"
        score_thres = 0.0 if self.last_score_threshold is None else float(self.last_score_threshold)
        lines = [f"{title}:"]
        if len(parts) == 0:
            lines.append("(none)")
            return "\n".join(lines)
        for idx, part in enumerate(parts):
            if active:
                indices = self._part_visible_indices(part, active=True)
            else:
                indices = self._filter_part_indices(part, score_thres)
            count = 0 if indices is None else int(indices.numel())
            lines.append(f"{prefix}{idx}: cluster {int(part['cluster_id'])} ({count})")
        return "\n".join(lines)

    def _clear_cluster_selection(self):
        self.pending_cluster_ids = []
        self.pending_cluster_seed_indices = {}
        self.pending_parts = []
        self.active_group = None
        self._mark_part_list_dirty()
        self.selection_debug_render = None
        self.cluster_pick_request = None
        self.marquee_request = None
        self.cluster_pick_feature_render = None
        self.last_pick_confidence = None
        self.drag_mode = None
        self.drag_last_local_xy = None
        self.left_dragging = False
        self.middle_dragging = False
        self.right_dragging = False
        self.right_drag_start_local_xy = None
        self.right_drag_current_local_xy = None
        self.marquee_overlay_rect = None
        self.extend_handle = None

    def _set_cluster_status(self, message, state=None):
        if state is not None:
            self.cluster_cache_state = state
        self.cluster_status_message = message
        self._update_cluster_status_widget()

    def _update_cluster_status_widget(self):
        active_parts = [] if self.active_group is None else self.active_group.get("parts", [])
        pending_parts = self.pending_parts
        selected_points = 0
        if self.cluster_cache is not None:
            score_thres = 0.0 if self.last_score_threshold is None else self.last_score_threshold
            selection_mask = self._build_selection_mask(score_thres)
            if selection_mask is not None:
                selected_points = int(selection_mask.sum().item())

        if self.cluster_cache is not None and self.cluster_cache_state == "ready":
            cluster_summary = f"{self.cluster_cache['num_clusters']} {self.cluster_cache.get('method', 'HDBSCAN')} @ {self.cluster_cache['scale']:.3f}"
        elif self.cluster_cache_state == "clustering":
            cluster_summary = f"{self.cluster_method} @ {self.last_scale_value:.3f}" if self.last_scale_value is not None else self.cluster_method
        else:
            cluster_summary = "none"

        last_pick_value = "none" if self.last_picked_cluster_id is None else str(self.last_picked_cluster_id)
        global_hidden = 0 if self.cluster_cache is None else int(self.cluster_cache.get("global_residue_hidden_count", 0) or 0)
        active_hidden = 0 if self.active_group is None else int(self.active_group.get("residue_hidden_count", 0) or 0)

        line1 = f"Cache: {self.cluster_cache_state}\nClusters: {cluster_summary}"
        line2 = f"Last: {last_pick_value}"
        line3 = f"Pending: {len(pending_parts)} | Active: {len(active_parts)} | G: {selected_points}\nPick: {self.last_pick_result}"
        residue_line = f"Residue: active {active_hidden} | global {global_hidden}"

        if dpg.does_item_exist("_cluster_summary_line1"):
            dpg.set_value("_cluster_summary_line1", line1)
        if dpg.does_item_exist("_cluster_summary_line2"):
            dpg.set_value("_cluster_summary_line2", line2)
        if dpg.does_item_exist("_cluster_summary_line3"):
            dpg.set_value("_cluster_summary_line3", line3)
        if dpg.does_item_exist("_cluster_residue_line"):
            dpg.set_value("_cluster_residue_line", residue_line)
        self._refresh_part_row_lists_if_needed()

        has_active = self._active_group_has_cuboid()
        has_pending = len(pending_parts) > 0
        if dpg.does_item_exist("_CreateCuboidButton"):
            dpg.configure_item("_CreateCuboidButton", show=not has_active, enabled=(not has_active and has_pending))
        if dpg.does_item_exist("_AddActiveButton"):
            dpg.configure_item("_AddActiveButton", show=has_active, enabled=(has_active and has_pending))
        if dpg.does_item_exist("_RemoveActiveButton"):
            dpg.configure_item("_RemoveActiveButton", show=has_active, enabled=(has_active and has_pending))
        if dpg.does_item_exist("_ShowActiveButton"):
            dpg.configure_item("_ShowActiveButton", enabled=has_active)
        if dpg.does_item_exist("_PendingTab"):
            dpg.configure_item("_PendingTab", label=f"Pending ({len(pending_parts)})")
        if dpg.does_item_exist("_ActiveTab"):
            dpg.configure_item("_ActiveTab", label=f"Active ({len(active_parts)})")
        self._update_show_active_button_theme()

    def _set_pick_feedback(self, cluster_id, result):

            self.last_picked_cluster_id = cluster_id
            self.last_pick_result = result
            if cluster_id is None:
                self.last_pick_confidence = None
                print(f"Pick result: {result}")
            else:
                confidence_suffix = ""
                if self.last_pick_confidence is not None:
                    confidence_suffix = f", confidence {self.last_pick_confidence:.3f}"
                print(f"Pick result: {result}, cluster {cluster_id}{confidence_suffix}")
            self._update_cluster_status_widget()

    def _build_mask_for_cluster_ids(self, cluster_ids, score_thres):
        if self.cluster_cache is None:
            return None
        if len(cluster_ids) == 0:
            return None

        labels = self.cluster_cache["labels"]
        confidence = self.cluster_cache["confidence"]
        selection_mask = torch.zeros_like(labels, dtype=torch.bool)
        for cluster_id in cluster_ids:
            selection_mask |= labels == int(cluster_id)
        selection_mask &= confidence >= score_thres
        if int(selection_mask.sum().item()) == 0:
            return None
        return selection_mask

    def _residue_scope_includes(self, scope_name):
        scope = str(self.residue_scope).strip().lower()
        target = str(scope_name).strip().lower()
        return scope == "both" or scope == target

    def _active_residue_enabled(self):
        return self.hide_small_residue and self.residue_max_size >= 0 and self._residue_scope_includes("active")

    def _global_residue_enabled(self):
        return self.hide_small_residue and self.residue_max_size >= 0 and self._residue_scope_includes("global")

    def _clear_global_residue_cache(self):
        if self.cluster_cache is None:
            return
        self.cluster_cache["global_residue_hidden_mask"] = None
        self.cluster_cache["global_residue_hidden_count"] = 0

    def _split_indices_by_small_residue(self, indices):
        if indices is None or indices.numel() == 0:
            return indices, indices
        cache = self._build_connectivity_components(indices)
        if cache is None or cache["component_count"] <= 1:
            empty = indices[:0]
            return indices, empty

        component_indices = cache["component_indices"]
        component_sizes = [int(component.numel()) for component in component_indices]
        if len(component_sizes) == 0:
            empty = indices[:0]
            return indices, empty
        largest_component_id = int(np.argmax(component_sizes))

        visible_chunks = []
        residue_chunks = []
        for component_id, component in enumerate(component_indices):
            if component_id == largest_component_id or int(component.numel()) > int(self.residue_max_size):
                visible_chunks.append(component)
            else:
                residue_chunks.append(component)

        visible_indices = torch.unique(torch.cat(visible_chunks, dim=0)) if len(visible_chunks) > 0 else component_indices[largest_component_id]
        residue_indices = torch.unique(torch.cat(residue_chunks, dim=0)) if len(residue_chunks) > 0 else indices[:0]
        return visible_indices, residue_indices

    def _compute_global_residue_hidden_mask(self):
        if self.cluster_cache is None:
            return None
        labels = self.cluster_cache["labels"]
        hidden_mask = torch.zeros_like(labels, dtype=torch.bool)
        unique_labels = torch.unique(labels).detach().cpu().tolist()
        for label_value in unique_labels:
            if int(label_value) < -1:
                continue
            cluster_indices = torch.nonzero(labels == int(label_value), as_tuple=False).squeeze(-1)
            if cluster_indices.numel() == 0:
                continue
            _, residue_indices = self._split_indices_by_small_residue(cluster_indices)
            if residue_indices is not None and residue_indices.numel() > 0:
                hidden_mask[residue_indices] = True
        return hidden_mask

    def _current_global_residue_hidden_mask(self):
        if self.cluster_cache is None or self.cluster_cache_state != "ready" or not self._global_residue_enabled():
            return None
        cached_mask = self.cluster_cache.get("global_residue_hidden_mask")
        if cached_mask is None:
            cached_mask = self._compute_global_residue_hidden_mask()
            self.cluster_cache["global_residue_hidden_mask"] = cached_mask
            self.cluster_cache["global_residue_hidden_count"] = 0 if cached_mask is None else int(cached_mask.sum().item())
        return cached_mask

    def _current_active_residue_hidden_mask(self):
        if not self._active_group_has_cuboid() or not self._active_residue_enabled():
            return None
        return self.active_group.get("residue_hidden_mask")

    def _combined_hidden_mask(self):
        combined_mask = None
        if self._active_group_has_cuboid() and not self.show_active:
            selection_mask = self._get_active_selection_mask()
            if selection_mask is not None:
                combined_mask = selection_mask.clone()

        global_hidden_mask = self._current_global_residue_hidden_mask()
        if global_hidden_mask is not None:
            combined_mask = global_hidden_mask.clone() if combined_mask is None else (combined_mask | global_hidden_mask)

        active_hidden_mask = self._current_active_residue_hidden_mask()
        if active_hidden_mask is not None:
            combined_mask = active_hidden_mask.clone() if combined_mask is None else (combined_mask | active_hidden_mask)

        if combined_mask is None or int(combined_mask.sum().item()) == 0:
            return None
        return combined_mask

    def _part_visible_indices(self, part, active=False, score_thres=None):
        if active:
            indices = part.get("visible_global_indices")
            if indices is None:
                indices = part.get("global_indices")
        else:
            score_thres = 0.0 if score_thres is None else float(score_thres)
            indices = self._filter_part_indices(part, score_thres)
        if indices is None or indices.numel() == 0:
            return indices

        global_hidden_mask = self._current_global_residue_hidden_mask()
        if active and global_hidden_mask is not None:
            visible = indices[~global_hidden_mask[indices]]
            return visible if visible.numel() > 0 else indices[:0]
        return indices

    def _part_export_indices(self, part):
        if self.exclude_residue_on_active_export:
            return self._part_visible_indices(part, active=True)
        indices = part.get("all_global_indices")
        if indices is None:
            indices = part.get("global_indices")
        return indices

    def _on_residue_settings_changed(self, reason):
        self._clear_global_residue_cache()
        if self._active_group_has_cuboid():
            self._rebuild_active_group_for_score(0.0 if self.last_score_threshold is None else float(self.last_score_threshold))
        else:
            self._update_cluster_status_widget()
        self._set_cluster_status(reason, state=self.cluster_cache_state)

    def _effective_connectivity_cap(self):
        preview_cap = self.preview_max_gaussians
        if dpg.does_item_exist("_PreviewMaxGaussians"):
            preview_cap = dpg.get_value("_PreviewMaxGaussians")
        return int(max(1, preview_cap))

    def _limit_indices_around_focus(self, indices, focus_global_indices, limit):
        if indices.shape[0] <= limit:
            return indices
        if focus_global_indices is None:
            return indices[:limit]
        focus_global_indices = torch.as_tensor(focus_global_indices, device=indices.device, dtype=indices.dtype).reshape(-1)
        if focus_global_indices.numel() == 0:
            return indices[:limit]
        focus_global_indices = focus_global_indices[(focus_global_indices >= 0) & (focus_global_indices < self.engine['scene'].get_xyz.shape[0])]
        if focus_global_indices.numel() == 0:
            return indices[:limit]
        focus_mask = torch.isin(indices, focus_global_indices)
        focus_members = indices[focus_mask]
        if focus_members.shape[0] >= limit:
            return focus_members[:limit]
        focus_center = self.engine['scene'].get_xyz[focus_members].mean(dim=0) if focus_members.numel() > 0 else self.engine['scene'].get_xyz[focus_global_indices].mean(dim=0)
        points = self.engine['scene'].get_xyz[indices]
        dist2 = torch.sum((points - focus_center.unsqueeze(0)) ** 2, dim=1)
        candidate_order = torch.topk(dist2, k=indices.shape[0], largest=False).indices
        ordered = indices[candidate_order]
        if focus_members.numel() == 0:
            return ordered[:limit]
        ordered = torch.cat([focus_members, ordered[~torch.isin(ordered, focus_members)]], dim=0)
        return ordered[:limit]

    def _limit_indices_around_seed(self, indices, seed_global_index, limit):
        if seed_global_index is None:
            return indices[:limit] if indices.shape[0] > limit else indices
        return self._limit_indices_around_focus(indices, [int(seed_global_index)], limit)

    def _build_connectivity_components(self, cluster_indices):
        if cluster_indices.numel() == 0:
            return None
        points = self.engine['scene'].get_xyz[cluster_indices].detach().cpu().numpy().astype(np.float32, copy=False)
        if points.shape[0] == 1:
            return {
                "indices": cluster_indices,
                "component_labels": np.zeros((1,), dtype=np.int32),
                "component_count": 1,
                "component_indices": [cluster_indices],
                "global_to_component": {int(cluster_indices[0].item()): 0},
            }
        tree = cKDTree(points)
        k = min(self.connectivity_knn + 1, points.shape[0])
        distances, neighbors = tree.query(points, k=k, workers=-1)
        if distances.ndim == 1:
            distances = distances[:, None]
            neighbors = neighbors[:, None]
        neighbor_distances = distances[:, 1:].astype(np.float32, copy=False)
        neighbor_indices = neighbors[:, 1:].astype(np.int32, copy=False)
        if neighbor_distances.size == 0:
            labels_np = np.arange(points.shape[0], dtype=np.int32)
        else:
            local_k = min(self.connectivity_local_scale_neighbors, neighbor_distances.shape[1])
            local_distances = neighbor_distances[:, :local_k]
            local_scale = np.median(local_distances, axis=1)
            valid_scale = np.isfinite(local_scale) & (local_scale > 0)
            base_scale = max(float(np.median(local_scale[valid_scale])), 1e-3) if np.any(valid_scale) else 1e-3
            local_scale = np.where(valid_scale, np.maximum(local_scale, base_scale), base_scale)
            neighbor_scale = local_scale[np.clip(neighbor_indices, 0, points.shape[0] - 1)]
            edge_threshold = self.connectivity_scale * np.maximum(np.maximum(local_scale[:, None], neighbor_scale), base_scale)
            valid_edges = np.isfinite(neighbor_distances) & (neighbor_indices >= 0) & (neighbor_distances <= edge_threshold)
            if np.any(valid_edges):
                rows = np.repeat(np.arange(points.shape[0], dtype=np.int32), neighbor_indices.shape[1])
                cols = neighbor_indices.reshape(-1)
                edge_mask = valid_edges.reshape(-1)
                rows = rows[edge_mask]
                cols = cols[edge_mask]
                graph_rows = np.concatenate([rows, cols])
                graph_cols = np.concatenate([cols, rows])
                graph = csr_matrix((np.ones(graph_rows.shape[0], dtype=np.uint8), (graph_rows, graph_cols)), shape=(points.shape[0], points.shape[0]))
                _, labels_np = connected_components(graph, directed=False, return_labels=True)
            else:
                labels_np = np.arange(points.shape[0], dtype=np.int32)

        component_indices = []
        global_to_component = {}
        unique_components = np.unique(labels_np)
        for component_id in unique_components.tolist():
            member_idx = np.nonzero(labels_np == component_id)[0]
            member_t = torch.from_numpy(member_idx.astype(np.int64)).to(device=cluster_indices.device)
            member_indices = cluster_indices[member_t]
            component_indices.append(member_indices)
            for global_index in member_indices.detach().cpu().tolist():
                global_to_component[int(global_index)] = int(component_id)
        return {
            "indices": cluster_indices,
            "component_labels": labels_np,
            "component_count": len(component_indices),
            "component_indices": component_indices,
            "global_to_component": global_to_component,
        }

    def _get_cluster_component_cache(self, cluster_id, preferred_global_indices=None):
        if self.cluster_cache is None:
            return None
        cluster_indices = torch.nonzero(self.cluster_cache["labels"] == int(cluster_id), as_tuple=False).squeeze(-1)
        if cluster_indices.numel() == 0:
            return None
        cap = self._effective_connectivity_cap()
        cacheable = cluster_indices.shape[0] <= cap and (preferred_global_indices is None or len(preferred_global_indices) == 0)
        cache_key = (self.cluster_request_generation, int(cluster_id), int(cap))
        if cacheable and cache_key in self.component_cache:
            return self.component_cache[cache_key]
        capped_indices = cluster_indices if cacheable else self._limit_indices_around_focus(cluster_indices, preferred_global_indices, cap)
        cache = self._build_connectivity_components(capped_indices)
        if cacheable and cache is not None:
            self.component_cache[cache_key] = cache
        return cache

    def _extract_connected_component_from_seed(self, cluster_indices, seed_global_index):
        if seed_global_index is None:
            return cluster_indices
        if cluster_indices.numel() == 0:
            return cluster_indices
        capped_indices = self._limit_indices_around_seed(cluster_indices, seed_global_index, self._effective_connectivity_cap())
        seed_matches = torch.nonzero(capped_indices == int(seed_global_index), as_tuple=False).squeeze(-1)
        if seed_matches.numel() == 0:
            return None
        cache = self._build_connectivity_components(capped_indices)
        if cache is None:
            return None
        component_id = cache["global_to_component"].get(int(seed_global_index))
        if component_id is None:
            return capped_indices[seed_matches[:1]]
        component_indices = cache["component_indices"][int(component_id)]
        print(f"Connected component build: {int(cluster_indices.shape[0])} cluster gaussians -> {int(capped_indices.shape[0])} candidates -> {int(component_indices.shape[0])} connected gaussians")
        return component_indices

    def _quat_multiply(self, lhs, rhs):

            lw, lx, ly, lz = lhs.unbind(dim=-1)
            rw, rx, ry, rz = rhs.unbind(dim=-1)
            return torch.stack([
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ], dim=-1)

    def _rotation_matrix_to_quaternion(self, rotation_matrix):
        quat_xyzw = R.from_matrix(rotation_matrix.detach().cpu().numpy()).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        return torch.from_numpy(quat_wxyz).to(rotation_matrix.device)

    def _group_rotation_quaternion(self, active_group):
        quat = self._rotation_matrix_to_quaternion(active_group["group_rotation"])
        return torch.nn.functional.normalize(quat, dim=0)

    def _transform_points_with_group(self, active_group, points):
        centered = points - active_group["base_center"].unsqueeze(0)
        rotated = centered @ active_group["group_rotation"].transpose(0, 1)
        return rotated + active_group["base_center"].unsqueeze(0) + active_group["group_translation"].unsqueeze(0)

    def _transform_rotations_with_group(self, active_group, rotations):
        group_quat = self._group_rotation_quaternion(active_group).unsqueeze(0).expand(rotations.shape[0], -1)
        return torch.nn.functional.normalize(self._quat_multiply(group_quat, rotations), dim=-1)

    def _active_group_has_cuboid(self):
        return self.active_group is not None and self.active_group.get("cuboid_ready", False)

    def _get_active_selection_mask(self):
        if self.active_group is None:
            return None
        if self.active_group.get("selection_mask") is not None:
            return self.active_group["selection_mask"]
        score_thres = 0.0 if self.last_score_threshold is None else self.last_score_threshold
        return self._build_mask_for_cluster_ids(self.active_group["cluster_ids"], score_thres)

    def _get_cuboid_percentile(self):
        percentile = self.cuboid_percentile
        if dpg.does_item_exist("_CuboidPercentile"):
            percentile = dpg.get_value("_CuboidPercentile")
        percentile = float(np.clip(percentile, 50.0, 100.0))
        self.cuboid_percentile = percentile
        return percentile

    def _compute_cuboid_obb(self, indices, percentile=None):
        points = self.engine['scene'].get_xyz[indices]
        center = points.mean(dim=0)
        centered = points - center.unsqueeze(0)

        if points.shape[0] >= 3:
            covariance = torch.matmul(centered.t(), centered) / max(points.shape[0], 1)
            eigenvalues, eigenvectors = torch.linalg.eigh(covariance.float())
            order = torch.argsort(eigenvalues, descending=True)
            axes = eigenvectors[:, order]
        else:
            axes = torch.eye(3, device=points.device, dtype=points.dtype)

        if torch.det(axes) < 0:
            axes[:, 2] = -axes[:, 2]

        percentile = self._get_cuboid_percentile() if percentile is None else float(percentile)
        percentile = float(np.clip(percentile, 50.0, 100.0))
        local = centered @ axes
        abs_local = local.abs().float()
        if percentile >= 100.0 - 1e-6:
            extents = abs_local.max(dim=0).values
        else:
            extents = torch.quantile(abs_local, percentile / 100.0, dim=0)
        extents = extents.to(dtype=points.dtype)
        local_scaling = self.engine['scene'].get_scaling[indices]
        min_extent = torch.clamp(local_scaling.mean(dim=0).max(), min=1e-3)
        extents = torch.clamp(extents + min_extent * 0.5, min=min_extent)
        return center, axes, extents

    def _sample_preview_indices(self, indices, labels, preview_count):
        total = int(indices.shape[0])
        if total <= preview_count:
            return indices.clone()

        unique_labels, counts = torch.unique(labels, return_counts=True)
        counts_f = counts.float()
        quotas_f = counts_f * (float(preview_count) / float(total))
        quotas = torch.floor(quotas_f).long()
        if preview_count >= unique_labels.shape[0]:
            quotas = torch.clamp(quotas, min=1)
        quotas = torch.minimum(quotas, counts)

        assigned = int(quotas.sum().item())
        remainder = preview_count - assigned
        if remainder > 0:
            fractional = quotas_f - torch.floor(quotas_f)
            priority = torch.argsort(fractional, descending=True)
            for idx in priority.tolist():
                if remainder <= 0:
                    break
                if quotas[idx] < counts[idx]:
                    quotas[idx] += 1
                    remainder -= 1
        elif remainder < 0:
            priority = torch.argsort(quotas_f - torch.floor(quotas_f))
            for idx in priority.tolist():
                if remainder >= 0:
                    break
                if quotas[idx] > 1:
                    quotas[idx] -= 1
                    remainder += 1

        sampled = []
        for label_value, quota in zip(unique_labels.tolist(), quotas.tolist()):
            if quota <= 0:
                continue
            cluster_indices = indices[labels == label_value]
            if cluster_indices.shape[0] <= quota:
                sampled.append(cluster_indices)
                continue
            perm = torch.randperm(cluster_indices.shape[0], device=cluster_indices.device)[:quota]
            sampled.append(cluster_indices[perm])

        if len(sampled) == 0:
            return indices[:preview_count].clone()
        return torch.cat(sampled, dim=0)

    def _build_part_from_component_indices(self, cluster_id, component_indices, seed_index=None, source="click"):
        if component_indices is None or component_indices.numel() == 0:
            return None
        source_indices = torch.unique(component_indices.long())
        if source_indices.numel() == 0:
            return None
        return {
            "cluster_id": int(cluster_id),
            "seed_index": None if seed_index is None else int(seed_index),
            "source": source,
            "source_indices": source_indices.detach().clone(),
            "fingerprint": self._part_fingerprint(source_indices),
        }

    def _build_part_from_seed(self, cluster_id, seed_index):
        if self.cluster_cache is None or seed_index is None:
            return None
        cluster_indices = torch.nonzero(self.cluster_cache["labels"] == int(cluster_id), as_tuple=False).squeeze(-1)
        if cluster_indices.numel() == 0:
            return None
        component_indices = self._extract_connected_component_from_seed(cluster_indices, seed_index)
        if component_indices is None or component_indices.numel() == 0:
            return None
        return self._build_part_from_component_indices(cluster_id, component_indices, seed_index=seed_index, source="click")

    def _clip_indices_to_cuboid(self, indices, center, axes, extents):
        if indices is None or indices.numel() == 0:
            return indices
        points = self.engine['scene'].get_xyz[indices]
        local = (points - center.unsqueeze(0)) @ axes
        inside = (torch.abs(local) <= (extents.unsqueeze(0) + 1e-5)).all(dim=1)
        clipped = indices[inside]
        return clipped if clipped.numel() > 0 else indices

    def _build_active_group(self, source_parts, score_thres):
        if self.cluster_cache is None or source_parts is None or len(source_parts) == 0:
            return None

        scene_count = self.engine['scene'].get_xyz.shape[0]
        feature_count = self.engine['feature'].get_xyz.shape[0]
        max_count = min(scene_count, feature_count)
        filtered_parts = []
        filtered_chunks = []
        for part in source_parts:
            filtered = self._filter_part_indices(part, score_thres)
            if filtered is None or filtered.numel() == 0:
                continue
            filtered_parts.append((self._clone_part(part), filtered.detach().clone()))
            filtered_chunks.append(filtered)

        if len(filtered_parts) == 0:
            return None

        candidate_indices = torch.unique(torch.cat(filtered_chunks, dim=0))
        if candidate_indices.numel() == 0:
            return None

        percentile = self._get_cuboid_percentile()
        initial_center, initial_axes, initial_extents = self._compute_cuboid_obb(candidate_indices, percentile=percentile)
        inside_candidate = self._clip_indices_to_cuboid(candidate_indices, initial_center, initial_axes, initial_extents)
        if inside_candidate is None or inside_candidate.numel() == 0:
            inside_candidate = candidate_indices

        active_parts = []
        clipped_chunks = []
        visible_chunks = []
        residue_hidden_chunks = []
        for part_copy, filtered in filtered_parts:
            clipped = filtered[torch.isin(filtered, inside_candidate)]
            if clipped.numel() == 0:
                continue
            visible_indices = clipped
            residue_hidden_indices = clipped[:0]
            if self._active_residue_enabled():
                visible_indices, residue_hidden_indices = self._split_indices_by_small_residue(clipped)
            part_copy["global_indices"] = clipped.detach().clone()
            part_copy["all_global_indices"] = clipped.detach().clone()
            part_copy["visible_global_indices"] = visible_indices.detach().clone()
            part_copy["residue_hidden_indices"] = residue_hidden_indices.detach().clone()
            active_parts.append(part_copy)
            clipped_chunks.append(clipped)
            visible_chunks.append(visible_indices)
            if residue_hidden_indices.numel() > 0:
                residue_hidden_chunks.append(residue_hidden_indices)

        if len(active_parts) == 0:
            return None

        indices = torch.unique(torch.cat(clipped_chunks, dim=0))
        if indices.numel() == 0:
            return None
        visible_indices = torch.unique(torch.cat(visible_chunks, dim=0))
        if visible_indices.numel() == 0:
            return None

        center, axes, extents = self._compute_cuboid_obb(visible_indices, percentile=100.0)
        selection_mask = torch.zeros((max_count,), dtype=torch.bool, device=indices.device)
        selection_mask[indices] = True
        visible_selection_mask = torch.zeros_like(selection_mask)
        visible_selection_mask[visible_indices] = True
        residue_hidden_mask = torch.zeros_like(selection_mask)
        if len(residue_hidden_chunks) > 0:
            residue_hidden_indices = torch.unique(torch.cat(residue_hidden_chunks, dim=0))
            residue_hidden_mask[residue_hidden_indices] = True

        selection_labels = self.cluster_cache["labels"][visible_indices]
        preview_cap = int(max(1, dpg.get_value("_PreviewMaxGaussians") if dpg.does_item_exist("_PreviewMaxGaussians") else self.preview_max_gaussians))
        preview_count = min(int(visible_indices.shape[0]), preview_cap)
        preview_indices = self._sample_preview_indices(visible_indices, selection_labels, preview_count)
        preview_mask = torch.zeros_like(selection_mask)
        preview_mask[preview_indices] = True
        hidden_mask = visible_selection_mask & ~preview_mask

        return {
            "cluster_ids": sorted({int(part["cluster_id"]) for part in active_parts}),
            "parts": active_parts,
            "source_parts": [self._clone_part(part) for part in source_parts],
            "cuboid_percentile": percentile,
            "cuboid_ready": True,
            "selection_mask": selection_mask,
            "visible_selection_mask": visible_selection_mask,
            "global_indices": indices,
            "visible_global_indices": visible_indices,
            "preview_indices": preview_indices,
            "preview_mask": preview_mask,
            "hidden_mask": hidden_mask,
            "residue_hidden_mask": residue_hidden_mask,
            "residue_hidden_count": int(residue_hidden_mask.sum().item()),
            "preview_only": visible_indices.shape[0] > preview_indices.shape[0],
            "base_center": center.detach().clone(),
            "base_axes": axes.detach().clone(),
            "base_extents": extents.detach().clone(),
            "group_rotation": torch.eye(3, device=center.device, dtype=center.dtype),
            "group_translation": torch.zeros(3, device=center.device, dtype=center.dtype),
            "group_scale_local": torch.ones(3, device=center.device, dtype=center.dtype),
            "dirty_transform": False,
            "previewing_sparse": False,
            "score_thres": float(score_thres),
            "scene_base_xyz": self.engine['scene']._xyz[indices].detach().clone(),
            "scene_base_rotation": self.engine['scene'].get_rotation[indices].detach().clone(),
            "scene_base_scaling_raw": self.engine['scene']._scaling[indices].detach().clone(),
            "scene_base_opacity_raw": self.engine['scene']._opacity[indices].detach().clone(),
            "feature_base_xyz": self.engine['feature']._xyz[indices].detach().clone(),
            "feature_base_rotation": self.engine['feature'].get_rotation[indices].detach().clone(),
            "feature_base_scaling_raw": self.engine['feature']._scaling[indices].detach().clone(),
            "feature_base_opacity_raw": self.engine['feature']._opacity[indices].detach().clone(),
        }

    def _restore_group_state(self, active_group):
        indices = active_group["global_indices"]
        with torch.no_grad():
            self.engine['scene']._xyz[indices] = active_group["scene_base_xyz"]
            self.engine['scene']._rotation[indices] = active_group["scene_base_rotation"]
            self.engine['scene']._scaling[indices] = active_group["scene_base_scaling_raw"]
            self.engine['scene']._opacity[indices] = active_group["scene_base_opacity_raw"]
            self.engine['feature']._xyz[indices] = active_group["feature_base_xyz"]
            self.engine['feature']._rotation[indices] = active_group["feature_base_rotation"]
            self.engine['feature']._scaling[indices] = active_group["feature_base_scaling_raw"]
            self.engine['feature']._opacity[indices] = active_group["feature_base_opacity_raw"]
        active_group["previewing_sparse"] = False

    def _current_group_scale_local(self, active_group):
        if "group_scale_local" not in active_group or active_group["group_scale_local"] is None:
            active_group["group_scale_local"] = torch.ones(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        return active_group["group_scale_local"]

    def _current_cuboid_center(self, active_group):
        return active_group["base_center"] + active_group["group_translation"]

    def _current_cuboid_axes(self, active_group):
        return active_group["group_rotation"] @ active_group["base_axes"]

    def _current_cuboid_extents(self, active_group):
        return active_group["base_extents"] * self._current_group_scale_local(active_group)

    def _linear_transform_matrix(self, active_group):
        return self._current_cuboid_axes(active_group) @ torch.diag(self._current_group_scale_local(active_group)) @ active_group["base_axes"].transpose(0, 1)

    def _group_has_nonuniform_scale(self, active_group):
        scale_local = self._current_group_scale_local(active_group)
        return bool(torch.max(torch.abs(scale_local - 1.0)).item() > 1e-4)

    def _transform_points_with_group(self, active_group, points):
        linear_transform = self._linear_transform_matrix(active_group)
        centered = points - active_group["base_center"].unsqueeze(0)
        return centered @ linear_transform.transpose(0, 1) + active_group["base_center"].unsqueeze(0) + active_group["group_translation"].unsqueeze(0)

    def _transform_rotations_with_group(self, active_group, rotations):
        group_quat = self._group_rotation_quaternion(active_group).unsqueeze(0).expand(rotations.shape[0], -1)
        return torch.nn.functional.normalize(self._quat_multiply(group_quat, rotations), dim=-1)

    def _rotation_matrices_to_quaternions(self, rotation_matrices):
        mats = rotation_matrices.detach().cpu().numpy()
        quat_xyzw = R.from_matrix(mats).as_quat()
        quat_wxyz = np.stack([quat_xyzw[:, 3], quat_xyzw[:, 0], quat_xyzw[:, 1], quat_xyzw[:, 2]], axis=1).astype(np.float32)
        return torch.from_numpy(quat_wxyz).to(rotation_matrices.device)

    def _apply_affine_to_scales_rotations(self, scaling_raw, rotation_raw, linear_transform):
        scaling = torch.exp(scaling_raw)
        rotation_mats = build_rotation(rotation_raw)
        L = rotation_mats @ torch.diag_embed(scaling)
        transformed = linear_transform.unsqueeze(0) @ L
        U, S, _ = torch.linalg.svd(transformed.float())
        det = torch.det(U)
        neg_mask = det < 0
        if torch.any(neg_mask):
            U[neg_mask, :, 2] *= -1.0
        new_rotation = self._rotation_matrices_to_quaternions(U.to(dtype=rotation_raw.dtype))
        new_scaling = torch.log(torch.clamp(S.to(dtype=scaling_raw.dtype), min=1e-6))
        return new_scaling, new_rotation

    def _update_group_preview(self, active_group):
        indices = active_group["preview_indices"] if active_group["preview_only"] else active_group["global_indices"]
        selection_indices = active_group["global_indices"]
        scene_base_xyz = active_group["scene_base_xyz"]
        scene_base_rotation = active_group["scene_base_rotation"]
        feature_base_xyz = active_group["feature_base_xyz"]
        feature_base_rotation = active_group["feature_base_rotation"]

        if active_group["preview_only"]:
            local_preview = active_group["preview_mask"][selection_indices]
            scene_base_xyz = scene_base_xyz[local_preview]
            scene_base_rotation = scene_base_rotation[local_preview]
            feature_base_xyz = feature_base_xyz[local_preview]
            feature_base_rotation = feature_base_rotation[local_preview]

        transformed_scene_xyz = self._transform_points_with_group(active_group, scene_base_xyz)
        transformed_feature_xyz = self._transform_points_with_group(active_group, feature_base_xyz)
        if self._group_has_nonuniform_scale(active_group):
            transformed_scene_rotation = scene_base_rotation
            transformed_feature_rotation = feature_base_rotation
        else:
            transformed_scene_rotation = self._transform_rotations_with_group(active_group, scene_base_rotation)
            transformed_feature_rotation = self._transform_rotations_with_group(active_group, feature_base_rotation)

        with torch.no_grad():
            if active_group["preview_only"] and not active_group["previewing_sparse"]:
                hidden_indices = torch.nonzero(active_group["hidden_mask"], as_tuple=False).squeeze(-1)
                if hidden_indices.numel() > 0:
                    self.engine['scene']._opacity[hidden_indices] = self.hidden_opacity_logit
                    self.engine['feature']._opacity[hidden_indices] = self.hidden_opacity_logit
                active_group["previewing_sparse"] = True

            self.engine['scene']._xyz[indices] = transformed_scene_xyz
            self.engine['scene']._rotation[indices] = transformed_scene_rotation
            self.engine['feature']._xyz[indices] = transformed_feature_xyz
            self.engine['feature']._rotation[indices] = transformed_feature_rotation

    def _commit_active_group(self):
        if not self._active_group_has_cuboid():
            return
        active_group = self.active_group
        if not active_group["dirty_transform"]:
            self._restore_group_state(active_group)
            return

        indices = active_group["global_indices"]
        linear_transform = self._linear_transform_matrix(active_group)
        scene_xyz = self._transform_points_with_group(active_group, active_group["scene_base_xyz"])
        feature_xyz = self._transform_points_with_group(active_group, active_group["feature_base_xyz"])
        if self._group_has_nonuniform_scale(active_group):
            scene_scaling_raw, scene_rot = self._apply_affine_to_scales_rotations(active_group["scene_base_scaling_raw"], active_group["scene_base_rotation"], linear_transform)
            feature_scaling_raw, feature_rot = self._apply_affine_to_scales_rotations(active_group["feature_base_scaling_raw"], active_group["feature_base_rotation"], linear_transform)
        else:
            scene_scaling_raw = active_group["scene_base_scaling_raw"]
            feature_scaling_raw = active_group["feature_base_scaling_raw"]
            scene_rot = self._transform_rotations_with_group(active_group, active_group["scene_base_rotation"])
            feature_rot = self._transform_rotations_with_group(active_group, active_group["feature_base_rotation"])

        with torch.no_grad():
            self.engine['scene']._xyz[indices] = scene_xyz
            self.engine['scene']._rotation[indices] = scene_rot
            self.engine['scene']._scaling[indices] = scene_scaling_raw
            self.engine['scene']._opacity[indices] = active_group["scene_base_opacity_raw"]
            self.engine['feature']._xyz[indices] = feature_xyz
            self.engine['feature']._rotation[indices] = feature_rot
            self.engine['feature']._scaling[indices] = feature_scaling_raw
            self.engine['feature']._opacity[indices] = active_group["feature_base_opacity_raw"]

        active_group["scene_base_xyz"] = scene_xyz.detach().clone()
        active_group["scene_base_rotation"] = scene_rot.detach().clone()
        active_group["scene_base_scaling_raw"] = scene_scaling_raw.detach().clone()
        active_group["feature_base_xyz"] = feature_xyz.detach().clone()
        active_group["feature_base_rotation"] = feature_rot.detach().clone()
        active_group["feature_base_scaling_raw"] = feature_scaling_raw.detach().clone()
        active_group["base_center"] = self._current_cuboid_center(active_group).detach().clone()
        active_group["base_axes"] = self._current_cuboid_axes(active_group).detach().clone()
        active_group["base_extents"] = self._current_cuboid_extents(active_group).detach().clone()
        active_group["group_rotation"] = torch.eye(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["group_translation"] = torch.zeros(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["group_scale_local"] = torch.ones(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["dirty_transform"] = False
        active_group["previewing_sparse"] = False

    def _cancel_active_group(self):
        if not self._active_group_has_cuboid():
            return
        active_group = self.active_group
        self._restore_group_state(active_group)
        active_group["group_rotation"] = torch.eye(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["group_translation"] = torch.zeros(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["group_scale_local"] = torch.ones(3, device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
        active_group["dirty_transform"] = False
        active_group["previewing_sparse"] = False
        self._set_cluster_status("active transform canceled")

    def _recompute_active_group_cuboid(self):
        if not self._active_group_has_cuboid():
            return
        source_parts = [self._clone_part(part) for part in self.active_group.get("source_parts", [])]
        self._cancel_preview_for_selection_change()
        self._set_active_group_from_source_parts(
            source_parts,
            success_status=f"active cuboid percentile={self.cuboid_percentile:.1f}",
            empty_status="selection cleared by cuboid re-fit",
        )

    def _commit_active_group_if_dirty(self):
        if self._active_group_has_cuboid() and self.active_group["dirty_transform"]:
            self._commit_active_group()

    def _capture_state_snapshot(self, indices, use_active_group_base=None):
        indices_gpu = indices
        indices_cpu = indices.detach().cpu().clone()
        if use_active_group_base is not None:
            active_group = use_active_group_base
            return {
                "indices": indices_cpu,
                "scene_xyz": active_group["scene_base_xyz"].detach().cpu().clone(),
                "scene_rotation": active_group["scene_base_rotation"].detach().cpu().clone(),
                "scene_scaling_raw": active_group["scene_base_scaling_raw"].detach().cpu().clone(),
                "scene_opacity_raw": active_group["scene_base_opacity_raw"].detach().cpu().clone(),
                "feature_xyz": active_group["feature_base_xyz"].detach().cpu().clone(),
                "feature_rotation": active_group["feature_base_rotation"].detach().cpu().clone(),
                "feature_scaling_raw": active_group["feature_base_scaling_raw"].detach().cpu().clone(),
                "feature_opacity_raw": active_group["feature_base_opacity_raw"].detach().cpu().clone(),
            }
        return {
            "indices": indices_cpu,
            "scene_xyz": self.engine['scene']._xyz[indices_gpu].detach().cpu().clone(),
            "scene_rotation": self.engine['scene']._rotation[indices_gpu].detach().cpu().clone(),
            "scene_scaling_raw": self.engine['scene']._scaling[indices_gpu].detach().cpu().clone(),
            "scene_opacity_raw": self.engine['scene']._opacity[indices_gpu].detach().cpu().clone(),
            "feature_xyz": self.engine['feature']._xyz[indices_gpu].detach().cpu().clone(),
            "feature_rotation": self.engine['feature']._rotation[indices_gpu].detach().cpu().clone(),
            "feature_scaling_raw": self.engine['feature']._scaling[indices_gpu].detach().cpu().clone(),
            "feature_opacity_raw": self.engine['feature']._opacity[indices_gpu].detach().cpu().clone(),
        }

    def _restore_state_snapshot(self, snapshot):
        indices = snapshot["indices"].to(device=self.engine['scene']._xyz.device, dtype=torch.long)
        with torch.no_grad():
            self.engine['scene']._xyz[indices] = snapshot["scene_xyz"].to(self.engine['scene']._xyz.device)
            self.engine['scene']._rotation[indices] = snapshot["scene_rotation"].to(self.engine['scene']._rotation.device)
            self.engine['scene']._scaling[indices] = snapshot["scene_scaling_raw"].to(self.engine['scene']._scaling.device)
            self.engine['scene']._opacity[indices] = snapshot["scene_opacity_raw"].to(self.engine['scene']._opacity.device)
            self.engine['feature']._xyz[indices] = snapshot["feature_xyz"].to(self.engine['feature']._xyz.device)
            self.engine['feature']._rotation[indices] = snapshot["feature_rotation"].to(self.engine['feature']._rotation.device)
            self.engine['feature']._scaling[indices] = snapshot["feature_scaling_raw"].to(self.engine['feature']._scaling.device)
            self.engine['feature']._opacity[indices] = snapshot["feature_opacity_raw"].to(self.engine['feature']._opacity.device)

    def _push_apply_history(self, before_snapshot, after_snapshot):
        self.apply_undo_stack.append({"before": before_snapshot, "after": after_snapshot})
        if len(self.apply_undo_stack) > self.max_apply_history:
            self.apply_undo_stack = self.apply_undo_stack[-self.max_apply_history:]
        self.apply_redo_stack = []

    def _prepare_history_navigation(self):
        if self._active_group_has_cuboid():
            self._cancel_active_group()
            self._clear_active_ui_state()
            self._clear_cluster_selection()

    def _undo_last_apply(self):
        self._prepare_history_navigation()
        if len(self.apply_undo_stack) == 0:
            self._set_cluster_status("undo: no earlier applied version")
            return
        entry = self.apply_undo_stack.pop()
        self._restore_state_snapshot(entry["before"])
        self.apply_redo_stack.append(entry)
        self._set_cluster_status(f"undo applied ({len(self.apply_undo_stack)} left)")

    def _redo_last_apply(self):
        self._prepare_history_navigation()
        if len(self.apply_redo_stack) == 0:
            self._set_cluster_status("redo: no later applied version")
            return
        entry = self.apply_redo_stack.pop()
        self._restore_state_snapshot(entry["after"])
        self.apply_undo_stack.append(entry)
        if len(self.apply_undo_stack) > self.max_apply_history:
            self.apply_undo_stack = self.apply_undo_stack[-self.max_apply_history:]
        self._set_cluster_status(f"redo applied ({len(self.apply_redo_stack)} forward)")

    def _clear_active_ui_state(self):
        self._end_cuboid_drag()
        self.cuboid_manipulation = False
        if dpg.does_item_exist("_CuboidManipulation"):
            dpg.set_value("_CuboidManipulation", False)

    def _cancel_preview_for_selection_change(self):
        self._end_cuboid_drag()
        if self._active_group_has_cuboid() and self.active_group.get("dirty_transform", False):
            self._cancel_active_group()

    def _set_active_group_from_source_parts(self, source_parts, success_status, empty_status):
        score_thres = 0.0 if self.last_score_threshold is None else float(self.last_score_threshold)
        rebuilt = self._build_active_group([self._clone_part(part) for part in source_parts], score_thres)
        if rebuilt is None:
            self.active_group = None
            self._mark_part_list_dirty()
            self._clear_active_ui_state()
            self._update_cluster_status_widget()
            self._set_cluster_status(empty_status)
            return False
        self.active_group = rebuilt
        self._mark_part_list_dirty()
        self._set_cluster_status(success_status)
        return True

    def _remove_pending_part_by_fingerprint(self, fingerprint):
        pending_idx = self._find_pending_part_index(fingerprint)
        if pending_idx is None:
            self._set_cluster_status("remove pending: part not found")
            return
        del self.pending_parts[pending_idx]
        self._mark_part_list_dirty()
        self._set_cluster_status(f"removed pending part ({len(self.pending_parts)} pending left)")

    def _move_pending_part_to_active(self, fingerprint):
        pending_idx = self._find_pending_part_index(fingerprint)
        if pending_idx is None:
            self._set_cluster_status("move to active: pending part not found")
            return
        part = self._clone_part(self.pending_parts[pending_idx])
        self._cancel_preview_for_selection_change()
        if self._active_group_has_cuboid():
            source_parts = [self._clone_part(src) for src in self.active_group.get("source_parts", [])]
            existing = {src.get("fingerprint") for src in source_parts}
            if fingerprint in existing:
                del self.pending_parts[pending_idx]
                self._mark_part_list_dirty()
                self._set_cluster_status("move to active: part already active")
                return
            source_parts.append(part)
            if self._set_active_group_from_source_parts(
                source_parts,
                success_status="moved pending part to active",
                empty_status="move to active cleared selection",
            ):
                del self.pending_parts[pending_idx]
                self._mark_part_list_dirty()
                self._update_cluster_status_widget()
            return
        if self._set_active_group_from_source_parts(
            [part],
            success_status="created active cuboid from pending part",
            empty_status="move to active: no gaussians passed ScoreThres",
        ):
            del self.pending_parts[pending_idx]
            self._mark_part_list_dirty()
            self._update_cluster_status_widget()

    def _remove_active_part_by_fingerprint(self, fingerprint, move_to_pending=False):
        if not self._active_group_has_cuboid():
            self._set_cluster_status("active row action: no active cuboid")
            return
        source_idx = self._find_active_source_part_index(fingerprint)
        if source_idx is None:
            self._set_cluster_status("active row action: part not found")
            return
        self._cancel_preview_for_selection_change()
        source_parts = [self._clone_part(part) for part in self.active_group.get("source_parts", [])]
        source_idx = None
        for idx, part in enumerate(source_parts):
            if part.get("fingerprint") == fingerprint:
                source_idx = idx
                break
        if source_idx is None:
            self._set_cluster_status("active row action: part not found")
            return
        removed_part = self._clone_part(source_parts[source_idx])
        kept_parts = source_parts[:source_idx] + source_parts[source_idx + 1:]
        if move_to_pending and self._find_pending_part_index(fingerprint) is None:
            self.pending_part_counter += 1
            removed_part["part_id"] = self.pending_part_counter
            self.pending_parts.append(removed_part)
            self._mark_part_list_dirty()
        elif not move_to_pending:
            pending_idx = self._find_pending_part_index(fingerprint)
            if pending_idx is not None:
                del self.pending_parts[pending_idx]
                self._mark_part_list_dirty()
        if len(kept_parts) == 0:
            self.active_group = None
            self._mark_part_list_dirty()
            self._clear_active_ui_state()
            if move_to_pending:
                self._set_cluster_status("moved active part to pending (active cleared)")
            else:
                self._set_cluster_status("removed active part (active cleared)")
            return
        self._set_active_group_from_source_parts(
            kept_parts,
            success_status="moved active part to pending" if move_to_pending else "removed active part",
            empty_status="active selection cleared",
        )

    def _add_pending_to_active_group(self):
        if len(self.pending_parts) == 0:
            self._set_cluster_status("add active: no pending parts")
            return
        if not self._active_group_has_cuboid():
            self._set_cluster_status("add active: no active cuboid")
            return
        self._cancel_preview_for_selection_change()
        source_parts = [self._clone_part(part) for part in self.active_group.get("source_parts", [])]
        existing = {part.get("fingerprint") for part in source_parts}
        additions = [self._clone_part(part) for part in self.pending_parts if part.get("fingerprint") not in existing]
        self._clear_pending_selection()
        if len(additions) == 0:
            self._set_cluster_status("add active: no new pending parts")
            return
        source_parts.extend(additions)
        self._set_active_group_from_source_parts(
            source_parts,
            success_status=f"added {len(additions)} pending part(s) to active",
            empty_status="add active cleared selection",
        )

    def _remove_pending_from_active_group(self):
        if len(self.pending_parts) == 0:
            self._set_cluster_status("remove active: no pending parts")
            return
        if not self._active_group_has_cuboid():
            self._set_cluster_status("remove active: no active cuboid")
            return
        self._cancel_preview_for_selection_change()
        source_parts = [self._clone_part(part) for part in self.active_group.get("source_parts", [])]
        pending_fingerprints = {part.get("fingerprint") for part in self.pending_parts}
        kept_parts = [part for part in source_parts if part.get("fingerprint") not in pending_fingerprints]
        removed_count = len(source_parts) - len(kept_parts)
        if removed_count == 0:
            self._set_cluster_status("remove active: no pending parts matched active")
            return
        self._set_active_group_from_source_parts(
            kept_parts,
            success_status=f"removed {removed_count} active part(s)",
            empty_status="removed all active parts",
        )

    def _apply_and_clear_active_group(self):
        if not self._active_group_has_cuboid():
            return
        active_group = self.active_group
        history_pushed = False
        if active_group["dirty_transform"]:
            before_snapshot = self._capture_state_snapshot(active_group["global_indices"], use_active_group_base=active_group)
            self._commit_active_group()
            after_snapshot = self._capture_state_snapshot(active_group["global_indices"])
            self._push_apply_history(before_snapshot, after_snapshot)
            history_pushed = True
        self._clear_active_ui_state()
        self._clear_cluster_selection()
        if history_pushed:
            self._set_cluster_status(f"active transform applied ({len(self.apply_undo_stack)} undo)")
        else:
            self._set_cluster_status("active transform applied")

    def _clear_active_group_and_restore(self):
        if self._active_group_has_cuboid():
            self._cancel_active_group()
        self._clear_active_ui_state()
        self._clear_cluster_selection()
        self._set_cluster_status("selection cleared and restored")

    def _rebuild_active_group_for_score(self, score_thres):
        if not self._active_group_has_cuboid():
            return
        source_parts = [self._clone_part(part) for part in self.active_group.get("source_parts", [])]
        self._cancel_preview_for_selection_change()
        rebuilt = self._build_active_group(source_parts, score_thres)
        if rebuilt is None:
            self.active_group = None
            self._mark_part_list_dirty()
            self._clear_active_ui_state()
            self._update_cluster_status_widget()
            self._set_cluster_status("selection cleared by ScoreThres")
            return
        self.active_group = rebuilt
        self._mark_part_list_dirty()
        self._set_cluster_status(f"active: {len(self.active_group.get('parts', []))} part(s)")

    def _cuboid_corners_world(self, active_group):
        center = self._current_cuboid_center(active_group)
        axes = self._current_cuboid_axes(active_group)
        extents = self._current_cuboid_extents(active_group)
        signs = torch.tensor([
            [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
            [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
        ], device=center.device, dtype=center.dtype)
        local = signs * extents.unsqueeze(0)
        return center.unsqueeze(0) + local @ axes.transpose(0, 1)

    def _screen_space_translation_delta(self, view_camera, center, delta_xy):
        dist = torch.norm(center - view_camera.camera_center).item()
        world_per_pixel_y = 2.0 * dist * math.tan(view_camera.FoVy * 0.5) / max(view_camera.image_height, 1)
        world_per_pixel_x = 2.0 * dist * math.tan(view_camera.FoVx * 0.5) / max(view_camera.image_width, 1)
        camera_rot = self.camera.rot.as_matrix()
        right = torch.tensor(camera_rot[:3, 0], device=center.device, dtype=center.dtype)
        up = torch.tensor(camera_rot[:3, 1], device=center.device, dtype=center.dtype)
        return right * (delta_xy[0] * world_per_pixel_x) + up * (delta_xy[1] * world_per_pixel_y)

    def _screen_space_rotation_delta(self, delta_xy, device, dtype):
        camera_rot = self.camera.rot.as_matrix()
        up = camera_rot[:3, 1]
        right = camera_rot[:3, 0]
        rotvec = up * np.radians(-0.3 * float(delta_xy[0])) + right * np.radians(0.3 * float(delta_xy[1]))
        delta_rot = R.from_rotvec(rotvec).as_matrix().astype(np.float32)
        return torch.from_numpy(delta_rot).to(device=device, dtype=dtype)

    def _draw_marquee_overlay(self):
        if self.marquee_overlay_rect is None or not dpg.does_item_exist("_render_image"):
            return
        image_rect_min = dpg.get_item_rect_min("_render_image")
        if image_rect_min is None:
            return
        start_xy, end_xy = self.marquee_overlay_rect
        pmin = (float(image_rect_min[0] + min(start_xy[0], end_xy[0])), float(image_rect_min[1] + min(start_xy[1], end_xy[1])))
        pmax = (float(image_rect_min[0] + max(start_xy[0], end_xy[0])), float(image_rect_min[1] + max(start_xy[1], end_xy[1])))
        dpg.draw_rectangle(pmin, pmax, color=(255, 220, 64, 255), fill=(255, 220, 64, 32), thickness=2.0, parent="_overlay_drawlist")

    def _draw_active_cuboid_overlay(self, view_camera):
        if not self._active_group_has_cuboid() or not dpg.does_item_exist("_overlay_drawlist") or not dpg.does_item_exist("_render_image"):
            return
        image_rect_min = dpg.get_item_rect_min("_render_image")
        image_rect_size = dpg.get_item_rect_size("_render_image")
        if image_rect_min is None or image_rect_size is None:
            return
        active_group = self.active_group
        corners = self._cuboid_corners_world(active_group)
        projected_xy, depth, valid = self._project_points(view_camera, corners)
        if int(valid.sum().item()) != 8:
            return
        within_image = ((projected_xy[:, 0] >= 0.0) & (projected_xy[:, 0] < float(image_rect_size[0])) & (projected_xy[:, 1] >= 0.0) & (projected_xy[:, 1] < float(image_rect_size[1])))
        visible_count = int(within_image.sum().item())
        all_inside = visible_count == 8
        points = [(float(image_rect_min[0] + x), float(image_rect_min[1] + y)) for x, y in projected_xy.detach().cpu().tolist()]
        depths_cpu = depth.detach().cpu().tolist()
        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]
        face_color = (64, 160, 255, 36)
        edge_color = (64, 160, 255, 220)
        sorted_faces = sorted(faces, key=lambda face: sum(depths_cpu[idx] for idx in face) / len(face), reverse=True)
        if all_inside:
            for face in sorted_faces:
                dpg.draw_polygon([points[idx] for idx in face], fill=face_color, color=(0, 0, 0, 0), parent="_overlay_drawlist")
        for start, end in edges:
            dpg.draw_line(points[start], points[end], color=edge_color, thickness=2.0, parent="_overlay_drawlist")
        if self.cuboid_manipulation and self.transform_mode == "Extend":
            for handle in self._iter_extend_handles(active_group):
                projected_handle, _, valid_handle = self._project_points(view_camera, handle["world"].unsqueeze(0))
                if int(valid_handle.sum().item()) == 0:
                    continue
                hx, hy = projected_handle[0].detach().cpu().tolist()
                center = (float(image_rect_min[0] + hx), float(image_rect_min[1] + hy))
                if handle["kind"] == "vertex":
                    radius = 4.0
                    color = (255, 220, 64, 255)
                elif handle["kind"] == "edge":
                    radius = 3.0
                    color = (255, 255, 255, 220)
                else:
                    radius = 5.0
                    color = (64, 255, 196, 255)
                dpg.draw_circle(center, radius, color=color, fill=(color[0], color[1], color[2], 140), parent="_overlay_drawlist")

    def _refresh_overlay_drawlist(self, view_camera):
        if not dpg.does_item_exist("_overlay_drawlist") or not dpg.does_item_exist("_render_image"):
            return
        dpg.delete_item("_overlay_drawlist", children_only=True)
        self._draw_active_cuboid_overlay(view_camera)
        self._draw_marquee_overlay()

    def _invalidate_cluster_cache(self, reason="scale changed"):
        self._cancel_preview_for_selection_change()
        self.cluster_request_generation += 1
        self.component_cache = {}
        self.cluster_cache = None
        self.cluster_cache_scale = None
        self.cluster_pending_result = None
        self.cluster_job_error = None
        self.cluster_thread = None
        self.cluster_point_colors = None
        self.seg_score = None
        self._clear_cluster_selection()
        self._set_cluster_status(f"stale: {reason}", state="stale")

    def _generate_label_codes(self, count, seed=0):
        rng = np.random.default_rng(seed)
        codes = rng.choice([-1.0, 1.0], size=(count, self.opt.FEATURE_DIM)).astype(np.float32)
        norms = np.linalg.norm(codes, axis=1, keepdims=True) + 1e-6
        return codes / norms

    def _ensure_label_color_capacity(self, required_count):
        if required_count <= self.label_to_color.shape[0]:
            return
        extra = np.random.rand(required_count - self.label_to_color.shape[0], 3)
        self.label_to_color = np.concatenate([self.label_to_color, extra], axis=0)

    def _ensure_label_code_capacity(self, required_count):
        if required_count <= self.label_to_code.shape[0]:
            return
        extra = self._generate_label_codes(
            required_count - self.label_to_code.shape[0],
            seed=self.label_to_code.shape[0] + self.opt.FEATURE_DIM,
        )
        self.label_to_code = np.concatenate([self.label_to_code, extra], axis=0)

    def _build_cluster_colors_from_labels(self, labels):
        labels = np.asarray(labels)
        colors = np.zeros((labels.shape[0], 3), dtype=np.float32)
        valid = labels >= 0
        if np.any(valid):
            self._ensure_label_color_capacity(int(labels[valid].max()) + 1)
            colors[valid] = self.label_to_color[labels[valid]]
        return colors

    def _build_cluster_codes_from_labels(self, labels):
        labels = np.asarray(labels)
        codes = np.zeros((labels.shape[0], self.opt.FEATURE_DIM), dtype=np.float32)
        valid = labels >= 0
        if np.any(valid):
            self._ensure_label_code_capacity(int(labels[valid].max()) + 1)
            codes[valid] = self.label_to_code[labels[valid]]
        return codes

    def _cluster_worker(self, generation, scale, normed_point_features, point_xyz, point_sh0_rgb, method, point_mesh_vertex_idx=None, mesh_vertex_adjacency=None):
        try:
            method_key = str(method).lower()
            if method_key == "normalizedcut":
                result = cluster_gaussians_normalized_cut(
                    point_xyz,
                    normed_point_features,
                    sample_size=self.cluster_sample_size,
                    graph_k=self.cluster_graph_k,
                    propagation_k=max(8, min(self.cluster_graph_k, 32)),
                    max_clusters=self.cluster_max_clusters,
                    min_cluster_size=self.cluster_min_cluster_size,
                    cut_threshold=self.cluster_cut_threshold,
                    spatial_weight=self.cluster_spatial_weight,
                    sh0_rgb=point_sh0_rgb,
                    sh0_color_weight=self.cluster_sh0_color_weight,
                    sh0_color_sigma=self.cluster_sh0_color_sigma,
                    point_mesh_vertex_idx=point_mesh_vertex_idx,
                    mesh_vertex_adjacency=mesh_vertex_adjacency,
                    mesh_weight=self.cluster_mesh_weight,
                )
                self.cluster_pending_result = {
                    "generation": generation,
                    "scale": scale,
                    "labels": result["labels"],
                    "confidence": result["confidence"],
                    "counts": result["counts"],
                    "num_clusters": result["num_clusters"],
                    "method": "NormalizedCut",
                }
                return

            if method_key == "hdbscanrefined":
                result = cluster_gaussians_hdbscan_refined(
                    point_xyz,
                    normed_point_features,
                    sample_size=self.cluster_sample_size,
                    hdbscan_min_cluster_size=10,
                    hdbscan_epsilon=0.01,
                    graph_k=self.cluster_graph_k,
                    min_cluster_size=self.cluster_min_cluster_size,
                    spatial_scale=2.5,
                    spatial_weight=self.cluster_spatial_weight,
                    sh0_rgb=point_sh0_rgb,
                    sh0_color_weight=self.cluster_sh0_color_weight,
                    sh0_color_sigma=self.cluster_sh0_color_sigma,
                    random_state=0,
                )
                self.cluster_pending_result = {
                    "generation": generation,
                    "scale": scale,
                    "labels": result["labels"],
                    "confidence": result["confidence"],
                    "counts": result["counts"],
                    "num_clusters": result["num_clusters"],
                    "method": "HDBSCANRefined",
                }
                return

            result = cluster_gaussians_hdbscan(
                normed_point_features,
                sample_size=self.cluster_sample_size,
                hdbscan_min_cluster_size=10,
                hdbscan_epsilon=0.01,
                random_state=0,
            )
            self.cluster_pending_result = {
                "generation": generation,
                "scale": scale,
                "labels": result["labels"],
                "confidence": result["confidence"],
                "counts": result["counts"],
                "num_clusters": result["num_clusters"],
                "method": "HDBSCAN",
            }
        except Exception as exc:
            self.cluster_job_error = {
                "generation": generation,
                "message": str(exc),
            }

    def _validate_scene_feature_alignment(self, context="loaded data"):
        scene_count = int(self.engine['scene'].get_xyz.shape[0])
        feature_count = int(self.engine['feature'].get_xyz.shape[0])
        if scene_count == feature_count:
            return
        raise RuntimeError(
            f"Scene/feature Gaussian count mismatch during {context}: "
            f"scene has {scene_count} points from iteration {self.opt.SCENE_GAUSSIAN_ITERATION} "
            f"({self.opt.SCENE_PCD_PATH}), but feature has {feature_count} points from iteration "
            f"{self.opt.FEATURE_GAUSSIAN_ITERATION} ({self.opt.FEATURE_PCD_PATH}). "
            "Use matching scene/feature checkpoints, or retrain contrastive features for the selected "
            "scene checkpoint. For MILo covered_desk, the valid pair is scene iteration 30000 with "
            "feature iteration 10000."
        )

    def _start_recluster(self, scale):
        if self.cluster_cache_state == "clustering":
            return
        print(f"Clustering in 3D with {self.cluster_method}...")
        self._validate_scene_feature_alignment(context="clustering")

        point_features = self.engine['feature'].get_point_features.detach()
        normed_point_features = torch.nn.functional.normalize(point_features, dim=-1, p=2)
        gates = self.engine['scale_gate'](torch.tensor([scale], device=point_features.device)).detach().squeeze(0)
        scale_conditioned = normed_point_features * gates.unsqueeze(0)
        normed_conditioned = torch.nn.functional.normalize(scale_conditioned, dim=-1, p=2)

        num_points = int(normed_conditioned.shape[0])
        full_features_cpu = normed_conditioned.detach().cpu().numpy().astype(np.float32, copy=False)
        full_xyz_cpu = self.engine['scene'].get_xyz.detach().cpu().numpy().astype(np.float32, copy=False)
        full_sh0_rgb_cpu = None
        if self.cluster_sh0_color_weight > 0:
            full_sh0_rgb = SH2RGB(self.engine['scene'].get_features[:, 0, :].detach()).clamp(0.0, 1.0)
            if full_sh0_rgb.shape[0] != num_points:
                raise RuntimeError(
                    f"Scene Gaussian count ({full_sh0_rgb.shape[0]}) does not match feature Gaussian count ({num_points}) for SH0 color weighting."
                )
            full_sh0_rgb_cpu = full_sh0_rgb.detach().cpu().numpy().astype(np.float32, copy=False)

        mesh_prior = None
        if self.cluster_mesh_weight > 0:
            mesh_prior = load_mesh_prior_for_points(
                full_xyz_cpu,
                model_path=self.opt.MODEL_PATH,
                mesh_path=self.cluster_mesh_path,
            )
            if mesh_prior is None:
                print("Mesh-aware clustering requested, but no mesh was found. Falling back to xyz+feature graph.")
        self.cluster_mesh_prior = mesh_prior

        self.cluster_request_generation += 1
        generation = self.cluster_request_generation
        self.cluster_pending_result = None
        self.cluster_job_error = None
        self.cluster_cache = None
        self.cluster_cache_scale = None
        self.cluster_point_colors = None
        self.seg_score = None
        self.component_cache = {}
        self._clear_cluster_selection()
        self.cluster_thread = threading.Thread(
            target=self._cluster_worker,
            args=(
                generation,
                scale,
                full_features_cpu,
                full_xyz_cpu,
                full_sh0_rgb_cpu,
                self.cluster_method,
                None if mesh_prior is None else mesh_prior['point_mesh_vertex_idx'],
                None if mesh_prior is None else mesh_prior['mesh_vertex_adjacency'],
            ),
            daemon=True,
        )
        self.cluster_thread.start()
        self._set_cluster_status(f"clustering {self.cluster_method} scale={scale:.3f}", state="clustering")

    def _poll_cluster_job(self):
        if self.cluster_job_error is not None:
            generation = self.cluster_job_error["generation"]
            message = self.cluster_job_error["message"]
            self.cluster_job_error = None
            if generation == self.cluster_request_generation:
                self.cluster_thread = None
                self.cluster_cache = None
                self.cluster_cache_scale = None
                self.cluster_point_colors = None
                self._set_cluster_status(f"error: {message}", state="stale")
                print(f"Clustering failed: {message}")
            return

        if self.cluster_pending_result is None:
            return

        result = self.cluster_pending_result
        self.cluster_pending_result = None
        if result["generation"] != self.cluster_request_generation:
            return

        labels_np = result["labels"]
        confidence_np = result["confidence"]
        point_colors = self._build_cluster_colors_from_labels(labels_np)
        point_codes = self._build_cluster_codes_from_labels(labels_np)
        self._ensure_label_code_capacity(result["num_clusters"])
        cluster_codes = self.label_to_code[:result["num_clusters"]].astype(np.float32, copy=True)
        self.cluster_cache = {
            "scale": result["scale"],
            "method": result.get("method", self.cluster_method),
            "labels": torch.from_numpy(labels_np).cuda().long(),
            "confidence": torch.from_numpy(confidence_np).cuda().float(),
            "counts": result["counts"],
            "num_clusters": result["num_clusters"],
            "point_colors_torch": torch.from_numpy(point_colors).cuda().float(),
            "point_codes_torch": torch.from_numpy(point_codes).cuda().float(),
            "cluster_codes_torch": torch.from_numpy(cluster_codes).cuda().float(),
            "global_residue_hidden_mask": None,
            "global_residue_hidden_count": 0,
        }
        self.cluster_cache_scale = result["scale"]
        self.cluster_point_colors = point_colors
        self.cluster_thread = None
        self._set_cluster_status(
            f"ready: {result['num_clusters']} {result.get('method', self.cluster_method)} clusters @ scale={result['scale']:.3f}",
            state="ready",
        )
        print("Clustering finished.")

    def _refresh_cluster_colors(self):
        if self.cluster_cache is None:
            self.cluster_point_colors = None
            return
        labels_np = self.cluster_cache["labels"].detach().cpu().numpy()
        point_colors = self._build_cluster_colors_from_labels(labels_np)
        self.cluster_point_colors = point_colors
        self.cluster_cache["point_colors_torch"] = torch.from_numpy(point_colors).cuda().float()
        self._update_cluster_status_widget()


    def _build_selection_mask(self, score_thres):
        if self._active_group_has_cuboid():
            selection_mask = self.active_group.get("visible_selection_mask", self.active_group.get("selection_mask"))
            global_hidden_mask = self._current_global_residue_hidden_mask()
            if selection_mask is not None and global_hidden_mask is not None:
                selection_mask = selection_mask & ~global_hidden_mask
            return selection_mask
        return self._pending_selection_mask(score_thres)

    def _build_parts_for_debug_render(self, score_thres):
        if self._active_group_has_cuboid():
            if not self.show_active:
                return []
            debug_parts = []
            for part in self.active_group.get("parts", []):
                indices = self._part_visible_indices(part, active=True)
                if indices is None or indices.numel() == 0:
                    continue
                part_copy = self._clone_part(part)
                part_copy["global_indices"] = indices
                debug_parts.append(part_copy)
            return debug_parts
        debug_parts = []
        for part in self.pending_parts:
            indices = self._filter_part_indices(part, score_thres)
            if indices is None or indices.numel() == 0:
                continue
            part_copy = self._clone_part(part)
            part_copy["global_indices"] = indices
            debug_parts.append(part_copy)
        return debug_parts

    def _build_selection_debug_render(self, view_camera, score_thres):
        if not self.selection_highlight_enabled:
            return None
        parts = self._build_parts_for_debug_render(score_thres)
        if len(parts) == 0:
            return None
        scene_count = self.engine['scene'].get_xyz.shape[0]
        color_buffer = torch.zeros((scene_count, 3), device=self.engine['scene'].get_xyz.device, dtype=torch.float32)
        selection_mask = torch.zeros((scene_count,), device=self.engine['scene'].get_xyz.device, dtype=torch.bool)
        active_mode = self._active_group_has_cuboid()
        for part_idx, part in enumerate(parts):
            indices = part.get("global_indices")
            if indices is None or indices.numel() == 0:
                continue
            color = torch.tensor(self._part_display_color(part_idx, active=active_mode), device=color_buffer.device, dtype=color_buffer.dtype)
            color_buffer[indices] = color
            selection_mask[indices] = True
        if int(selection_mask.sum().item()) == 0:
            return None
        return render(view_camera, self.engine['scene'], self.opt, self.bg_color, override_color=color_buffer, filtered_mask=~selection_mask)["render"].permute(1, 2, 0)

    def _project_points(self, view_camera, points):

            ones = torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)
            points_h = torch.cat([points, ones], dim=1)
            clip = points_h @ view_camera.full_proj_transform
            view = points_h @ view_camera.world_view_transform
            w = clip[:, 3:4]
            valid = (torch.abs(w[:, 0]) > 1e-6) & (view[:, 2] > 0.2)
            ndc = torch.zeros_like(clip[:, :3])
            ndc[valid] = clip[valid, :3] / w[valid]
            x = ((ndc[:, 0] + 1.0) * float(view_camera.image_width) - 1.0) * 0.5
            y = ((ndc[:, 1] + 1.0) * float(view_camera.image_height) - 1.0) * 0.5
            depth = torch.norm(points - view_camera.camera_center.unsqueeze(0), dim=1)
            return torch.stack([x, y], dim=1), depth, valid

    def _pick_cluster_from_code_buffer(self, mouse_xy):
        if self.cluster_cache is None or self.cluster_cache_state != "ready":
            return None
        if self.cluster_pick_feature_render is None:
            return None

        cluster_codes = self.cluster_cache.get("cluster_codes_torch")
        if cluster_codes is None or cluster_codes.shape[0] == 0:
            return None

        render = self.cluster_pick_feature_render
        height, width = render.shape[:2]
        px = int(np.clip(round(float(mouse_xy[0])), 0, width - 1))
        py = int(np.clip(round(float(mouse_xy[1])), 0, height - 1))
        radius = int(self.cluster_pick_patch_radius)
        x0 = max(px - radius, 0)
        x1 = min(px + radius + 1, width)
        y0 = max(py - radius, 0)
        y1 = min(py + radius + 1, height)
        patch = render[y0:y1, x0:x1]
        if patch.numel() == 0:
            return None

        patch_vectors = patch.reshape(-1, patch.shape[-1])
        norms = torch.norm(patch_vectors, dim=1)
        valid = norms > 1e-6
        if int(valid.sum().item()) == 0:
            return None

        yy = torch.arange(y0, y1, device=patch.device, dtype=patch.dtype)
        xx = torch.arange(x0, x1, device=patch.device, dtype=patch.dtype)
        grid_y, grid_x = torch.meshgrid(yy, xx, indexing='ij')
        spatial_dist2 = (grid_x - float(px)) ** 2 + (grid_y - float(py)) ** 2
        sigma = max(float(radius), 1.0)
        spatial_weights = torch.exp(-0.5 * spatial_dist2.reshape(-1) / (sigma * sigma))

        patch_vectors = patch_vectors[valid]
        norms = norms[valid]
        spatial_weights = spatial_weights[valid]
        patch_vectors = patch_vectors / norms.unsqueeze(-1)

        sims = patch_vectors @ cluster_codes.t()
        top_k = 2 if cluster_codes.shape[0] > 1 else 1
        top_scores, top_ids = sims.topk(k=top_k, dim=1)
        top1_scores = torch.clamp(top_scores[:, 0], min=0.0)
        top1_ids = top_ids[:, 0]
        if top_k == 1:
            margin = top1_scores
        else:
            margin = torch.clamp(top_scores[:, 0] - top_scores[:, 1], min=0.0)

        visibility_weight = torch.clamp(norms, min=0.0, max=1.0)
        pixel_weights = spatial_weights * visibility_weight * top1_scores * (margin + 1e-4)
        if float(pixel_weights.sum().item()) <= 1e-8:
            return None

        label_scores = torch.zeros(cluster_codes.shape[0], device=patch.device, dtype=patch.dtype)
        label_scores.scatter_add_(0, top1_ids, pixel_weights)
        best_score, best_id = torch.max(label_scores, dim=0)
        total_score = torch.clamp(label_scores.sum(), min=1e-6)
        if float(best_score.item()) <= 1e-8:
            return None

        self.last_pick_confidence = float((best_score / total_score).item())
        return int(best_id.item())

    def _pick_cluster_from_mouse(self, mouse_xy, view_camera, scene_outputs):
        if self.cluster_cache is None or self.cluster_cache_state != "ready":
            return None

        cluster_id = self._pick_cluster_from_code_buffer(mouse_xy)
        if cluster_id is not None:
            return cluster_id

        visibility = scene_outputs["visibility_filter"]
        if int(visibility.sum().item()) == 0:
            return None

        points = self.engine['scene'].get_xyz[visibility]
        radii = scene_outputs["radii"][visibility].float()
        labels = self.cluster_cache["labels"][visibility]

        projected_xy, depth, valid = self._project_points(view_camera, points)
        projected_xy = projected_xy[valid]
        depth = depth[valid]
        radii = radii[valid]
        labels = labels[valid]

        if projected_xy.shape[0] == 0:
            return None

        mouse = torch.tensor(mouse_xy, device=projected_xy.device, dtype=projected_xy.dtype).unsqueeze(0)
        diff = projected_xy - mouse
        dist2 = (diff * diff).sum(dim=-1)
        tolerance = torch.maximum(radii, torch.full_like(radii, self.pick_tolerance_px))
        candidates = (labels >= 0) & (dist2 <= tolerance * tolerance)
        if int(candidates.sum().item()) == 0:
            return None

        candidate_indices = torch.nonzero(candidates, as_tuple=False).squeeze(-1)
        candidate_labels = labels[candidate_indices]
        unique_labels = torch.unique(candidate_labels)
        if unique_labels.numel() == 1:
            return int(unique_labels.item())

        candidate_depth = depth[candidate_indices]
        candidate_dist2 = dist2[candidate_indices]
        candidate_radii = radii[candidate_indices]

        clicked_color = None
        if self.rendered_cluster is not None:
            h, w = self.rendered_cluster.shape[:2]
            px = int(np.clip(round(float(mouse_xy[0])), 0, w - 1))
            py = int(np.clip(round(float(mouse_xy[1])), 0, h - 1))
            x0 = max(px - 2, 0)
            x1 = min(px + 3, w)
            y0 = max(py - 2, 0)
            y1 = min(py + 3, h)
            patch = self.rendered_cluster[y0:y1, x0:x1]
            if patch.numel() > 0:
                patch = patch.reshape(-1, patch.shape[-1])
                clicked_color = patch.median(dim=0).values

        coverage_scores = []
        color_scores = []
        label_list = []
        for label_value in unique_labels.tolist():
            mask = candidate_labels == label_value
            label_dist2 = candidate_dist2[mask]
            label_depth = candidate_depth[mask]
            label_radii = torch.clamp(candidate_radii[mask], min=1.0)
            weights = torch.exp(-0.5 * label_dist2 / (label_radii * label_radii)) / (label_depth + 1e-4)
            coverage_scores.append(weights.sum())

            if clicked_color is None:
                color_scores.append(torch.tensor(0.0, device=weights.device))
            else:
                label_color = torch.tensor(self.label_to_color[label_value], device=clicked_color.device, dtype=clicked_color.dtype)
                color_score = 1.0 - torch.norm(label_color - clicked_color) / math.sqrt(3.0)
                color_scores.append(torch.clamp(color_score, min=0.0, max=1.0))
            label_list.append(label_value)

        coverage_scores = torch.stack(coverage_scores)
        color_scores = torch.stack(color_scores)

        def _normalize(scores):
            score_min = scores.min()
            score_max = scores.max()
            if torch.abs(score_max - score_min) < 1e-8:
                return torch.ones_like(scores)
            return (scores - score_min) / (score_max - score_min)

        coverage_norm = _normalize(coverage_scores)
        color_norm = _normalize(color_scores) if clicked_color is not None else torch.zeros_like(coverage_norm)
        final_scores = 0.65 * coverage_norm + 0.35 * color_norm
        if not torch.isfinite(final_scores).any():
            fallback_depth = depth.clone()
            fallback_depth[~candidates] = float("inf")
            picked = int(torch.argmin(fallback_depth).item())
            if not torch.isfinite(fallback_depth[picked]):
                return None
            return int(labels[picked].item())

        best_idx = int(torch.argmax(final_scores).item())
        return int(label_list[best_idx])

    def _pick_seed_gaussian_from_mouse(self, mouse_xy, view_camera, scene_outputs, cluster_id):
        if self.cluster_cache is None or self.cluster_cache_state != "ready":
            return None

        visibility = scene_outputs["visibility_filter"]
        if int(visibility.sum().item()) == 0:
            return None

        global_indices = torch.nonzero(visibility, as_tuple=False).squeeze(-1)
        points = self.engine['scene'].get_xyz[global_indices]
        radii = scene_outputs["radii"][visibility].float()
        labels = self.cluster_cache["labels"][global_indices]

        projected_xy, depth, valid = self._project_points(view_camera, points)
        global_indices = global_indices[valid]
        projected_xy = projected_xy[valid]
        depth = depth[valid]
        radii = radii[valid]
        labels = labels[valid]
        if projected_xy.shape[0] == 0:
            return None

        mask = labels == int(cluster_id)
        if int(mask.sum().item()) == 0:
            return None

        mouse = torch.tensor(mouse_xy, device=projected_xy.device, dtype=projected_xy.dtype).unsqueeze(0)
        diff = projected_xy - mouse
        dist2 = (diff * diff).sum(dim=-1)
        tolerance = torch.maximum(radii, torch.full_like(radii, self.pick_tolerance_px))
        weights = torch.exp(-0.5 * dist2 / torch.clamp(tolerance * tolerance, min=1.0)) / (depth + 1e-4)
        weights[~mask] = 0.0
        if float(weights.max().item()) <= 1e-8:
            masked_dist2 = dist2.clone()
            masked_dist2[~mask] = float("inf")
            best_idx = int(torch.argmin(masked_dist2).item())
            if not torch.isfinite(masked_dist2[best_idx]):
                return None
        else:
            best_idx = int(torch.argmax(weights).item())
        return int(global_indices[best_idx].item())

    def _get_local_mouse_pos(self):
        if not dpg.does_item_exist("_render_image"):
            return None
        if not dpg.is_item_hovered("_render_image"):
            return None
        rect_min = dpg.get_item_rect_min("_render_image")
        rect_size = dpg.get_item_rect_size("_render_image")
        mouse = dpg.get_mouse_pos(local=False)
        local_x = mouse[0] - rect_min[0]
        local_y = mouse[1] - rect_min[1]
        if local_x < 0 or local_y < 0 or local_x >= rect_size[0] or local_y >= rect_size[1]:
            return None
        return np.array([local_x, local_y], dtype=np.float32)

    def _queue_cluster_pick_request(self, mouse_xy=None):
        if mouse_xy is None:
            mouse_xy = self._get_local_mouse_pos()
        if mouse_xy is not None:
            self.cluster_pick_request = np.asarray(mouse_xy, dtype=np.float32)

    def _queue_marquee_request(self, start_xy, end_xy):
        self.marquee_request = (np.asarray(start_xy, dtype=np.float32), np.asarray(end_xy, dtype=np.float32))

    def _consume_cluster_pick_request(self, view_camera, scene_outputs):
        if self.cluster_pick_request is None:
            return
        mouse_xy = self.cluster_pick_request
        self.cluster_pick_request = None
        cluster_id = self._pick_cluster_from_mouse(mouse_xy, view_camera, scene_outputs)
        if cluster_id is None:
            self._set_pick_feedback(None, "no candidate")
            return
        seed_index = self._pick_seed_gaussian_from_mouse(mouse_xy, view_camera, scene_outputs, cluster_id)
        self.last_picked_seed_index = seed_index
        part = self._build_part_from_seed(cluster_id, seed_index)
        if part is None:
            self._set_pick_feedback(cluster_id, "no component")
            return
        result = self._add_pending_part(part, clear_existing=not self.edit_multiclick)
        if result == "added":
            self._set_pick_feedback(cluster_id, "added")
            self._set_cluster_status(f"pending: {len(self.pending_parts)} part(s)")
        elif result == "removed":
            self._set_pick_feedback(cluster_id, "removed")
            self._set_cluster_status(f"pending: {len(self.pending_parts)} part(s)")
        else:
            self._set_pick_feedback(cluster_id, "noop")

    def _consume_marquee_pick_request(self, view_camera, scene_outputs):
        if self.marquee_request is None or self.cluster_cache is None or self.cluster_cache_state != "ready":
            return
        start_xy, end_xy = self.marquee_request
        self.marquee_request = None
        x0, x1 = sorted([float(start_xy[0]), float(end_xy[0])])
        y0, y1 = sorted([float(start_xy[1]), float(end_xy[1])])
        if (x1 - x0) < self.marquee_threshold_px or (y1 - y0) < self.marquee_threshold_px:
            return

        visibility = scene_outputs["visibility_filter"]
        if int(visibility.sum().item()) == 0:
            self._set_pick_feedback(None, "marquee empty")
            return
        global_indices = torch.nonzero(visibility, as_tuple=False).squeeze(-1)
        points = self.engine['scene'].get_xyz[global_indices]
        labels = self.cluster_cache["labels"][global_indices]
        projected_xy, _, valid = self._project_points(view_camera, points)
        global_indices = global_indices[valid]
        labels = labels[valid]
        projected_xy = projected_xy[valid]
        if projected_xy.shape[0] == 0:
            self._set_pick_feedback(None, "marquee empty")
            return

        in_rect = (
            (projected_xy[:, 0] >= x0)
            & (projected_xy[:, 0] <= x1)
            & (projected_xy[:, 1] >= y0)
            & (projected_xy[:, 1] <= y1)
            & (labels >= 0)
        )
        if int(in_rect.sum().item()) == 0:
            self._set_pick_feedback(None, "marquee empty")
            return

        score_thres = 0.0 if self.last_score_threshold is None else float(self.last_score_threshold)
        candidate_parts = []
        seen_fingerprints = set()
        candidate_cluster_ids = torch.unique(labels[in_rect]).detach().cpu().tolist()
        for cluster_id in candidate_cluster_ids:
            cluster_visible_mask = labels == int(cluster_id)
            cluster_visible_global = global_indices[cluster_visible_mask]
            cluster_visible_xy = projected_xy[cluster_visible_mask]
            preferred = global_indices[in_rect & (labels == int(cluster_id))]
            cache = self._get_cluster_component_cache(cluster_id, preferred_global_indices=preferred)
            if cache is None:
                continue
            visible_in_cache_mask = torch.isin(cluster_visible_global, cache["indices"])
            if int(visible_in_cache_mask.sum().item()) == 0:
                continue
            visible_global = cluster_visible_global[visible_in_cache_mask]
            visible_xy = cluster_visible_xy[visible_in_cache_mask]
            if score_thres > 0.0:
                visible_conf = self.cluster_cache["confidence"][visible_global]
                score_mask = visible_conf >= score_thres
                if int(score_mask.sum().item()) == 0:
                    continue
                visible_global = visible_global[score_mask]
                visible_xy = visible_xy[score_mask]
            visible_component_ids = np.asarray([cache["global_to_component"].get(int(idx), -1) for idx in visible_global.detach().cpu().tolist()], dtype=np.int32)
            if visible_component_ids.size == 0:
                continue
            visible_xy_np = visible_xy.detach().cpu().numpy()
            inside_mask_np = (
                (visible_xy_np[:, 0] >= x0)
                & (visible_xy_np[:, 0] <= x1)
                & (visible_xy_np[:, 1] >= y0)
                & (visible_xy_np[:, 1] <= y1)
            )
            candidate_component_ids = np.unique(visible_component_ids[inside_mask_np])
            for component_id in candidate_component_ids.tolist():
                if component_id < 0:
                    continue
                component_visible_mask = visible_component_ids == component_id
                if not np.any(component_visible_mask):
                    continue
                component_xy = visible_xy_np[component_visible_mask]
                component_inside = inside_mask_np[component_visible_mask]
                if component_xy.shape[0] == 0:
                    continue
                if component_xy.shape[0] == 1:
                    core_inside = bool(component_inside[0])
                else:
                    robust_center = np.median(component_xy, axis=0, keepdims=True)
                    dist2 = np.sum((component_xy - robust_center) ** 2, axis=1)
                    core_count = max(1, int(math.ceil(component_xy.shape[0] * 0.95)))
                    core_indices = np.argsort(dist2)[:core_count]
                    core_inside = bool(np.all(component_inside[core_indices]))
                if not core_inside:
                    continue
                component_indices = cache["component_indices"][int(component_id)]
                part = self._build_part_from_component_indices(cluster_id, component_indices, seed_index=None, source="marquee")
                if part is None:
                    continue
                fingerprint = part.get("fingerprint")
                if fingerprint in seen_fingerprints:
                    continue
                seen_fingerprints.add(fingerprint)
                candidate_parts.append(part)

        if len(candidate_parts) == 0:
            self._set_pick_feedback(None, "marquee no candidate")
            return

        added_count = 0
        removed_count = 0
        if not self.edit_multiclick:
            existing_fingerprints = {part.get("fingerprint") for part in self.pending_parts}
            candidate_fingerprints = {part.get("fingerprint") for part in candidate_parts}
            if len(candidate_fingerprints) > 0 and candidate_fingerprints == existing_fingerprints:
                removed_count = len(self.pending_parts)
                self.pending_parts = []
                self._mark_part_list_dirty()
            else:
                removed_count = len(self.pending_parts)
                self.pending_parts = []
                for part in candidate_parts:
                    result = self._add_pending_part(part, clear_existing=False)
                    if result == "added":
                        added_count += 1
                    elif result == "removed":
                        removed_count += 1
        else:
            for part in candidate_parts:
                result = self._add_pending_part(part, clear_existing=False)
                if result == "added":
                    added_count += 1
                elif result == "removed":
                    removed_count += 1

        if added_count > 0 or removed_count > 0:
            if added_count > 0 and removed_count > 0:
                result_text = f"marquee +{added_count} -{removed_count}"
            elif added_count > 0:
                result_text = f"marquee +{added_count}"
            else:
                result_text = f"marquee -{removed_count}"
            self._set_pick_feedback(None, result_text)
            self._set_cluster_status(f"pending: {len(self.pending_parts)} part(s)")
        else:
            self._set_pick_feedback(None, "marquee no candidate")

    def _create_active_group_from_pending(self):
        if len(self.pending_parts) == 0:
            self._set_cluster_status("create cuboid: no pending parts")
            return
        if self._active_group_has_cuboid():
            self._set_cluster_status("active exists: use Add Active or Remove Active")
            return
        self._set_cluster_status("building connected cuboid")
        score_thres = 0.0 if self.last_score_threshold is None else float(self.last_score_threshold)
        active_group = self._build_active_group([self._clone_part(part) for part in self.pending_parts], score_thres)
        if active_group is None:
            self._set_cluster_status("no gaussians passed ScoreThres")
            return
        self.active_group = active_group
        self._clear_pending_selection()
        self._set_cluster_status(f"active cuboid: {len(self.active_group.get('parts', []))} part(s)")

    def _iter_extend_handles(self, active_group):
        signs = [
            np.array([-1, -1, -1], dtype=np.float32), np.array([1, -1, -1], dtype=np.float32),
            np.array([1, 1, -1], dtype=np.float32), np.array([-1, 1, -1], dtype=np.float32),
            np.array([-1, -1, 1], dtype=np.float32), np.array([1, -1, 1], dtype=np.float32),
            np.array([1, 1, 1], dtype=np.float32), np.array([-1, 1, 1], dtype=np.float32),
        ]
        corners = self._cuboid_corners_world(active_group)
        center = self._current_cuboid_center(active_group)
        axes = self._current_cuboid_axes(active_group)
        extents = self._current_cuboid_extents(active_group)
        for idx, sign in enumerate(signs):
            yield {"kind": "vertex", "sign": sign, "world": corners[idx]}
        edge_defs = [
            (0, 1, np.array([0, -1, -1], dtype=np.float32)), (1, 2, np.array([1, 0, -1], dtype=np.float32)),
            (2, 3, np.array([0, 1, -1], dtype=np.float32)), (3, 0, np.array([-1, 0, -1], dtype=np.float32)),
            (4, 5, np.array([0, -1, 1], dtype=np.float32)), (5, 6, np.array([1, 0, 1], dtype=np.float32)),
            (6, 7, np.array([0, 1, 1], dtype=np.float32)), (7, 4, np.array([-1, 0, 1], dtype=np.float32)),
            (0, 4, np.array([-1, -1, 0], dtype=np.float32)), (1, 5, np.array([1, -1, 0], dtype=np.float32)),
            (2, 6, np.array([1, 1, 0], dtype=np.float32)), (3, 7, np.array([-1, 1, 0], dtype=np.float32)),
        ]
        for start, end, sign in edge_defs:
            yield {"kind": "edge", "sign": sign, "world": 0.5 * (corners[start] + corners[end])}
        face_defs = [
            np.array([-1, 0, 0], dtype=np.float32), np.array([1, 0, 0], dtype=np.float32),
            np.array([0, -1, 0], dtype=np.float32), np.array([0, 1, 0], dtype=np.float32),
            np.array([0, 0, -1], dtype=np.float32), np.array([0, 0, 1], dtype=np.float32),
        ]
        for sign in face_defs:
            dim = int(np.argmax(np.abs(sign)))
            world = center + axes[:, dim] * extents[dim] * float(sign[dim])
            yield {"kind": "face", "sign": sign, "world": world}

    def _pick_extend_handle(self, local_xy):
        if not self._active_group_has_cuboid() or self.last_view_camera is None:
            return None
        best_handle = None
        best_score = float("inf")
        mouse = np.asarray(local_xy, dtype=np.float32)
        for handle in self._iter_extend_handles(self.active_group):
            projected_xy, _, valid = self._project_points(self.last_view_camera, handle["world"].unsqueeze(0))
            if int(valid.sum().item()) == 0:
                continue
            point = projected_xy[0].detach().cpu().numpy()
            dist = float(np.linalg.norm(point - mouse))
            if handle["kind"] == "face":
                tolerance = float(self.face_handle_tolerance_px)
            else:
                tolerance = float(self.extend_handle_tolerance_px)
            score = dist / max(tolerance, 1e-6)
            if score < best_score:
                best_score = score
                best_handle = dict(handle)
        if best_handle is None or best_score > 1.0:
            return None
        return best_handle

    def _begin_cuboid_drag(self, button):
        if not self.cuboid_manipulation or not self._active_group_has_cuboid():
            return False
        local_xy = self._get_local_mouse_pos()
        if local_xy is None:
            return False
        self.drag_last_local_xy = local_xy
        self.extend_handle = None
        if button == dpg.mvMouseButton_Middle:
            self.drag_mode = "translate"
            return True
        if self.transform_mode == "Extend":
            handle = self._pick_extend_handle(local_xy)
            if handle is None:
                return False
            self.extend_handle = handle
            self.drag_mode = "extend"
            return True
        self.drag_mode = "rotate"
        return True

    def _update_cuboid_drag(self, local_xy):
        if self.drag_mode is None or not self._active_group_has_cuboid() or local_xy is None:
            return False
        delta_xy = local_xy - self.drag_last_local_xy
        if abs(float(delta_xy[0])) < 1e-5 and abs(float(delta_xy[1])) < 1e-5:
            return False
        active_group = self.active_group
        if self.drag_mode == "translate":
            delta_translation = self._screen_space_translation_delta(self.last_view_camera, self._current_cuboid_center(active_group), delta_xy)
            active_group["group_translation"] = active_group["group_translation"] + delta_translation
        elif self.drag_mode == "extend" and self.extend_handle is not None:
            sign = torch.tensor(self.extend_handle["sign"], device=active_group["base_center"].device, dtype=active_group["base_center"].dtype)
            current_axes = self._current_cuboid_axes(active_group)
            current_extents = self._current_cuboid_extents(active_group)
            old_scale = self._current_group_scale_local(active_group).clone()
            new_scale = old_scale.clone()
            active_dims = torch.nonzero(torch.abs(sign) > 0, as_tuple=False).squeeze(-1)
            if active_dims.numel() == 0:
                return False
            updated = False
            for dim in active_dims.tolist():
                center = self._current_cuboid_center(active_group)
                axis_point = center + current_axes[:, dim] * current_extents[dim] * sign[dim]
                projected, _, valid = self._project_points(self.last_view_camera, torch.stack([center, axis_point], dim=0))
                if int(valid.sum().item()) < 2:
                    continue
                screen_dir = projected[1] - projected[0]
                screen_len = torch.norm(screen_dir).item()
                if screen_len < 1e-4:
                    continue
                delta_pixels = float(torch.dot(torch.tensor(delta_xy, device=screen_dir.device, dtype=screen_dir.dtype), screen_dir / screen_len).item())
                scale_factor = max(0.05, 1.0 + delta_pixels / max(screen_len, 1.0))
                new_scale[dim] = torch.clamp(old_scale[dim] * scale_factor, min=0.05)
                updated = True
            if not updated:
                return False
            old_extents = active_group["base_extents"] * old_scale
            new_extents = active_group["base_extents"] * new_scale
            delta_local = sign * (new_extents - old_extents)
            active_group["group_translation"] = active_group["group_translation"] + delta_local @ current_axes.transpose(0, 1)
            active_group["group_scale_local"] = new_scale
        else:
            delta_rotation = self._screen_space_rotation_delta(delta_xy, active_group["base_center"].device, active_group["base_center"].dtype)
            active_group["group_rotation"] = delta_rotation @ active_group["group_rotation"]
        active_group["dirty_transform"] = True
        self._update_group_preview(active_group)
        self.drag_last_local_xy = local_xy
        self._set_cluster_status("editing active cuboid")
        return True

    def _end_cuboid_drag(self):
        self.drag_mode = None
        self.drag_last_local_xy = None
        self.extend_handle = None

    def _make_export_elements(self, indices, cluster_idx_values):
        scene = self.engine['scene']
        feature = self.engine['feature']
        xyz = scene._xyz[indices].detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = scene._features_dc[indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = scene._features_rest[indices].detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = scene._opacity[indices].detach().cpu().numpy()
        scaling = scene._scaling[indices].detach().cpu().numpy()
        rotation = scene._rotation[indices].detach().cpu().numpy()
        affinity = feature.get_point_features[indices].detach().contiguous().cpu().numpy()
        cluster_idx_values = np.asarray(cluster_idx_values, dtype=np.int32).reshape(-1)

        dtype_full = [(attribute, 'f4') for attribute in scene.construct_list_of_attributes()]
        dtype_full.extend([(f'affinity_f_{i}', 'f4') for i in range(affinity.shape[1])])
        dtype_full.append(('cluster_idx', 'i4'))
        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        elements['x'] = xyz[:, 0]
        elements['y'] = xyz[:, 1]
        elements['z'] = xyz[:, 2]
        elements['nx'] = normals[:, 0]
        elements['ny'] = normals[:, 1]
        elements['nz'] = normals[:, 2]
        for i in range(f_dc.shape[1]):
            elements[f'f_dc_{i}'] = f_dc[:, i]
        for i in range(f_rest.shape[1]):
            elements[f'f_rest_{i}'] = f_rest[:, i]
        elements['opacity'] = opacities[:, 0]
        for i in range(scaling.shape[1]):
            elements[f'scale_{i}'] = scaling[:, i]
        for i in range(rotation.shape[1]):
            elements[f'rot_{i}'] = rotation[:, i]
        for i in range(affinity.shape[1]):
            elements[f'affinity_f_{i}'] = affinity[:, i]
        elements['cluster_idx'] = cluster_idx_values
        return elements

    def _save_active_gaussians(self):
        if not self._active_group_has_cuboid() or len(self.active_group.get('parts', [])) == 0:
            self._set_cluster_status('no active gaussians to save')
            return
        self._commit_active_group_if_dirty()
        export_root = os.path.join(self.opt.MODEL_PATH, 'cluster_exports', time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(export_root, exist_ok=True)
        parts = self.active_group.get('parts', [])
        export_parts = []
        for part_idx, part in enumerate(parts):
            indices = self._part_export_indices(part)
            if indices is None or indices.numel() == 0:
                continue
            export_parts.append((part_idx, part, indices))
        if len(export_parts) == 0:
            self._set_cluster_status('no active gaussians to save')
            return
        full_indices = torch.unique(torch.cat([indices for _, _, indices in export_parts], dim=0))
        cluster_lookup = np.full((full_indices.shape[0],), -1, dtype=np.int32)
        full_indices_cpu = full_indices.detach().cpu().numpy().astype(np.int64, copy=False)
        index_to_row = {int(idx): row for row, idx in enumerate(full_indices_cpu.tolist())}
        for part_idx, _, indices in export_parts:
            for global_index in indices.detach().cpu().numpy().astype(np.int64, copy=False).tolist():
                row = index_to_row.get(int(global_index))
                if row is not None:
                    cluster_lookup[row] = part_idx
        full_elements = self._make_export_elements(full_indices, cluster_lookup)
        PlyData([PlyElement.describe(full_elements, 'vertex')]).write(os.path.join(export_root, 'gaussian_full.ply'))
        for part_idx, _, indices in export_parts:
            elements = self._make_export_elements(indices, np.full((indices.shape[0],), part_idx, dtype=np.int32))
            PlyData([PlyElement.describe(elements, 'vertex')]).write(os.path.join(export_root, f'gaussian_part{part_idx}.ply'))
        print(f'Saved active gaussians to {export_root}')
        self._set_cluster_status(f'saved active gaussians: {len(export_parts)} part(s)')

    def _save_full_scene_gaussians(self):
        self._commit_active_group_if_dirty()
        scene_count = self.engine['scene']._xyz.shape[0]
        feature_count = self.engine['feature'].get_point_features.shape[0]
        count = int(min(scene_count, feature_count))
        if count <= 0:
            self._set_cluster_status('no scene gaussians to save')
            return
        export_root = os.path.join(self.opt.MODEL_PATH, 'scene_exports', time.strftime('%Y%m%d_%H%M%S'))
        os.makedirs(export_root, exist_ok=True)
        indices = torch.arange(count, device=self.engine['scene']._xyz.device, dtype=torch.long)
        if self.cluster_cache is not None and self.cluster_cache.get('labels') is not None and self.cluster_cache['labels'].shape[0] >= count:
            cluster_idx_values = self.cluster_cache['labels'][:count].detach().cpu().numpy().astype(np.int32, copy=False)
        else:
            cluster_idx_values = np.full((count,), -1, dtype=np.int32)
        elements = self._make_export_elements(indices, cluster_idx_values)
        output_path = os.path.join(export_root, 'gaussian_scene_full.ply')
        PlyData([PlyElement.describe(elements, 'vertex')]).write(output_path)
        print(f'Saved full scene gaussians to {output_path}')
        self._set_cluster_status(f'saved full scene gaussians: {count}')

    def register_dpg(self):

        
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(self.width, self.height, self.render_buffer, format=dpg.mvFormat_Float_rgb, tag="_texture")

        ### register window
        with dpg.window(tag="_primary_window", width=self.window_width+300, height=self.window_height):
            dpg.add_image("_texture", tag="_render_image")   # add the texture

        dpg.add_viewport_drawlist(front=True, tag="_overlay_drawlist")
        dpg.set_primary_window("_primary_window", True)

        # def callback_depth(sender, app_data):
            # self.img_mode = (self.img_mode + 1) % 4
            
        # --- interactive mode switch --- #
        def clickmode_callback(sender):
            self.clickmode_button = 1 - self.clickmode_button
        def clickmode_multi_callback(sender):
            self.clickmode_multi_button = dpg.get_value(sender)
            print("clickmode_multi_button = ", self.clickmode_multi_button)
        def preview_callback(sender):
            self.preview = dpg.get_value(sender)
            # print("binary_threshold_button = ", self.binary_threshold_button)
        def clear_edit():
            self.clear_edit = True
        def roll_back():
            self.roll_back = True
        def callback_segment3d():
            self.segment3d_flag = True
        def callback_save():
            self.save_flag = True
        def callback_reload():
            self.reload_flag = True
        def callback_cluster():
            self._cancel_preview_for_selection_change()
            self.recluster_requested = True
        def callback_recluster():
            self._cancel_preview_for_selection_change()
            self.recluster_requested = True
        def callback_reshuffle_color():
            self.label_to_color = np.random.rand(1000, 3)
            self._refresh_cluster_colors()
        def callback_cluster_method(sender):
            self.cluster_method = dpg.get_value(sender)
            self._invalidate_cluster_cache(reason=f"cluster method={self.cluster_method}")
        def callback_cluster_sample_size(sender):
            self.cluster_sample_size = int(max(1024, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_sample_size)
            self._invalidate_cluster_cache(reason=f"cluster sample={self.cluster_sample_size}")
        def callback_cluster_max_clusters(sender):
            self.cluster_max_clusters = int(max(2, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_max_clusters)
            self._invalidate_cluster_cache(reason=f"cluster max={self.cluster_max_clusters}")
        def callback_cluster_min_cluster_size(sender):
            self.cluster_min_cluster_size = int(max(8, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_min_cluster_size)
            self._invalidate_cluster_cache(reason=f"cluster min={self.cluster_min_cluster_size}")
        def callback_cluster_spatial_weight(sender):
            self.cluster_spatial_weight = float(max(0.0, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_spatial_weight)
            self._invalidate_cluster_cache(reason=f"cluster spatial weight={self.cluster_spatial_weight:.3f}")
        def callback_cluster_sh0_color_weight(sender):
            self.cluster_sh0_color_weight = float(max(0.0, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_sh0_color_weight)
            self._invalidate_cluster_cache(reason=f"cluster color weight={self.cluster_sh0_color_weight:.3f}")
        def callback_cluster_sh0_color_sigma(sender):
            self.cluster_sh0_color_sigma = float(max(1e-4, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_sh0_color_sigma)
            self._invalidate_cluster_cache(reason=f"cluster color sigma={self.cluster_sh0_color_sigma:.3f}")
        def callback_cluster_mesh_weight(sender):
            self.cluster_mesh_weight = float(max(0.0, dpg.get_value(sender)))
            dpg.set_value(sender, self.cluster_mesh_weight)
            self._invalidate_cluster_cache(reason=f"cluster mesh weight={self.cluster_mesh_weight:.3f}")
        def callback_hide_small_residue(sender):
            self.hide_small_residue = dpg.get_value(sender)
            self._on_residue_settings_changed(f"residue hide={'on' if self.hide_small_residue else 'off'}")
        def callback_residue_max_size(sender):
            self.residue_max_size = int(max(0, dpg.get_value(sender)))
            dpg.set_value(sender, self.residue_max_size)
            self._on_residue_settings_changed(f"residue max={self.residue_max_size}")
        def callback_residue_scope(sender):
            self.residue_scope = str(dpg.get_value(sender))
            self._on_residue_settings_changed(f"residue scope={self.residue_scope}")
        def callback_exclude_residue_export(sender):
            self.exclude_residue_on_active_export = dpg.get_value(sender)
            self._update_cluster_status_widget()
        def callback_cluster_click_select(sender):
            next_value = dpg.get_value(sender)
            if not next_value:
                self._commit_active_group_if_dirty()
                self.cluster_pick_request = None
                self.marquee_request = None
                self.marquee_overlay_rect = None
                self.right_dragging = False
            self.edit_click_select = next_value
            self._update_cluster_status_widget()
        def callback_cluster_multiclick(sender):
            self.edit_multiclick = dpg.get_value(sender)
            if not self.edit_multiclick and len(self.pending_parts) > 1:
                self.pending_parts = [self.pending_parts[-1]]
            self._update_cluster_status_widget()
        def callback_create_cuboid():
            self._create_active_group_from_pending()
        def callback_add_active():
            self._add_pending_to_active_group()
        def callback_remove_active():
            self._remove_pending_from_active_group()
        def callback_clear_active():
            self._clear_active_group_and_restore()
        def callback_cuboid_manipulation(sender):
            next_value = dpg.get_value(sender)
            if not next_value:
                self._commit_active_group_if_dirty()
                self._end_cuboid_drag()
            self.cuboid_manipulation = next_value
        def callback_show_active():
            self.show_active = not self.show_active
            self._update_cluster_status_widget()
        def callback_transform_mode(sender):
            self.transform_mode = dpg.get_value(sender)
            self.extend_handle = None
        def callback_apply_active():
            self._apply_and_clear_active_group()
        def callback_undo_apply():
            self._undo_last_apply()
        def callback_redo_apply():
            self._redo_last_apply()
        def callback_cancel_active():
            self._cancel_active_group()
        def callback_preview_max(sender):
            self.preview_max_gaussians = int(max(1, dpg.get_value(sender)))
            self.component_cache = {}
            if self._active_group_has_cuboid():
                self._rebuild_active_group_for_score(0.0 if self.last_score_threshold is None else self.last_score_threshold)
        def callback_cuboid_percentile(sender):
            percentile = float(np.clip(dpg.get_value(sender), 50.0, 100.0))
            self.cuboid_percentile = percentile
            if dpg.get_value(sender) != percentile:
                dpg.set_value(sender, percentile)
            if self._active_group_has_cuboid():
                self._recompute_active_group_cuboid()
        def callback_save_active():
            self.save_active_flag = True
        def callback_save_full_scene():
            self.save_full_scene_flag = True

        def render_mode_rgb_callback(sender):
            self.render_mode_rgb = dpg.get_value(sender)
        def render_mode_pca_callback(sender):
            self.render_mode_pca = dpg.get_value(sender)
        def render_mode_cluster_callback(sender):
            self.render_mode_cluster = dpg.get_value(sender)
        with dpg.theme(tag="_ShowActiveButtonOnTheme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (45, 125, 78), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (58, 150, 94), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (36, 102, 63), category=dpg.mvThemeCat_Core)
        with dpg.theme(tag="_ShowActiveButtonOffTheme"):
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (70, 70, 70), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (90, 90, 90), category=dpg.mvThemeCat_Core)
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (55, 55, 55), category=dpg.mvThemeCat_Core)

        # control window
        with dpg.window(label="Control", tag="_control_window", width=300, height=550, pos=[self.window_width+10, 0]):

            dpg.add_text("Mouse position: click anywhere to start.", tag="pos_item")
            dpg.add_slider_float(label="Scale", default_value=0.5,
                                 min_value=0.0, max_value=1.0, tag="_Scale")
            dpg.add_slider_float(label="ScoreThres", default_value=0.0,
                                 min_value=0.0, max_value=1.0, tag="_ScoreThres")

            dpg.add_text("Render option", tag="render")
            with dpg.group(horizontal=True):
                dpg.add_checkbox(label="RGB", default_value=self.render_mode_rgb, callback=render_mode_rgb_callback, user_data="Some Data")
                dpg.add_checkbox(label="PCA", default_value=self.render_mode_pca, callback=render_mode_pca_callback, user_data="Some Data")
                dpg.add_checkbox(label="3D CLUSTER", default_value=self.render_mode_cluster, callback=render_mode_cluster_callback, user_data="Some Data")

            dpg.add_text("Cluster Edit", tag="cluster_edit")
            dpg.add_button(label="Recluster", callback=callback_recluster, user_data="Some Data", width=-1)
            dpg.add_text("Method")
            dpg.add_combo(("HDBSCAN", "HDBSCANRefined", "NormalizedCut"), label="", default_value=self.cluster_method, callback=callback_cluster_method, tag="_ClusterMethod", width=-1)
            dpg.add_text("Cache: stale\nClusters: none", tag="_cluster_summary_line1")
            dpg.add_text("Last: none", tag="_cluster_summary_line2")
            dpg.add_text("Pending: 0 | Active: 0 | G: 0\nPick: none", tag="_cluster_summary_line3")

            dpg.add_checkbox(label="Cuboid Manipulation", callback=callback_cuboid_manipulation, user_data="Some Data", tag="_CuboidManipulation")
            dpg.add_radio_button(items=["Rotate", "Extend"], default_value=self.transform_mode, callback=callback_transform_mode, horizontal=True, tag="_TransformMode")

            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=False, borders_outerV=False, borders_innerH=False, borders_outerH=False):
                for _ in range(3):
                    dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_button(label="Create\nCuboid", callback=callback_create_cuboid, user_data="Some Data", tag="_CreateCuboidButton", width=-1, height=38)
                    dpg.add_button(label="Add\nActive", callback=callback_add_active, user_data="Some Data", tag="_AddActiveButton", width=-1, height=38)
                    dpg.add_button(label="Remove\nActive", callback=callback_remove_active, user_data="Some Data", tag="_RemoveActiveButton", width=-1, height=38)
                with dpg.table_row():
                    dpg.add_button(label="Clear\nActive", callback=callback_clear_active, user_data="Some Data", width=-1, height=38)
                    dpg.add_button(label="Apply", callback=callback_apply_active, user_data="Some Data", width=-1, height=38)
                    dpg.add_button(label="Cancel", callback=callback_cancel_active, user_data="Some Data", width=-1, height=38)
                with dpg.table_row():
                    dpg.add_button(label="Undo", callback=callback_undo_apply, user_data="Some Data", width=-1, height=38)
                    dpg.add_button(label="Redo", callback=callback_redo_apply, user_data="Some Data", width=-1, height=38)
                    dpg.add_button(label="Show\nActive", callback=lambda: callback_show_active(), tag="_ShowActiveButton", width=-1, height=38)
                with dpg.table_row():
                    dpg.add_button(label="Save\nActive", callback=callback_save_active, user_data="Some Data", width=-1, height=38)
                    dpg.add_button(label="Save\nFull", callback=callback_save_full_scene, user_data="Some Data", width=-1, height=38)
                    dpg.add_spacer()

            with dpg.tab_bar(tag="_cluster_part_tabs"):
                with dpg.tab(label="Pending (0)", tag="_PendingTab"):
                    with dpg.child_window(tag="_cluster_pending_rows", height=150, border=True):
                        dpg.add_text("(none)")
                with dpg.tab(label="Active (0)", tag="_ActiveTab"):
                    with dpg.child_window(tag="_cluster_active_rows", height=150, border=True):
                        dpg.add_text("(none)")

            dpg.add_text("Preview Max / Cuboid %")
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=False, borders_outerV=False, borders_innerH=False, borders_outerH=False):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_input_int(label="", default_value=self.preview_max_gaussians, min_value=1, min_clamped=True, step=10000, tag="_PreviewMaxGaussians", callback=callback_preview_max, width=-1)
                    dpg.add_input_float(label="", default_value=self.cuboid_percentile, min_value=50.0, max_value=100.0, min_clamped=True, max_clamped=True, step=0.5, step_fast=1.0, format="%.1f", on_enter=True, tag="_CuboidPercentile", callback=callback_cuboid_percentile, width=-1)

            dpg.add_text("Residue", tag="_ResidueHeader")
            dpg.add_text("Residue: active 0 | global 0", tag="_cluster_residue_line")
            dpg.add_checkbox(label="Hide Small Residue", default_value=self.hide_small_residue, callback=callback_hide_small_residue, tag="_HideSmallResidue")
            dpg.add_text("Residue Max / Scope")
            with dpg.table(header_row=False, resizable=False, policy=dpg.mvTable_SizingStretchProp, borders_innerV=False, borders_outerV=False, borders_innerH=False, borders_outerH=False):
                dpg.add_table_column()
                dpg.add_table_column()
                with dpg.table_row():
                    dpg.add_input_int(label="", default_value=self.residue_max_size, min_value=0, min_clamped=True, step=50, tag="_ResidueMaxSize", callback=callback_residue_max_size, width=-1)
                    dpg.add_combo(("Global", "Active", "Both"), label="", default_value=self.residue_scope, callback=callback_residue_scope, tag="_ResidueScope", width=-1)
            dpg.add_checkbox(label="Exclude Residue\non Active Export", default_value=self.exclude_residue_on_active_export, callback=callback_exclude_residue_export, tag="_ExcludeResidueExport")


            def callback(sender, app_data, user_data):
                self.load_model = False
                file_data = app_data["selections"]
                file_names = []
                for key in file_data.keys():
                    file_names.append(key)

                self.opt.ply_file = file_data[file_names[0]]

                # if not self.load_model:
                print("loading model file...")
                self.engine.load_ply(self.opt.ply_file)
                self.do_pca()   # calculate new self.proj_mat after loading new .ply file
                print("loading model file done.")
                self.load_model = True
        if self.debug:
            with dpg.collapsing_header(label="Debug"):
                dpg.add_separator()
                dpg.add_text("Camera Pose:")
                dpg.add_text(str(self.camera.pose), tag="_log_pose")


        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            delta = app_data
            self.camera.scale(delta)
            self.update_camera = True
            if self.debug:
                dpg.set_value("_log_pose", str(self.camera.pose))
        

        def start_left_drag():
            self.mouse_pos = dpg.get_mouse_pos(local=False)
            self.left_dragging = True
            self.moving = False
            handled = self._begin_cuboid_drag(dpg.mvMouseButton_Left)
            if handled:
                return
            if self.cuboid_manipulation and self._active_group_has_cuboid() and self.transform_mode == "Extend":
                return
            self.moving = True


        def stop_left_drag():
            self.left_dragging = False
            self.moving = False
            if self.drag_mode in {"rotate", "extend"}:
                self._end_cuboid_drag()


        def start_middle_drag():
            self.mouse_pos = dpg.get_mouse_pos(local=False)
            self.middle_dragging = True
            self.moving_middle = False
            if self._begin_cuboid_drag(dpg.mvMouseButton_Middle):
                return
            self.moving_middle = True


        def stop_middle_drag():
            self.middle_dragging = False
            self.moving_middle = False
            if self.drag_mode == "translate":
                self._end_cuboid_drag()


        def move_handler(sender, pos, user):
            local_xy = self._get_local_mouse_pos()
            if self.right_dragging and self.right_drag_start_local_xy is not None:
                current_xy = local_xy if local_xy is not None else self.right_drag_current_local_xy
                if current_xy is not None:
                    self.right_drag_current_local_xy = np.asarray(current_xy, dtype=np.float32)
                    self.marquee_overlay_rect = (self.right_drag_start_local_xy.copy(), self.right_drag_current_local_xy.copy())
                    self.update_camera = True
            if self.drag_mode is not None:
                if self._update_cuboid_drag(local_xy):
                    self.update_camera = True
            elif self.moving and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.orbit(-dx*30, dy*30)
                    self.update_camera = True

            if self.drag_mode is None and self.moving_middle and dpg.is_item_focused("_primary_window"):
                dx = self.mouse_pos[0] - pos[0]
                dy = self.mouse_pos[1] - pos[1]
                if dx != 0.0 or dy != 0.0:
                    self.camera.pan(-dx*20, dy*20)
                    self.update_camera = True
            
            self.mouse_pos = pos


        def start_right_drag():
            xy = dpg.get_mouse_pos(local=False)
            dpg.set_value("pos_item", f"Mouse position = ({xy[0]}, {xy[1]})")
            local_xy = self._get_local_mouse_pos()
            if local_xy is None:
                self.right_dragging = False
                self.right_drag_start_local_xy = None
                self.right_drag_current_local_xy = None
                self.marquee_overlay_rect = None
                return
            self.right_dragging = True
            self.right_drag_start_local_xy = np.asarray(local_xy, dtype=np.float32)
            self.right_drag_current_local_xy = np.asarray(local_xy, dtype=np.float32)
            self.marquee_overlay_rect = None


        def stop_right_drag():
            if not self.right_dragging:
                return
            end_xy = self._get_local_mouse_pos()
            if end_xy is None:
                end_xy = self.right_drag_current_local_xy
            start_xy = self.right_drag_start_local_xy
            self.right_dragging = False
            self.right_drag_start_local_xy = None
            self.right_drag_current_local_xy = None
            self.marquee_overlay_rect = None
            if start_xy is None or end_xy is None:
                return
            end_xy = np.asarray(end_xy, dtype=np.float32)
            drag_distance = float(np.linalg.norm(end_xy - start_xy))
            if self.cluster_cache_state == "ready":
                if drag_distance >= self.marquee_threshold_px:
                    self._queue_marquee_request(start_xy, end_xy)
                else:
                    self._queue_cluster_pick_request(mouse_xy=end_xy)
                self.update_camera = True
                return
            if drag_distance < self.marquee_threshold_px and self.clickmode_button:
                self.new_click_xy = end_xy
                self.new_click = True


        with dpg.handler_registry():
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Left, callback=lambda: start_left_drag())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Left, callback=lambda: stop_left_drag())
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Middle, callback=lambda: start_middle_drag())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Middle, callback=lambda: stop_middle_drag())
            dpg.add_mouse_move_handler(callback=lambda s, a, u:move_handler(s, a, u))
            dpg.add_mouse_click_handler(dpg.mvMouseButton_Right, callback=lambda: start_right_drag())
            dpg.add_mouse_release_handler(dpg.mvMouseButton_Right, callback=lambda: stop_right_drag())
            
        dpg.create_viewport(title="Gaussian-Splatting-Viewer", width=self.window_width+320, height=self.window_height, resizable=False)

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core)
                dpg.add_theme_style(dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core)
        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        dpg.show_viewport()


    def render(self):
        while dpg.is_dearpygui_running():
            # update texture every frame
            # TODO : fetch rgb and depth
            if self.load_model:
                cam = self.construct_camera()
                self.fetch_data(cam)
            dpg.render_dearpygui_frame()


    def construct_camera(
        self,
    ) -> Camera:
        if self.camera.rot_mode == 1:
            pose = self.camera.pose_movecenter
        elif self.camera.rot_mode == 0:
            pose = self.camera.pose_objcenter

        R = pose[:3, :3]
        t = pose[:3, 3]

        ss = math.pi / 180.0
        fovy = self.camera.fovy * ss

        fy = fov2focal(fovy, self.height)
        fovx = focal2fov(fy, self.width)

        cam = Camera(
            colmap_id=0,
            R=R,
            T=t,
            FoVx=fovx,
            FoVy=fovy,
            image=torch.zeros([3, self.height, self.width]),
            gt_alpha_mask=None,
            image_name=None,
            uid=0,
        )
        cam.feature_height, cam.feature_width = self.height, self.width
        return cam
    
    def cluster_in_3D(self):
        # try:
        #     self.engine['scene'].roll_back()
        #     self.engine['feature'].roll_back()
        # except:
        #     pass
        point_features = self.engine['feature'].get_point_features

        scale_conditioned_point_features = torch.nn.functional.normalize(point_features, dim = -1, p = 2) * self.gates.unsqueeze(0)

        normed_point_features = torch.nn.functional.normalize(scale_conditioned_point_features, dim = -1, p = 2)

        sampled_point_features = scale_conditioned_point_features[torch.rand(scale_conditioned_point_features.shape[0]) > 0.98]

        normed_sampled_point_features = sampled_point_features / torch.norm(sampled_point_features, dim = -1, keepdim = True)

        clusterer = HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01, allow_single_cluster = False)

        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())

        cluster_centers = torch.zeros(len(np.unique(cluster_labels)), normed_sampled_point_features.shape[-1])
        for i in range(0, len(np.unique(cluster_labels))):
            cluster_centers[i] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i-1].mean(dim = 0), dim = -1)

        self.seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), normed_point_features.cpu())
        self.cluster_point_colors = self.label_to_color[self.seg_score.argmax(dim = -1).cpu().numpy()]
        # self.cluster_point_colors[self.seg_score.max(dim = -1)[0].detach().cpu().numpy() < 0.5] = (0,0,0)


    def pca(self, X, n_components=3):
        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16,
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)
        idx = torch.argsort(eigenvalues, descending=True)
        proj_mat = eigenvectors[:, idx[:n_components]]

        return proj_mat
    

    def do_pca(self):
        sems = self.engine['feature'].get_point_features.clone().squeeze()
        N, C = sems.shape
        torch.manual_seed(0)
        randint = torch.randint(0, N, [200_000])
        sems /= (torch.norm(sems, dim=1, keepdim=True) + 1e-6)
        sem_chosen = sems[randint, :]
        self.proj_mat = self.pca(sem_chosen, n_components=3)
        print("project mat initialized !")


    @torch.no_grad()
    def fetch_data(self, view_camera):
        scale = dpg.get_value('_Scale')
        score_thres = dpg.get_value('_ScoreThres')
        self.last_view_camera = view_camera

        if self._active_group_has_cuboid() and (self.cluster_pick_request is not None or self.marquee_request is not None) and self.active_group["dirty_transform"]:
            self._commit_active_group()

        if self._active_group_has_cuboid() and abs(float(score_thres) - float(self.active_group.get("score_thres", score_thres))) > 1e-6:
            self._rebuild_active_group_for_score(score_thres)

        self.last_score_threshold = score_thres
        if self.last_scale_value is None:
            self.last_scale_value = scale
        elif abs(scale - self.last_scale_value) > 1e-6:
            self.last_scale_value = scale
            if self.cluster_cache is not None or self.cluster_cache_state == "clustering":
                self._commit_active_group_if_dirty()
                self._invalidate_cluster_cache(reason=f"scale changed to {scale:.3f}")

        if self.recluster_requested:
            self.recluster_requested = False
            self._start_recluster(scale)
        self._poll_cluster_job()

        hidden_mask = self._combined_hidden_mask()

        scene_outputs = render(view_camera, self.engine['scene'], self.opt, self.bg_color, filtered_mask=hidden_mask)
        feature_outputs = render_contrastive_feature(view_camera, self.engine['feature'], self.opt, self.bg_feature, filtered_mask=hidden_mask)
        self.last_scene_outputs = scene_outputs
        self.last_feature_outputs = feature_outputs

        self.rendered_cluster = None
        self.cluster_pick_feature_render = None
        cluster_ready = self.cluster_cache is not None and self.cluster_cache_state == "ready"
        need_cluster_color_render = cluster_ready and (self.render_mode_cluster or self.cluster_pick_request is not None)
        if need_cluster_color_render:
            self.rendered_cluster = render(
                view_camera,
                self.engine['scene'],
                self.opt,
                self.bg_color,
                override_color=self.cluster_cache["point_colors_torch"],
                filtered_mask=hidden_mask,
            )["render"].permute(1, 2, 0)
        if cluster_ready and self.cluster_pick_request is not None:
            self.cluster_pick_feature_render = render_contrastive_feature(
                view_camera,
                self.engine['feature'],
                self.opt,
                self.bg_feature,
                filtered_mask=hidden_mask,
                point_feature_override=self.cluster_cache["point_codes_torch"],
            )["render"].permute(1, 2, 0)

        # --- RGB image --- #
        img = scene_outputs["render"].permute(1, 2, 0)

        rgb_score = img.clone()
        depth_score = rgb_score.cpu().numpy().reshape(-1)

        # --- semantic image --- #
        sems = feature_outputs["render"].permute(1, 2, 0)
        H, W, C = sems.shape
        sems /= (torch.norm(sems, dim=-1, keepdim=True) + 1e-6)
        sem_transed = sems @ self.proj_mat
        sem_transed_rgb = torch.clip(sem_transed*0.5+0.5, 0, 1)

        self.gates = self.engine['scale_gate'](torch.tensor([scale]).cuda())
        scale_gated_feat = sems * self.gates.unsqueeze(0).unsqueeze(0)
        scale_gated_feat = torch.nn.functional.normalize(scale_gated_feat, dim = -1, p = 2)
        
        if self.clear_edit:
            self.new_click_xy = []
            self.clear_edit = False
            self.prompt_num = 0
            try:
                self.engine['scene'].clear_segment()
                self.engine['feature'].clear_segment()
            except:
                pass

        if self.roll_back:
            self.new_click_xy = []
            self.roll_back = False
            self.prompt_num = 0
            # try:
            self.engine['scene'].roll_back()
            self.engine['feature'].roll_back()
            # except:
                # pass
        
        if self.reload_flag:
            self.reload_flag = False
            print("loading model file...")
            self.engine['scene'].load_ply(self.opt.SCENE_PCD_PATH)
            self.engine['feature'].load_ply(self.opt.FEATURE_PCD_PATH)
            self.engine['scale_gate'].load_state_dict(_load_scale_gate_state_dict(self.opt.SCALE_GATE_PATH))
            self._validate_scene_feature_alignment(context="reload")
            self.do_pca()   # calculate self.proj_mat
            self.load_model = True
            self._invalidate_cluster_cache(reason="reloaded data")

        score_map = None
        if len(self.new_click_xy) > 0:

            featmap = scale_gated_feat.reshape(H, W, -1)
            
            if self.new_click:
                xy = self.new_click_xy
                new_feat = featmap[int(xy[1])%H, int(xy[0])%W, :].reshape(featmap.shape[-1], -1)
                if (self.prompt_num == 0) or (self.clickmode_multi_button == False):
                    self.chosen_feature = new_feat
                else:
                    self.chosen_feature = torch.cat([self.chosen_feature, new_feat], dim=-1)    # extend to get more prompt features
                self.prompt_num += 1
                self.new_click = False
            
            score_map = featmap @ self.chosen_feature
            # print(score_map.shape, score_map.min(), score_map.max(), "score_map_shape")

            score_map = (score_map + 1.0) / 2
            score_binary = score_map > dpg.get_value('_ScoreThres')
            
            score_map[~score_binary] = 0.0
            score_map = torch.max(score_map, dim=-1).values
            score_norm = (score_map - dpg.get_value('_ScoreThres')) / (1 - dpg.get_value('_ScoreThres'))

            if self.preview:
                rgb_score = img * torch.max(score_binary, dim=-1, keepdim=True).values    # option: binary
            else:
                rgb_score = img
            depth_score = 1 - torch.clip(score_norm, 0, 1)
            depth_score = depth2img(depth_score.cpu().numpy()).astype(np.float32)/255.0

            if self.segment3d_flag:
                """ gaussian point cloud core params
                self.engine._xyz            # (N, 3)
                self.engine._features_dc    # (N, 1, 3)
                self.engine._features_rest  # (N, 15, 3)
                self.engine._opacity        # (N, 1)
                self.engine._scaling        # (N, 3)
                self.engine._rotation       # (N, 4)
                self.engine._objects_dc     # (N, 1, 16)
                """
                self.segment3d_flag = False
                feat_pts = self.engine['feature'].get_point_features.squeeze()
                scale_gated_feat_pts = feat_pts * self.gates.unsqueeze(0)
                scale_gated_feat_pts = torch.nn.functional.normalize(scale_gated_feat_pts, dim = -1, p = 2)

                score_pts = scale_gated_feat_pts @ self.chosen_feature
                score_pts = (score_pts + 1.0) / 2
                self.score_pts_binary = (score_pts > dpg.get_value('_ScoreThres')).sum(1) > 0

                # save_path = "./debug_robot_{:0>3d}.ply".format(self.object_seg_id)
                # try:
                #     self.engine['scene'].roll_back()
                #     self.engine['feature'].roll_back()
                # except:
                #     pass
                self.engine['scene'].segment(self.score_pts_binary)
                self.engine['feature'].segment(self.score_pts_binary)

        if self.save_flag:
            print("Saving ...")
            self.save_flag = False
            try:
                os.makedirs("./segmentation_res", exist_ok=True)
                save_mask = self.engine['scene']._mask == self.engine['scene'].segment_times + 1
                torch.save(save_mask, f"./segmentation_res/{dpg.get_value('save_name')}.pt")
            except:
                with dpg.window(label="Tips"):
                    dpg.add_text('You should segment the 3D object before save it (click segment3d first).')

        if self.save_active_flag:
            self.save_active_flag = False
            self._save_active_gaussians()
        if self.save_full_scene_flag:
            self.save_full_scene_flag = False
            self._save_full_scene_gaussians()

        self._consume_cluster_pick_request(view_camera, scene_outputs)
        self._consume_marquee_pick_request(view_camera, scene_outputs)
        self.selection_debug_render = self._build_selection_debug_render(view_camera, score_thres)
        self._update_cluster_status_widget()

        self.render_buffer = None
        render_num = 0
        if self.render_mode_rgb:
            self.render_buffer = rgb_score.cpu().numpy().reshape(-1)
            render_num += 1

        if self.render_mode_pca:
            self.render_buffer = sem_transed_rgb.cpu().numpy().reshape(-1) if self.render_buffer is None else self.render_buffer + sem_transed_rgb.cpu().numpy().reshape(-1)
            render_num += 1
        if self.render_mode_cluster:
            if self.rendered_cluster is None:
                cluster_buffer = rgb_score.cpu().numpy().reshape(-1)
            else:
                cluster_buffer = self.rendered_cluster.cpu().numpy().reshape(-1)
            self.render_buffer = cluster_buffer if self.render_buffer is None else self.render_buffer + cluster_buffer
            render_num += 1

        if render_num == 0:
            self.render_buffer = np.zeros((self.height * self.width * 3,), dtype=np.float32)
        else:
            self.render_buffer /= render_num

        if self.selection_debug_render is not None:
            highlight_np = self.selection_debug_render.detach().cpu().numpy()
            render_img = self.render_buffer.reshape(self.height, self.width, 3)
            highlight_mask = np.max(highlight_np, axis=-1, keepdims=True) > 1e-6
            highlighted = np.clip(render_img * 0.15 + highlight_np * 1.05, 0, 1)
            render_img = np.where(highlight_mask, highlighted, render_img)
            self.render_buffer = render_img.reshape(-1)

        dpg.set_value("_texture", self.render_buffer)
        self._refresh_overlay_drawlist(view_camera)


if __name__ == "__main__":
    def _str2bool(value):
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ArgumentTypeError(f"Invalid boolean value: {value}")

    parser = ArgumentParser(description="GUI option")

    parser.add_argument('-m', '--model_path', type=str, default="./output/figurines")
    parser.add_argument('--feature_model_path', type=str, default="")
    parser.add_argument('--exp_name', type=str, default="")
    parser.add_argument('-f', '--feature_iteration', type=int, default=-1)
    parser.add_argument('-s', '--scene_iteration', type=int, default=-1)
    parser.add_argument('--cluster_method', type=str, default="HDBSCANRefined", choices=["HDBSCAN", "HDBSCANRefined", "NormalizedCut"])
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
    parser.add_argument('--hide_small_residue', type=_str2bool, default=True)
    parser.add_argument('--residue_max_size', type=int, default=1000)
    parser.add_argument('--residue_scope', type=str, default="Both", choices=["Global", "Active", "Both"])
    parser.add_argument('--exclude_residue_on_active_export', type=_str2bool, default=True)

    args = parser.parse_args()

    opt = CONFIG()

    opt.MODEL_PATH = args.model_path
    if args.feature_model_path:
        opt.FEATURE_MODEL_PATH = args.feature_model_path
    elif args.exp_name:
        exp_name = os.path.normpath(args.exp_name)
        if os.path.isabs(exp_name) or exp_name.startswith(".."):
            raise ValueError(f"exp_name must be a relative subdirectory under model_path, got: {args.exp_name}")
        opt.FEATURE_MODEL_PATH = os.path.join(opt.MODEL_PATH, exp_name)
    else:
        opt.FEATURE_MODEL_PATH = ""
    opt.FEATURE_GAUSSIAN_ITERATION = args.feature_iteration
    opt.SCENE_GAUSSIAN_ITERATION = args.scene_iteration
    opt.CLUSTER_METHOD = args.cluster_method
    opt.CLUSTER_SAMPLE_SIZE = args.cluster_sample_size
    opt.CLUSTER_GRAPH_K = args.cluster_graph_k
    opt.CLUSTER_MAX_CLUSTERS = args.cluster_max_clusters
    opt.CLUSTER_MIN_CLUSTER_SIZE = args.cluster_min_cluster_size
    opt.CLUSTER_CUT_THRESHOLD = args.cluster_cut_threshold
    opt.CLUSTER_SPATIAL_WEIGHT = args.cluster_spatial_weight
    opt.CLUSTER_SH0_COLOR_WEIGHT = args.cluster_sh0_color_weight
    opt.CLUSTER_SH0_COLOR_SIGMA = args.cluster_sh0_color_sigma
    opt.CLUSTER_MESH_WEIGHT = args.cluster_mesh_weight
    opt.CLUSTER_MESH_PATH = args.cluster_mesh_path
    opt.HIDE_SMALL_RESIDUE = args.hide_small_residue
    opt.RESIDUE_MAX_SIZE = args.residue_max_size
    opt.RESIDUE_SCOPE = args.residue_scope
    opt.EXCLUDE_RESIDUE_ON_ACTIVE_EXPORT = args.exclude_residue_on_active_export
    opt.CUBOID_PERCENTILE = 100.0

    _resolve_model_artifacts(opt)

    gs_model = GaussianModel(opt.sh_degree)
    feat_gs_model = FeatureGaussianModel(opt.FEATURE_DIM)
    scale_gate = torch.nn.Sequential(
        torch.nn.Linear(1, opt.FEATURE_DIM, bias=True),
        torch.nn.Sigmoid()
    ).cuda()
    gui = GaussianSplattingGUI(opt, gs_model, feat_gs_model, scale_gate)

    gui.render()
