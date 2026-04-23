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

import torch
import numpy as np
import os
import sys
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH

_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

try:
    from simple_knn._C import distCUDA2
except ModuleNotFoundError:
    _submodule_path = os.path.join(_ROOT_DIR, "submodules", "simple-knn")
    if _submodule_path not in sys.path:
        sys.path.insert(0, _submodule_path)
    from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.gaussian_backend_utils import (
    detect_gaussian_backend_from_scale_dims,
    ensure_rasterizer_scale_dims,
    export_scaling_for_ply,
    extract_scales_from_ply_element,
    resolve_gaussian_backend,
)

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self.source_scale_dims = 3
        self.gaussian_backend = "3dgs"
        self.loaded_from_path = None
        self._xyz = torch.empty(0)
        self._mask = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

        self.old_xyz = []
        self.old_mask = []

        self.old_features_dc = []
        self.old_features_rest = []
        self.old_opacity = []
        self.old_scaling = []
        self.old_rotation = []

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._mask,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            self.source_scale_dims,
            self.gaussian_backend,
        )
    
    def restore(self, model_args, training_args):
        if len(model_args) == 13:
            (self.active_sh_degree,
             self._xyz,
             self._mask,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale) = model_args
            self.source_scale_dims = min(int(self._scaling.shape[1]), 3)
            self.gaussian_backend = detect_gaussian_backend_from_scale_dims(self.source_scale_dims)
        else:
            (self.active_sh_degree,
             self._xyz,
             self._mask,
             self._features_dc,
             self._features_rest,
             self._scaling,
             self._rotation,
             self._opacity,
             self.max_radii2D,
             xyz_gradient_accum,
             denom,
             opt_dict,
             self.spatial_lr_scale,
             self.source_scale_dims,
             self.gaussian_backend) = model_args
        self.loaded_from_path = None
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_mask(self):
        return self._mask
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        self.source_scale_dims = 3
        self.gaussian_backend = "3dgs"
        self.loaded_from_path = None
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        mask = torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        # self._mask = nn.Parameter(mask.requires_grad_(True))
        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},

            # {'params': [self._mask], 'lr': training_args.mask_lr, "name": "mask"},

            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self, scaling_dims=None):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        scaling_dims = self.source_scale_dims if scaling_dims is None else scaling_dims
        for i in range(scaling_dims):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        # mask = self._mask.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = export_scaling_for_ply(self._scaling.detach().cpu().numpy(), self.source_scale_dims)
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes(scale.shape[1])]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, mask, normals, f_dc, f_rest, opacities, scale, rotation), axis=1) if has_mask else np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    # def save_ply(self, path):
    #     mkdir_p(os.path.dirname(path))

    #     xyz = self._xyz.detach().cpu().numpy()
    #     normals = np.zeros_like(xyz)
    #     f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    #     opacities = self._opacity.detach().cpu().numpy()
    #     scale = self._scaling.detach().cpu().numpy()
    #     rotation = self._rotation.detach().cpu().numpy()

    #     dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
    #     # edit
    #     add_color = True
    #     if add_color:
    #         dtype_full[3], dtype_full[4], dtype_full[5] = ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')
    #         rgbs = SH2RGB(f_dc)
    #         normals = (np.clip(rgbs, 0.0, 1.0) * 255).astype(np.uint8)
            
    #     elements = np.empty(xyz.shape[0], dtype=dtype_full)
    #     attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    #     elements[:] = list(map(tuple, attributes))
    #     el = PlyElement.describe(elements, 'vertex')
    #     PlyData([el]).write(path)

    def save_mask(self, path):
        mkdir_p(os.path.dirname(path))
        mask = self._mask.detach().cpu().numpy()        
        np.save(path, mask)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)
        ply_element = plydata.elements[0]

        xyz = np.stack((np.asarray(ply_element["x"]),
                        np.asarray(ply_element["y"]),
                        np.asarray(ply_element["z"])),  axis=1)
        
        # mask = np.asarray(plydata.elements[0]["mask"])[..., np.newaxis]

        opacities = np.asarray(ply_element["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(ply_element["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(ply_element["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(ply_element["f_dc_2"])

        extra_f_names = [p.name for p in ply_element.properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        expected_extra_dim = 3 * ((self.max_sh_degree + 1) ** 2 - 1)
        features_extra = np.zeros((xyz.shape[0], expected_extra_dim), dtype=np.float32)
        if len(extra_f_names) > expected_extra_dim:
            raise ValueError(
                f"Checkpoint at {path} has {len(extra_f_names)} f_rest_* properties, "
                f"but GaussianModel(sh_degree={self.max_sh_degree}) expects at most {expected_extra_dim}."
            )
        if len(extra_f_names) < expected_extra_dim:
            if len(extra_f_names) == 0:
                print(
                    f"PLY {path} has no f_rest_* properties. "
                    f"Loading as DC-only colors and zero-filling higher-order SH coeffs for sh_degree={self.max_sh_degree}."
                )
            else:
                print(
                    f"PLY {path} has only {len(extra_f_names)} / {expected_extra_dim} f_rest_* properties. "
                    f"Zero-filling the remaining higher-order SH coeffs."
                )
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(ply_element[attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scales, self.source_scale_dims = extract_scales_from_ply_element(ply_element)
        scales = ensure_rasterizer_scale_dims(scales)

        rot_names = [p.name for p in ply_element.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(ply_element[attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))

        # self._mask = nn.Parameter(torch.tensor(mask, dtype=torch.float, device="cuda").requires_grad_(True))

        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
        self.gaussian_backend = detect_gaussian_backend_from_scale_dims(self.source_scale_dims)
        self.loaded_from_path = path

        self.segment_times = 0
        self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")

    def resolve_gaussian_backend(self, requested_backend="auto", checkpoint_path=None):
        checkpoint_path = checkpoint_path or self.loaded_from_path or "<in-memory GaussianModel>"
        self.gaussian_backend = resolve_gaussian_backend(requested_backend, self.source_scale_dims, checkpoint_path)
        return self.gaussian_backend

    def _require_inference_editable(self):
        if self.optimizer is not None:
            raise RuntimeError("Inference-time Gaussian editing requires optimizer=None.")

    def _reset_inference_aux_buffers(self, count, device, mask=None):
        self.max_radii2D = torch.zeros((count,), device=device)
        self.xyz_gradient_accum = torch.zeros((count, 1), device=device)
        self.denom = torch.zeros((count, 1), device=device)
        if mask is None:
            self._mask = torch.ones((count,), dtype=torch.float, device=device)
        else:
            self._mask = mask.to(device=device, dtype=torch.float).reshape(-1).contiguous()

    def set_inference_tensors(self, xyz, features_dc, features_rest, opacity, scaling, rotation, mask=None):
        self._require_inference_editable()
        device = self._xyz.device if torch.is_tensor(self._xyz) and self._xyz.numel() > 0 else xyz.device
        self._xyz = nn.Parameter(xyz.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features_rest.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(opacity.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scaling.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._rotation = nn.Parameter(rotation.to(device=device, dtype=torch.float).contiguous().requires_grad_(True))
        self._reset_inference_aux_buffers(int(self._xyz.shape[0]), device=device, mask=mask)

    def append_inference_tensors(self, xyz, features_dc, features_rest, opacity, scaling, rotation):
        self._require_inference_editable()
        if xyz is None or int(xyz.shape[0]) == 0:
            return
        self.set_inference_tensors(
            xyz=torch.cat((self._xyz.detach(), xyz.detach().to(self._xyz.device)), dim=0),
            features_dc=torch.cat((self._features_dc.detach(), features_dc.detach().to(self._features_dc.device)), dim=0),
            features_rest=torch.cat((self._features_rest.detach(), features_rest.detach().to(self._features_rest.device)), dim=0),
            opacity=torch.cat((self._opacity.detach(), opacity.detach().to(self._opacity.device)), dim=0),
            scaling=torch.cat((self._scaling.detach(), scaling.detach().to(self._scaling.device)), dim=0),
            rotation=torch.cat((self._rotation.detach(), rotation.detach().to(self._rotation.device)), dim=0),
            mask=None,
        )

    def prune_inference_tensors(self, prune_mask):
        self._require_inference_editable()
        if prune_mask is None:
            return
        prune_mask = prune_mask.to(device=self._xyz.device, dtype=torch.bool).reshape(-1)
        if prune_mask.shape[0] != self._xyz.shape[0]:
            raise ValueError(
                f"Prune mask length {prune_mask.shape[0]} does not match Gaussian count {self._xyz.shape[0]}."
            )
        keep_mask = ~prune_mask
        self.set_inference_tensors(
            xyz=self._xyz.detach()[keep_mask],
            features_dc=self._features_dc.detach()[keep_mask],
            features_rest=self._features_rest.detach()[keep_mask],
            opacity=self._opacity.detach()[keep_mask],
            scaling=self._scaling.detach()[keep_mask],
            rotation=self._rotation.detach()[keep_mask],
            mask=None,
        )

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    @torch.no_grad()
    def segment(self, mask=None):
        assert mask is not None
            # mask = (self._mask > 0)
        mask = mask.squeeze()
        # assert mask.shape[0] == self._xyz.shape[0]
        if torch.count_nonzero(mask) == 0:
            mask = ~mask
            print("Seems like the mask is empty, segmenting the whole point cloud. Please run seg.py first.")

        self.old_xyz.append(self._xyz)
        self.old_mask.append(self._mask)

        self.old_features_dc.append(self._features_dc)
        self.old_features_rest.append(self._features_rest)
        self.old_opacity.append(self._opacity)
        self.old_scaling.append(self._scaling)
        self.old_rotation.append(self._rotation)
        
        if self.optimizer is None:
            self._xyz = self._xyz[mask]
            # self._mask = self._mask[mask]

            self._features_dc = self._features_dc[mask]
            self._features_rest = self._features_rest[mask]
            self._opacity = self._opacity[mask]
            self._scaling = self._scaling[mask]
            self._rotation = self._rotation[mask]

        else:
            optimizable_tensors = self._prune_optimizer(mask)

            self._xyz = optimizable_tensors["xyz"]

            # self._mask = optimizable_tensors["mask"]

            self._features_dc = optimizable_tensors["f_dc"]
            self._features_rest = optimizable_tensors["f_rest"]
            self._opacity = optimizable_tensors["opacity"]
            self._scaling = optimizable_tensors["scaling"]
            self._rotation = optimizable_tensors["rotation"]

            self.xyz_gradient_accum = self.xyz_gradient_accum[mask]

            self.denom = self.denom[mask]

        # print(self.segment_times, torch.unique(self._mask))
        self.segment_times += 1
        tmp = self._mask[self._mask == self.segment_times]
        tmp[mask] += 1
        self._mask[self._mask == self.segment_times] = tmp

        # print(self._mask[self._mask == self.segment_times][mask].shape)
        # print(self.segment_times, torch.unique(self._mask), torch.unique(mask))
        
    def roll_back(self):
        try:
            self._xyz = self.old_xyz.pop()
            # self._mask = self.old_mask.pop()

            self._features_dc = self.old_features_dc.pop()
            self._features_rest = self.old_features_rest.pop()
            self._opacity = self.old_opacity.pop()
            self._scaling = self.old_scaling.pop()
            self._rotation = self.old_rotation.pop()

            
            self._mask[self._mask == self.segment_times+1] -= 1
            self.segment_times -= 1
        except:
            pass
    
    @torch.no_grad()
    def clear_segment(self):
        try:
            self._xyz = self.old_xyz[0]
            # self._mask = self.old_mask[0]

            self._features_dc = self.old_features_dc[0]
            self._features_rest = self.old_features_rest[0]
            self._opacity = self.old_opacity[0]
            self._scaling = self.old_scaling[0]
            self._rotation = self.old_rotation[0]

            self.old_xyz = []
            self.old_mask = []

            self.old_features_dc = []
            self.old_features_rest = []
            self.old_opacity = []
            self.old_scaling = []
            self.old_rotation = []

            self.segment_times = 0
            self._mask = torch.ones((self._xyz.shape[0],), dtype=torch.float, device="cuda")
        except:
            # print("Roll back failed. Please run gaussians.segment() first.")
            pass

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        # "mask": new_mask,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        # self._mask = optimizable_tensors["mask"]

        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)

        # new_mask = self._mask[selected_pts_mask].repeat(N,1)

        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]

        # new_mask = self._mask[selected_pts_mask]

        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
