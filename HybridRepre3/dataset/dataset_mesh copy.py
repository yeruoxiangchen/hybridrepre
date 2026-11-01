# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
import os
from render import util
from render import mesh
from render import render
from render import light

from .dataset import Dataset

import glob
###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

def _load_mask(fn):
    img = torch.tensor(util.load_image(fn), dtype=torch.float32)
    if len(img.shape) == 2:
        img = img[..., None].repeat(1, 1, 3)
    return img

def _load_img(fn):
    img = util.load_image_raw(fn)
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img


class DatasetMesh(Dataset):

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, base_dir, validate=False, examples=None):
        # Init 
        self.FLAGS              = FLAGS
        self.examples           = examples
        self.base_dir           = base_dir
        
        # Enumerate all image files and get resolution
        all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        self.resolution = _load_img(all_img[0]).shape[0:2]
        
        # Load camera poses 
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))  # [N, 17] 每行包含 ​3x4 位姿矩阵（前12个元素）​ 和 ​场景边界参数（后5个元素）​
        
        poses        = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])  # [3,5,N]
        poses        = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Taken from nerf, swizzles from LLFF to expected coordinate system  ​OpenGL风格
        poses        = np.moveaxis(poses, -1, 0).astype(np.float32)
        
        lcol         = np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(poses.shape[0], 0)
        self.imvs    = torch.tensor(np.concatenate((poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)
        # self.imvs = torch.tensor(
        #     np.concatenate(
        #         (poses[::4, :, 0:4],   # 对poses进行步长5采样
        #         lcol[::4]),           # 对lcol同步采样（假设lcol与poses帧数对齐）
        #         axis=1
        #     ), 
        #     dtype=torch.float32
        # )
        self.aspect  = self.resolution[1] / self.resolution[0] # width / height
        self.fovy    = util.focal_length_to_fovy(poses[:, 2, 4], poses[:, 0, 4])
        
        # Recenter scene so lookat position is origin
        center                = util.lines_focal(self.imvs[..., :3, 3], -self.imvs[..., :3, 2])
        self.imvs[..., :3, 3] = self.imvs[..., :3, 3] - center[None, ...]
            
        if self.FLAGS.local_rank == 0:
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))
            print("DatasetLLFF: %d images with shape [%d, %d]" % (len(all_img), self.resolution[0], self.resolution[1]))
            # print("DatasetLLFF: auto-centering at %s" % (center.cpu().numpy()))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))
        # self.ref_mesh = mesh.compute_tangents(ref_mesh)
        # self.ref_mesh = self.normalize_mesh()

        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

        # Pre-load from disc to avoid slow png parsing
        self.all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) 
                        if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        self.all_mask = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "masks", "*"))) 
                         if f.lower().endswith(('png', 'jpg', 'jpeg'))]
        assert len(self.all_img) == self.imvs.shape[0] and len(self.all_mask) == self.imvs.shape[0]
        if self.FLAGS.pre_load:
            self.preloaded_data = []
            for i in range(self.imvs.shape[0]):
                print(f"{i}/{self.imvs.shape[0]} is loading...", end="\r")
                self.preloaded_data += [self._parse_frame(i)]
                
    def _parse_frame(self, idx):

        # Load image+mask data
        img  = _load_img(self.all_img[idx])
        mask = _load_mask(self.all_mask[idx])
        img  = torch.cat((img, mask[..., 0:1]), dim=-1)

        # Setup transforms
        proj   = util.perspective(self.fovy[idx, ...], self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
        mv     = torch.linalg.inv(self.imvs[idx, ...])
        # mv[1, :] *= -1  # Y轴取反
        # mv[2, :] *= -1  # Z轴取反
        # R = mv[:3, :3].T  # [3, 3]
        # mv[:3, :3] = R
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...] # Add batch dimension
    
    def _get_proj(self):
        return util.perspective(self.fovy[0], self.aspect, self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])
    
    def normalize_mesh(self):
        # 计算模型包围盒
        vmin = torch.min(self.ref_mesh.v_pos, dim=0).values
        vmax = torch.max(self.ref_mesh.v_pos, dim=0).values
        center = (vmin + vmax) / 2.0  # 包围盒中心
        scale = 2.0 / (torch.max(vmax - vmin)).item()  # 缩放至 [-1,1]^3

        # 平移顶点使中心位于原点，并缩放
        self.ref_mesh.v_pos = (self.ref_mesh.v_pos - center) * scale
        print(f"Normalized vertices range: x [{self.ref_mesh.v_pos[:,0].min()}, {self.ref_mesh.v_pos[:,0].max()}]")
        print(f"Normalized vertices range: y [{self.ref_mesh.v_pos[:,1].min()}, {self.ref_mesh.v_pos[:,1].max()}]")
        print(f"Normalized vertices range: z [{self.ref_mesh.v_pos[:,2].min()}, {self.ref_mesh.v_pos[:,2].max()}]")
        return self.ref_mesh
    
    def __len__(self):
        return self.imvs.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        
        if self.FLAGS.pre_load:
            img, mv, mvp, campos = self.preloaded_data[itr % self.imvs.shape[0]]
        else:
            img, mv, mvp, campos = self._parse_frame(itr % self.imvs.shape[0])
        # if torch.all(img == 0):
        #     print("是")
        # else:
        #     print("否")
        # import matplotlib.pyplot as plt


        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.resolution,
            'spp' : self.FLAGS.spp,
            'img' : img
        }
