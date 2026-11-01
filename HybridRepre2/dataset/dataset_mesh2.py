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
import torch.nn as nn
import os

from render import util
from render import mesh
from render import render
from render import light

from .dataset2 import Dataset
from nerf.utils import get_rays
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

    def __init__(self, FLAGS, base_dir, center = None, examples=None):
        # Init 
        self.FLAGS              = FLAGS
        self.examples           = examples
        self.base_dir           = base_dir
        self.center             = center
        
        # Enumerate all image files and get resolution
        all_img = [f for f in sorted(glob.glob(os.path.join(self.base_dir, "images", "*"))) if f.lower().endswith('png') or f.lower().endswith('jpg') or f.lower().endswith('jpeg')]
        self.resolution = _load_img(all_img[0]).shape[0:2]
        
        # Load camera poses
        poses_bounds = np.load(os.path.join(self.base_dir, 'poses_bounds.npy'))
        
        poses        = poses_bounds[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])  # [3,5,N]
        if self.center is not None:
            poses[0, 3, :] -= self.center[0]  # 假设平移部分在第3行
            poses[1, 3, :] -= self.center[1]
            poses[2, 3, :] -= self.center[2]
        poses        = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1) # Taken from nerf, swizzles from LLFF to expected coordinate system  ​OpenGL风格
        poses        = np.moveaxis(poses, -1, 0).astype(np.float32)
        
        lcol         = np.array([0,0,0,1], dtype=np.float32)[None, None, :].repeat(poses.shape[0], 0)
        self.imvs    = torch.tensor(np.concatenate((poses[:, :, 0:4], lcol), axis=1), dtype=torch.float32)

        self.aspect  = self.resolution[1] / self.resolution[0] # width / height
        self.fovy    = util.focal_length_to_fovy(poses[:, 2, 4], poses[:, 0, 4])
        
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
        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        # --- 新增代码：调用 get_rays 生成 rays_o 和 rays_d ---
        # 构造内参 [fl_x, fl_y, cx, cy]
        H, W = self.resolution
        fovy_rad = np.deg2rad(self.fovy[idx].item())
        fl_y = 1 / (2 * np.tan(fovy_rad / 2)) * H  # 焦距计算
        fl_x = fl_y * self.aspect  # 水平焦距
        cx = W / 2.0  
        cy = H / 2.0
        intrinsics = torch.tensor([fl_x, fl_y, cx, cy], dtype=torch.float32, device=mv.device)

        # 构造位姿（相机到世界变换矩阵）
        poses = torch.linalg.inv(mv).unsqueeze(0)  # [1, 4, 4]

        # 调用 get_rays
        rays = get_rays(
            poses=poses, 
            intrinsics=intrinsics.unsqueeze(0),  # [1, 4]
            H=H, 
            W=W, 
            N=-1,  # 生成全部光线
            patch_size=None  # 无局部采样
        )

        return img[None, ...], mv[None, ...], mvp[None, ...], campos[None, ...], rays['rays_o'], rays['rays_d'] # Add batch dimension
        
    def normalize_mesh(self):
        # 计算模型包围盒
        vmin = torch.min(self.ref_mesh.v_pos, dim=0).values
        vmax = torch.max(self.ref_mesh.v_pos, dim=0).values
        center = (vmin + vmax) / 2.0  # 包围盒中心
        scale = 2.0 / (torch.max(vmax - vmin)).item()  # 缩放至 [-1,1]^3

        # 平移顶点使中心位于原点，并缩放
        self.ref_mesh.v_pos = (self.ref_mesh.v_pos - center) * scale
        return self.ref_mesh
    
    def __len__(self):
        return self.imvs.shape[0] if self.examples is None else self.examples

    def __getitem__(self, itr):
        
        if self.FLAGS.pre_load:
            img, mv, mvp, campos, rays_o, rays_d = self.preloaded_data[itr % self.imvs.shape[0]]
        else:
            img, mv, mvp, campos, rays_o, rays_d = self._parse_frame(itr % self.imvs.shape[0])
        # if torch.all(img == 0):
        #     print("是")
        # else:
        #     print("否")
        # import matplotlib.pyplot as plt

        # # 假设 img 是一个 torch.Tensor，形状为 (1, H, W, 4)
        # # 将 img 从 GPU 移到 CPU，并转换为 numpy 数组，同时提取 batch 中的第一个图像
        # img_cpu = img.detach().cpu().numpy()[0]

        # # 只取 RGB 通道（前三个通道），如果你不需要透明度信息
        # img_rgb = img_cpu[..., :3]

        # # 显示图像
        # plt.figure(figsize=(8, 8))
        # plt.imshow(img_rgb)
        # plt.axis('off')  # 不显示坐标轴
        # plt.show()


        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : self.resolution,
            'spp' : self.FLAGS.spp,
            'img' : img,
            'rays_o': rays_o,
            'rays_d': rays_d
        }
