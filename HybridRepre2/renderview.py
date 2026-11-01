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
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

import nvdiffrast.torch as dr
from render import mesh
from dataset.dataset_mesh import DatasetMesh
from geometry.dmtet import DMTetGeometry
import numpy as np
from render import texture
from render import mlptexture
from render import material
from render import light
from render import util
import json

from torch import nn
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, uid, image_width, image_height,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.image_width = image_width
        self.image_height = image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['img'] = target['img'].cuda()
    target['background'] = background
    # 混合背景与前景图像
    target['img'] = torch.cat((torch.lerp(background, target['img'][..., 0:3], target['img'][..., 3:4]), target['img'][..., 3:4]), dim=-1)

    return target


def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)
        
        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        return result_image, result_dict
import math
def euler_to_rotation_matrix(theta_x, theta_y, theta_z):
    """将欧拉角转换为旋转矩阵（XYZ顺序）"""
    Rx = np.array([[1, 0, 0],
                   [0, math.cos(theta_x), -math.sin(theta_x)],
                   [0, math.sin(theta_x), math.cos(theta_x)]])
    
    Ry = np.array([[math.cos(theta_y), 0, math.sin(theta_y)],
                   [0, 1, 0],
                   [-math.sin(theta_y), 0, math.cos(theta_y)]])
    
    Rz = np.array([[math.cos(theta_z), -math.sin(theta_z), 0],
                   [math.sin(theta_z), math.cos(theta_z), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx  # XYZ顺序：先绕X轴，再Y轴，最后Z轴

def update_camera_pose(initial_camera, roll=0, pitch=0, yaw=0, tx=0, ty=0, tz=0):
    """
    更新相机位姿（以初始相机为基准进行旋转）
    参数：
        roll: X轴旋转角度（弧度）
        pitch: Y轴旋转角度（弧度） 
        yaw: Z轴旋转角度（弧度）
    返回：
        新的Camera对象
    """
    # 将角度转换为旋转矩阵
    delta_R = euler_to_rotation_matrix(roll, pitch, yaw)
    
    # 应用新的旋转（相对于初始相机坐标系）
    new_R = initial_camera.R @ delta_R
    
    delta_t_local = np.array([tx, ty, tz])
    delta_t_world = initial_camera.R @ delta_t_local  # 将局部平移转为世界坐标系
    
    # 应用新的平移
    new_T = initial_camera.T + delta_t_world
    
    # 创建新相机对象（需根据实际Camera类调整参数）
    new_camera = Camera(
        colmap_id=initial_camera.colmap_id,
        R=new_R,
        T=new_T,
        FoVx=initial_camera.FoVx,
        FoVy=initial_camera.FoVy,
        uid=initial_camera.uid ,
        image_width=initial_camera.image_width,
        image_height=initial_camera.image_height
    )
    
    return new_camera
# def generate_random_cameras(num_views, radius, fovx, fovy, img_size):

#     cameras = []
#     for _ in range(num_views):
        
#         # theta = np.random.uniform(0, 2*np.pi)  # 方位角
#         # phi = np.random.uniform(-np.pi/3, np.pi/3)      # 仰角（全球）
        
#         # x = radius * np.sin(phi) * np.cos(theta)
#         # y = radius * np.sin(phi) * np.sin(theta)
#         # z = radius * np.cos(phi)
#         theta = np.random.uniform(0, np.pi)  # 极角
#         phi = np.random.uniform(0, 2 * np.pi)  # 方位角
#         x = radius * np.sin(theta) * np.cos(phi)
#         y = radius * np.sin(theta) * np.sin(phi)
#         z = radius * np.cos(theta)
#         # x = 0
#         # y = 0
#         # z = -1
#         camera_pos = np.array([x, y, z])
#         print(camera_pos)

#         # 2. 计算朝向原点的旋转矩阵（世界→相机坐标系）
#         # 定义相机坐标系的Z轴指向原点，Y轴尽量朝上
#         forward = -camera_pos / np.linalg.norm(camera_pos)  # 相机Z轴方向（指向原点）
#         up = np.array([0, 1, 0]) 
#         # up = np.array([0, -1, 0]) 

#         # 构造正交基
#         right = np.cross(forward, up)
#         right /= np.linalg.norm(right)
#         up = np.cross(right, forward)  # 重新正交化
#         up /= np.linalg.norm(up)

#         # 旋转矩阵：世界坐标系→相机坐标系
#         R_world2cam = np.column_stack([right, up, -forward])  # 世界→相机的旋转矩阵
#         # R_world2cam = np.column_stack([right, up, forward])  # 世界→相机的旋转矩阵
        
#         # 计算正确的 tvec
#         # tvec = -R_world2cam @ camera_pos  
#         # T = np.array(tvec)
#         T = camera_pos
#         # R_world2cam = R_world2cam @ np.diag([1, -1, -1])
#         # T = T * np.array([1, -1, -1])

#         cam = Camera(
#             colmap_id=-1,
#             R=R_world2cam.T,
#             T=T,
#             FoVx=fovx,
#             FoVy=fovy,
#             image_width=img_size[0],
#             image_height=img_size[1],
#             uid=len(cameras)
#         )
#         cameras.append(cam)
#     return cameras

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    blender_path = os.path.join(model_path, name, "ours_{}".format(iteration), "blender")
    mesh_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mesh")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(blender_path, exist_ok=True)
    makedirs(mesh_image_path, exist_ok=True)
     # mesh
    glctx = dr.RasterizeGLContext() # 管理 ​可微分光栅化（differentiable rasterization）的GPU上下文
    RADIUS = 3.0
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_dir = args.ref_mesh
    mesh_dir = os.path.join(base_dir, "mesh")
    mesh_path = os.path.join(base_dir, "mesh/mesh.obj")
    ref_mesh         = mesh.load_mesh(mesh_path, args.mtl_override)
    geometry = DMTetGeometry(args.dmtet_grid, args.mesh_scale, args, ref_mesh) # 加载预定义初始网格、初始化sdf和顶点形变参数
    
    mtl_path = os.path.join(mesh_dir, "mesh.mtl")
    mat = material.load_mtl(mtl_path,clear_ks=True)[0]
    
    lgt = light.load_env(args.envmap, scale=args.env_scale)
    dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, args, base_dir, validate=False, examples=(args.iter+1)*args.batch)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate)

    def generate_modified_views(initial_camera):
        """生成多轴位姿变化视图"""
        views = []
        
        # 参数配置
        rotation_params = {
            'roll': np.arange(0, 2.01, 0.1),    # X轴旋转 0-2弧度
            'pitch': np.arange(0, 2.01, 0.1),   # Y轴旋转 
            'yaw': np.arange(0, 2.01, 0.1)      # Z轴旋转
        }
        
        translation_params = {
            'tx': np.arange(0, 0.101, 0.01),    # X轴平移 0-0.1
            'ty': np.arange(0, 0.101, 0.01)     # Y轴平移
        }

        total = sum(len(v) for v in rotation_params.values()) + sum(len(v) for v in translation_params.values())
        current = 0
        
        # 生成旋转视角 (各轴独立变化)
        for axis in rotation_params:
            for val in rotation_params[axis]:
                params = {axis: val, 'tx':0, 'ty':0, 'tz':0}
                new_cam = update_camera_pose(initial_camera, **params)
                new_cam.uid = f"{initial_camera.uid}_{axis}_{val:.1f}"
                views.append(new_cam)
                
                current += 1
                print(f"Generating views: {current}/{total} [rot {axis}]", end='\r')

        # 生成平移视角 (XY轴独立变化)
        for axis in translation_params:
            for val in translation_params[axis]:
                params = {'roll':0, 'pitch':0, 'yaw':0, axis: val, 'tz':0}
                new_cam = update_camera_pose(initial_camera, **params)
                new_cam.uid = f"{initial_camera.uid}_{axis}_{val:.2f}"
                views.append(new_cam)
                
                current += 1
                print(f"Generating views: {current}/{total} [trans {axis}]", end='\r')

        print(f"\nGeneration completed. Total views: {len(views)}")
        return views
    views = generate_modified_views(views[0])
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # rendering = render(view, gaussians, pipeline, background)["render"]
        render_pkg = render(view, gaussians, pipeline, background)
        image = render_pkg["render"]

        if name == "train":
            index = int(view.image_name) 
            sample = dataset_train[index]  # 调用 __getitem__(5)
        else:
            mv = np.eye(4)
            mv[:3, :3] = view.R.T # 旋转部分
            mv[:3, 3] = view.T   # 平移部分 (相机位置)
            mv[1, :] *= -1  # Y轴取反
            mv[2, :] *= -1  # Z轴取反
            mv = torch.tensor(mv, dtype=torch.float32)
            proj = dataset_train._get_proj()
            mvp    = proj @ mv
            campos = torch.linalg.inv(mv)[:3, 3]
    
            resolution = (view.image_height, view.image_width)
            spp = args.spp
            img = dataset_train[0]['img']
            sample = {
                'mv' : mv.unsqueeze(0),
                'mvp' : mvp.unsqueeze(0),
                'campos' : campos.unsqueeze(0),
                'resolution' : resolution,
                'spp' : spp, 
                'img' : img
            }

        # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
        #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
        batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）
        # 3. 调用 prepare_batch 处理（与训练循环一致）
        target = prepare_batch(batch, args.background)
        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, args)
        
        # alpha = gaussians.get_alpha
        
        mesh_image = result_dict['opt']
        mesh_image = mesh_image.detach().permute(2, 0, 1)

        # blender_image = alpha * rendering + (1 - alpha) * mesh_image
        blender_image = render_pkg["rendered_alpha"] * image + (1 - render_pkg["rendered_alpha"]) * mesh_image
        if name == "train":
            gt = view.original_image[0:3, :, :]
            gt = gt * view.mask + background[:, None, None] * (1 - view.mask).squeeze()
            # blender_image = (blender_image - blender_image.min()) / (blender_image.max() - blender_image.min())
            # rendering = (rendering - rendering.min()) / (rendering.max() - rendering.min())

            torchvision.utils.save_image(mesh_image, os.path.join(mesh_image_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(blender_image, os.path.join(blender_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        else:
            torchvision.utils.save_image(mesh_image, os.path.join(mesh_image_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(blender_image, os.path.join(blender_path, '{0:05d}'.format(idx) + ".png"))
            torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))



def render_set2(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    blender_path = os.path.join(model_path, name, "ours_{}".format(iteration), "blender")
    mesh_image_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mesh")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(blender_path, exist_ok=True)
    makedirs(mesh_image_path, exist_ok=True)
     # mesh
    glctx = dr.RasterizeGLContext() # 管理 ​可微分光栅化（differentiable rasterization）的GPU上下文
    RADIUS = 3.0
    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_dir = args.ref_mesh
    mesh_dir = os.path.join(base_dir, "mesh")
    mesh_path = os.path.join(base_dir, "mesh/mesh.obj")
    ref_mesh         = mesh.load_mesh(mesh_path, args.mtl_override)
    geometry = DMTetGeometry(args.dmtet_grid, args.mesh_scale, args, ref_mesh) # 加载预定义初始网格、初始化sdf和顶点形变参数
    
    mtl_path = os.path.join(mesh_dir, "mesh.mtl")
    mat = material.load_mtl(mtl_path,clear_ks=True)[0]
    
    lgt = light.load_env(args.envmap, scale=args.env_scale)
    dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, args, base_dir, validate=False, examples=(args.iter+1)*args.batch)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate)
    
    initial_cam = views[0]
    
    # 初始化控制参数
    roll, pitch, yaw = 0.0, 0.0, 0.0  # 弧度
    tx, ty, tz = 0.0, 0.0, 0.0       # 平移量
    step_angle = 0.1                 # 单次按键旋转角度
    step_trans = 0.05                # 单次按键平移量
    import cv2
    # 创建OpenCV窗口
    cv2.namedWindow("Interactive View", cv2.WINDOW_NORMAL)
    
    while True:
        view = update_camera_pose(
            initial_cam,
            roll=roll,
            pitch=pitch,
            yaw=yaw,
            tx=tx,
            ty=ty,
            tz=tz
        )
        # rendering = render(view, gaussians, pipeline, background)["render"]
        render_pkg = render(view, gaussians, pipeline, background)
        image = render_pkg["render"]

        mv = np.eye(4)
        mv[:3, :3] = view.R.T # 旋转部分
        mv[:3, 3] = view.T   # 平移部分 (相机位置)
        mv[1, :] *= -1  # Y轴取反
        mv[2, :] *= -1  # Z轴取反
        mv = torch.tensor(mv, dtype=torch.float32)
        proj = dataset_train._get_proj()
        mvp    = proj @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        resolution = (view.image_height, view.image_width)
        spp = args.spp
        img = dataset_train[0]['img']
        sample = {
            'mv' : mv.unsqueeze(0),
            'mvp' : mvp.unsqueeze(0),
            'campos' : campos.unsqueeze(0),
            'resolution' : resolution,
            'spp' : spp, 
            'img' : img
        }

        # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
        #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
        batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）
        # 3. 调用 prepare_batch 处理（与训练循环一致）
        target = prepare_batch(batch, args.background)
        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, args)
        
        # alpha = gaussians.get_alpha
        
        mesh_image = result_dict['opt']
        mesh_image = mesh_image.detach().permute(2, 0, 1)

        # blender_image = alpha * rendering + (1 - alpha) * mesh_image
        blender_image = render_pkg["rendered_alpha"] * image + (1 - render_pkg["rendered_alpha"]) * mesh_image
        # if name == "train":
        #     gt = view.original_image[0:3, :, :]
        #     gt = gt * view.mask + background[:, None, None] * (1 - view.mask).squeeze()
        #     # blender_image = (blender_image - blender_image.min()) / (blender_image.max() - blender_image.min())
        #     # rendering = (rendering - rendering.min()) / (rendering.max() - rendering.min())

        #     torchvision.utils.save_image(mesh_image, os.path.join(mesh_image_path, '{0:05d}'.format(idx) + ".png"))
        #     torchvision.utils.save_image(blender_image, os.path.join(blender_path, '{0:05d}'.format(idx) + ".png"))
        #     torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        #     torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # else:
        #     torchvision.utils.save_image(mesh_image, os.path.join(mesh_image_path, '{0:05d}'.format(idx) + ".png"))
        #     torchvision.utils.save_image(blender_image, os.path.join(blender_path, '{0:05d}'.format(idx) + ".png"))
        #     torchvision.utils.save_image(image, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
         # 转换为OpenCV显示格式
        def tensor_to_cv(tensor):
            img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
            return (np.clip(img, 0, 1) * 255).astype(np.uint8)[..., ::-1]  # RGB->BGR

        # display_img = np.concatenate([
        #     tensor_to_cv(image),
        #     tensor_to_cv(mesh_image),
        #     tensor_to_cv(blender_image)
        # ], axis=1)
        
        # # 显示控制提示
        # cv2.putText(display_img, 
        #            f"Rotation(R/P/Y): {np.degrees([roll, pitch, yaw])}  |  Translation(X/Y/Z): [{tx:.2f}, {ty:.2f}, {tz:.2f}]",
        #            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        # cv2.putText(display_img,
        #            "W/S:Forward/Back  A/D:Left/Right  Q/E:Up/Down  Arrows:Rotation  C:Reset  ESC:Quit",
        #            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        
        # cv2.imshow("Interactive View", display_img)
        display_img = tensor_to_cv(blender_image)

        # 在图像顶部添加半透明背景用于显示文字
        overlay = display_img.copy()
        cv2.rectangle(overlay, (0,0), (display_img.shape[1], 80), (0,0,0), -1)
        display_img = cv2.addWeighted(overlay, 0.6, display_img, 0.4, 0)

        # 显示控制提示
        rot_deg = np.degrees([roll, pitch, yaw])  # 转换为角度并保留数组结构
        cv2.putText(display_img, 
           f"Rotation(R/P/Y): {rot_deg[0]:.1f}°, {rot_deg[1]:.1f}°, {rot_deg[2]:.1f}°  |  Translation: X={tx:.2f} Y={ty:.2f} Z={tz:.2f}",
           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_img,
                "W/S:Forward  A/D:Strafe  Q/E:Ascend  Arrows:Rotate  C:Reset  ESC:Quit",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        cv2.imshow("Interactive Blender View", display_img)
         
        # 处理键盘输入 方向键↑/↓：绕Y轴旋转（俯仰角 方向键←/→：绕Z轴旋转（偏航角）
        key = cv2.waitKey(1)
        if key != -1:
            # 平移控制
            if key == ord('w'): tz += step_trans    # 前移
            elif key == ord('s'): tz -= step_trans  # 后移
            elif key == ord('a'): tx -= step_trans  # 左移
            elif key == ord('d'): tx += step_trans  # 右移
            elif key == ord('q'): ty += step_trans  # 上移
            elif key == ord('e'): ty -= step_trans  # 下移
            
            # 旋转控制
            elif key == 82: pitch += step_angle     # 上箭头：绕Y轴+
            elif key == 84: pitch -= step_angle     # 下箭头：绕Y轴-
            elif key == 81: yaw += step_angle       # 左箭头：绕Z轴+
            elif key == 83: yaw -= step_angle       # 右箭头：绕Z轴-
            
            # 特殊操作
            elif key == ord('c'):                   # 重置参数
                roll = pitch = yaw = 0.0
                tx = ty = tz = 0.0
            elif key == 27: break                   # ESC退出

    cv2.destroyAllWindows()

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        new_cam = update_camera_pose(scene.getTrainCameras()[0],0.5,0,0,0,-0.01)
        new_cams=[]
        new_cams.append(scene.getTrainCameras()[0])
        # render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)
        # render_set(dataset.model_path, "modify", scene.loaded_iter, new_cams, gaussians, pipeline, background)
        render_set2(dataset.model_path, "modify", scene.loaded_iter, new_cams, gaussians, pipeline, background)



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-spp', '--spp', type=int, default=1)

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    args.mtl_override        = None                     # Override material of model
    args.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    args.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    args.env_scale           = 1.0                      # Env map intensity multiplier
    args.envmap              = None                     # HDR environment probe
    args.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    args.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    args.lock_light          = False                    # Disable light optimization in the second pass
    args.lock_pos            = False                    # Disable vertex position optimization in the second pass
    args.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    args.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    args.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    args.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    args.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    args.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    args.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    args.ks_max              = [ 1.0,  1.0,  1.0]
    args.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    args.nrm_max             = [ 1.0,  1.0,  1.0]
    args.cam_near_far        = [0.01, 100.0]
    args.learn_light         = True

    args.local_rank = 0
    args.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if args.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        args.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if args.config is not None:
        data = json.load(open(args.config, 'r'))
        for key in data:
            args.__dict__[key] = data[key]



    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)