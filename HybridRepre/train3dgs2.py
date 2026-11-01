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
import numpy  # Added to avoid issue
import torch
import torch.nn.functional as F
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

import nvdiffrast.torch as dr
from render import mesh
from dataset.dataset_mesh import DatasetMesh
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh2 import DLMesh

import numpy as np
from render import texture
from render import mlptexture
from render import material
from render import light
from render import util
import json

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat


def prepare_batch(target, bg_type='black'):
    assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'black':
        background = torch.zeros(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand(target['img'].shape[0:3] + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type
    # import matplotlib.pyplot as plt
    # img_cpu = target['img'].detach().cpu().numpy()[0]
    # # 只取 RGB 通道（前三个通道），如果你不需要透明度信息
    # img_rgb = img_cpu[..., :3]
    # # 显示图像
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img_rgb)
    # plt.axis('off')  # 不显示坐标轴
    # plt.show()
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
        result_dict['alpha'] = (buffers['shaded'][..., 3] > 0).permute(1, 2, 0).float()
        # result_dict['depth'] = buffers['depth'][..., 0].cpu().numpy()[0]
        
        # import cv2
        # mask_np = result_dict['alpha'].permute(1, 2, 0).cpu().numpy()  # 确保在 CPU 且为二
        # # 转换为 0-255 的 uint8 类型
        # mask_uint8 = (mask_np * 255).astype(np.uint8)
        # # 显示图像
        # cv2.imshow('Mask Visualization', mask_uint8)
        # cv2.waitKey(0)  # 按任意键关闭窗口
        # cv2.destroyAllWindows()
        return result_image, result_dict

def training(args, dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):

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
    # geometry = DMTetGeometry(args.dmtet_grid, args.mesh_scale, args, ref_mesh) # 加载预定义初始网格、初始化sdf和顶点形变参数
    geometry = DLMesh(ref_mesh, args)
    import glob
    mtl_files = glob.glob(os.path.join(mesh_dir, "*.mtl"))
    # mtl_path = os.path.join(mesh_dir, "mesh.mtl")
    # mat = material.load_mtl(mtl_files[0],clear_ks=True)[0]
    mat = initial_guess_material(geometry, False, args, init_mat=ref_mesh.material)

    lgt = light.load_env(args.envmap, scale=args.env_scale)
    dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, args, base_dir, validate=False, examples=(args.iter+1)*args.batch)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate)


    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None

    ema_loss_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
    
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop()

        index = int(viewpoint_cam.image_name) 
        sample = dataset_train[index]  # 调用 __getitem__(5)
        # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
        #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
        batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）

        # 3. 调用 prepare_batch 处理（与训练循环一致）
        target = prepare_batch(batch, args.background)
        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, args)
        
        # depth_ndc = result_dict['depth']  # 提取深度通道
        # near = viewpoint_cam.znear
        # far = viewpoint_cam.zfar
        # # 转换为视图空间深度
        # depth_view = (2.0 * near * far) / (far + near - depth_ndc * (far - near))
        # depth_view = torch.tensor(depth_view, dtype=torch.float32, device='cuda')
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        # Loss
        loss, Ll1 = cal_loss(opt, image, render_pkg, result_dict, viewpoint_cam, bg, iteration)
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            num_gauss = len(gaussians._xyz) # add
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0: # 100一剪 分裂和剪枝
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter): # 3000一次 重置不透明度
                    gaussians.reset_opacity()
                # remove后面可加
                #     if iteration % opt.remove_outliers_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                #         gaussians.remove_outliers(opt, iteration, linear=True)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


def cal_loss(opt, image, render_pkg, result_dict, viewpoint_cam, bg, iteration, silhouette_loss_type="bce"):
    """
    Calculate the loss of the image, contains l1 loss and ssim loss.
    l1 loss: Ll1 = l1_loss(image, gt_image)
    ssim loss: Lssim = 1 - ssim(image, gt_image)
    Optional: [silhouette loss, monodepth loss]
    """
    gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
    gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()

    mesh_image = result_dict['opt']
    mesh_image = mesh_image.detach().permute(2, 0, 1)
    mesh_mask = result_dict['alpha']
    mesh_mask = mesh_mask.detach().permute(2,0,1)
 
    blender_image = image + (1 - render_pkg["rendered_alpha"]) * mesh_image

    mask_diff = torch.abs(mesh_mask.detach() - viewpoint_cam.mask.detach()).mean(dim=0, keepdim=True).detach()
    rgb_diff = torch.abs(mesh_image.detach() - gt_image.detach()).mean(dim=0, keepdim=True).detach()

    Ll1 = l1_loss(blender_image, gt_image)
    Lssim = (1.0 - ssim(blender_image, gt_image))
    loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * Lssim

    # alpha loss
    # 差异权重函数：diff越大 → 权重越小（抑制效果减弱）
    # alpha_weights = 1 / (1 + 10 * rgb_diff)  # 10为调节系数，控制衰减速度
    alpha_weights = 1 - torch.sigmoid(10 * (rgb_diff - 0.5))
    loss_alpha = torch.mean(alpha_weights * render_pkg["rendered_alpha"]** 2)
    loss = loss + 0.1 * loss_alpha

    # # ======== 深度一致性损失 ========
    # # 1. 获取两种渲染的深度图
    # depth_gs = render_pkg["depth"]  # 3DGS渲染的深度图
    # depth_mesh = depth_view  # Mesh渲染的深度图（已从result_dict获取）

    # # 确保深度图有正确的维度 [1, H, W]
    # if depth_gs.dim() == 2:
    #     depth_gs = depth_gs.unsqueeze(0)  # 增加通道维度
    # if depth_mesh.dim() == 2:
    #     depth_mesh = depth_mesh.unsqueeze(0)
        
    # # 2. 创建有效区域掩码（两种渲染都有效的区域）
    # valid_mask_gs = (render_pkg["rendered_alpha"] > 0.01)  # GS有效区域
    # valid_mask_mesh = (mesh_mask > 0.01)  # Mesh有效区域
    # valid_region = valid_mask_gs & valid_mask_mesh  # 两者都有效的区域
    
    # # 3. 计算深度差异
    # depth_diff = torch.abs(depth_gs - depth_mesh)
    
    # # 4. 鲁棒的深度损失函数
    # if valid_region.any():
    #     # 4.1 基础L1损失
    #     depth_l1 = depth_diff[valid_region].mean()
        
    #     # # 4.2 相对深度损失（处理尺度问题）
    #     # depth_relative = depth_diff[valid_region] / (depth_mesh[valid_region] + 1e-6)
    #     # depth_relative_loss = depth_relative.mean()
        
    #     # 4.3 梯度一致性损失（保持深度边缘对齐）
    #     depth_grad_x_gs = depth_gs[:, :, 1:] - depth_gs[:, :, :-1]
    #     depth_grad_x_mesh = depth_mesh[:, :, 1:] - depth_mesh[:, :, :-1]
    #     grad_diff_x = torch.abs(depth_grad_x_gs - depth_grad_x_mesh)
        
    #     depth_grad_y_gs = depth_gs[:, 1:, :] - depth_gs[:, :-1, :]
    #     depth_grad_y_mesh = depth_mesh[:, 1:, :] - depth_mesh[:, :-1, :]
    #     grad_diff_y = torch.abs(depth_grad_y_gs - depth_grad_y_mesh)
        
    #     # 只计算有效区域的梯度损失
    #     valid_grad_region = valid_region[:, :, 1:] & valid_region[:, :, :-1]
    #     grad_loss_x = grad_diff_x[valid_grad_region].mean() if valid_grad_region.any() else 0
    #     valid_grad_region = valid_region[:, 1:, :] & valid_region[:, :-1, :]
    #     grad_loss_y = grad_diff_y[valid_grad_region].mean() if valid_grad_region.any() else 0
        
    #     # 4.4 组合深度损失
    #     depth_loss = 0.5 * depth_l1 + 0.5 * (grad_loss_x + grad_loss_y)
    # else:
    #     depth_loss = torch.tensor(0.0, device=image.device)
    
    # depth_weight = min(0.01, iteration / 100000)  # 随着训练增加权重
    # loss = loss + depth_weight * depth_loss

    mask_pred = render_pkg["rendered_alpha"]
    # mask_gt = viewpoint_cam.mask
    # mask_loss = F.binary_cross_entropy(mask_pred, mask_gt, weight=rgb_diff)
    # loss = loss + 0.01 * mask_loss

    if iteration % 100 == 0:
        import imageio
        import os
        
        # 转换图像数据为 numpy 格式
        blender_np = blender_image.permute(1, 2, 0).detach().cpu().numpy()
        opt_np = mesh_image.permute(1, 2, 0).detach().cpu().numpy() 
        gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
        image_np = image.permute(1, 2, 0).detach().cpu().numpy()
        mask_pred_np = mask_pred.permute(1, 2, 0).detach().cpu().numpy()
        
        gs_mask_np = render_pkg["rendered_alpha"].permute(1, 2, 0).detach().cpu().numpy()
        mesh_mask_np = mesh_mask.permute(1, 2, 0).detach().cpu().numpy()
        mask_np = viewpoint_cam.mask.permute(1, 2, 0).detach().cpu().numpy()
        mask_diff_np = mask_diff.permute(1, 2, 0).detach().cpu().numpy()
        rgb_diff_np = rgb_diff.permute(1, 2, 0).detach().cpu().numpy()
        # 创建输出目录
        output_dir = "output_8_iter/kitchen3"
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成五位数字序号
        iter_str = f"{iteration:05d}"
        
        # 分别保存四个图像
        imageio.imwrite(f"{output_dir}/{iter_str}_blender.png", np.clip(np.rint(blender_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        imageio.imwrite(f"{output_dir}/{iter_str}_mesh.png", np.clip(np.rint(opt_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        imageio.imwrite(f"{output_dir}/{iter_str}_gt.png", np.clip(np.rint(gt_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        imageio.imwrite(f"{output_dir}/{iter_str}_render.png", np.clip(np.rint(image_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
        import cv2
        cv2.imwrite(f"{output_dir}/{iter_str}_mask_gs.png",np.clip(np.rint(gs_mask_np * 255.0), 0, 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{iter_str}_mask_mesh.png",np.clip(np.rint(mesh_mask_np * 255.0), 0, 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{iter_str}_mask_gt.png",np.clip(np.rint(mask_np * 255.0), 0, 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{iter_str}_mask_pred.png",np.clip(np.rint(mask_pred_np * 255.0), 0, 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{iter_str}_mask_diff.png",np.clip(np.rint(mask_diff_np * 255.0), 0, 255).astype(np.uint8))
        cv2.imwrite(f"{output_dir}/{iter_str}_rgb_diff.png",np.clip(np.rint(rgb_diff_np * 255.0), 0, 255).astype(np.uint8))
    return loss, Ll1

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-spp', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

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
            
    
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
