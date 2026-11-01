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
import imageio
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
    geometry = DMTetGeometry(args.dmtet_grid, args.mesh_scale, args, ref_mesh) # 加载预定义初始网格、初始化sdf和顶点形变参数
    
    mtl_path = os.path.join(mesh_dir, "mesh.mtl")
    mat = material.load_mtl(mtl_path,clear_ks=True)[0]
    
    lgt = light.load_env(args.envmap, scale=args.env_scale)
    dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, args, base_dir, validate=False, examples=(args.iter+1)*args.batch)

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)

    gaussians.load_ply(checkpoint)

    gaussians.training_setup(opt)

    # 冻结除透明度外的所有参数（新增部分）
    gaussians._xyz.requires_grad_(False)
    gaussians._features_dc.requires_grad_(False)
    gaussians._features_rest.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._opacity.requires_grad_(True)

    # 重新配置优化器只优化透明度参数（新增部分）
    opacity_group = None
    for group in gaussians.optimizer.param_groups:
        if group['name'] == 'opacity':
            opacity_group = group
            break
    if opacity_group:
        gaussians.optimizer = torch.optim.Adam([opacity_group], lr=opacity_group['lr'], eps=1e-15)
    
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    target_dict = {}
    # 获取所有视角
    cameras = scene.getTrainCameras().copy()
    with torch.no_grad():
        for cam in tqdm(cameras):
            bg = torch.rand((3), device="cuda") if opt.random_background else background
            render_pkg = render(cam, gaussians, pipe, background)
            image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
            gt_image = cam.original_image.to(image.dtype).cuda()
            gt_image = gt_image * cam.mask + bg[:, None, None] * (1 - cam.mask).squeeze()
            
            index = int(cam.image_name) 
            sample = dataset_train[index]  # 调用 __getitem__(5)
            # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
            #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
            batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）

            # 3. 调用 prepare_batch 处理（与训练循环一致）
            target = prepare_batch(batch, args.background)
            result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, args)
            mesh_image = result_dict['opt']
            mesh_image = mesh_image.detach().permute(2, 0, 1)

            def compute_alpha(gt, img, mesh_img, eps=1e-6):
                numerator = gt - mesh_img
                denominator = img - mesh_img + eps
                alpha = numerator / denominator
                return torch.clamp(alpha, 0.0, 1.0).mean(dim=0, keepdim=True)  # 合并通道维度
                
            target_alpha = compute_alpha(gt_image, image, mesh_image)  # [1, H, W]
                # 释放显存
            image_name = cam.image_name
            target_dict[image_name] = target_alpha
            
            del render_pkg
            torch.cuda.empty_cache()

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

        

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        # Loss
        gt_image = viewpoint_cam.original_image.to(image.dtype).cuda()
        gt_image = gt_image * viewpoint_cam.mask + bg[:, None, None] * (1 - viewpoint_cam.mask).squeeze()

        index = int(viewpoint_cam.image_name) 
        sample = dataset_train[index]  # 调用 __getitem__(5)
        # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
        #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
        batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）

        # 3. 调用 prepare_batch 处理（与训练循环一致）
        target = prepare_batch(batch, args.background)
        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, args)
        mesh_image = result_dict['opt']
        mesh_image = mesh_image.detach().permute(2, 0, 1)
        
        target_alpha = target_dict[viewpoint_cam.image_name]  # [1, H, W]
        # alpha_loss = F.mse_loss(
        #     render_pkg["rendered_alpha"], 
        #     target_alpha,
        #     reduction='mean'
        # )
        # alpha_loss = F.smooth_l1_loss(
        #     render_pkg["rendered_alpha"], 
        #     target_alpha,
        #     beta=0.1,  # 控制平滑区间
        #     reduction='mean'
        # )
        alpha_loss = F.l1_loss(
            render_pkg["rendered_alpha"],
            target_alpha,
            reduction='mean'
        )
        loss = alpha_loss

        # def get_smooth_loss(alpha_map):
        #     # 计算水平和垂直梯度
        #     h_grad = alpha_map[:, :, 1:, :] - alpha_map[:, :, :-1, :]  # 水平差分
        #     v_grad = alpha_map[:, :, :, 1:] - alpha_map[:, :, :, :-1]  # 垂直差分
        #     return (h_grad.abs().mean() + v_grad.abs().mean()) / 2.0
        # smooth_loss = get_smooth_loss(render_pkg["rendered_alpha"].unsqueeze(0))  # 添加维度
        # loss = loss + smooth_loss
        
        loss.backward()

        if iteration % 100 == 0: 
            blender_image = render_pkg["rendered_alpha"] * image + (1 - render_pkg["rendered_alpha"]) * mesh_image
            # 转换图像数据为 numpy 格式
            blender_np = blender_image.permute(1, 2, 0).detach().cpu().numpy()
            opt_np = mesh_image.permute(1, 2, 0).detach().cpu().numpy() 
            gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            image_np = image.permute(1, 2, 0).detach().cpu().numpy() 
            target_alpha_np = target_alpha.permute(1, 2, 0).detach().cpu().numpy() 
            alpha_np = render_pkg["rendered_alpha"].permute(1, 2, 0).detach().cpu().numpy()
            # 创建输出目录
            output_dir = "../HybridRepre/output/gsrender/pineapple_itr"
            os.makedirs(output_dir, exist_ok=True)
            # 生成五位数字序号
            iter_str = f"{iteration:05d}"
            # 分别保存四个图像
            imageio.imwrite(f"{output_dir}/{iter_str}_blender.png", np.clip(np.rint(blender_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            imageio.imwrite(f"{output_dir}/{iter_str}_mesh.png", np.clip(np.rint(opt_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            imageio.imwrite(f"{output_dir}/{iter_str}_gt.png", np.clip(np.rint(gt_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            imageio.imwrite(f"{output_dir}/{iter_str}_render.png", np.clip(np.rint(image_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            import cv2
            cv2.imwrite(f"{output_dir}/{iter_str}_target_alpha.png",np.clip(np.rint(target_alpha_np * 255.0), 0, 255).astype(np.uint8))
            cv2.imwrite(f"{output_dir}/{iter_str}_alpha_np.png",np.clip(np.rint(alpha_np * 255.0), 0, 255).astype(np.uint8))
        
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

            # # Log and save
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            # if (iteration in saving_iterations):
            #     print("\n[ITER {}] Saving Gaussians".format(iteration))
            #     scene.save(iteration)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

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
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)

    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-spp', '--spp', type=int, default=1)
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
    model_path = os.path.join(args.source_path, "point_cloud.ply")
    training(args, lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, model_path, args.debug_from)

    # All done
    print("\nTraining complete.")
