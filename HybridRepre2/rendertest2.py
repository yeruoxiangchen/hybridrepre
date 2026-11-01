# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import argparse
import json

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import data readers / generators
from dataset.dataset_mesh2 import DatasetMesh

# Import topology / geometry trainers
from geometry.dmtet2 import DMTetGeometry

import render.renderutils as ru
from render import obj
from render import material
from render import util
from render import mesh
from render import texture
from render import mlptexture
from render import light
from render import render

from tqdm import tqdm
import cv2
RADIUS = 3.0
# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

@torch.no_grad()
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

def validate_itr(glctx, target, geometry, args, index):
    result_dict = {}
    with torch.no_grad():

        buffers = geometry.render(glctx, target)
        result_dict['ref'] = util.rgb_to_srgb(target['img'][...,0:3])[0]
        result_dict['opt'] = util.rgb_to_srgb(buffers['image'][...,0:3])[0]
        # pred = buffers['image']
        # H, W = target['resolution'][0],target['resolution'][1]
        # pred_rgb_vis = pred.reshape(H, W, 3)
        # pred_canvas = util.rgb_to_srgb(pred_rgb_vis)
        # result_dict['opt'] = pred_canvas

        result_image = torch.cat([result_dict['opt'], result_dict['ref']], axis=1)

        return result_image, result_dict
    
def training_loop(
    glctx, 
    geometry, 
    dataloader_train, 
    args, 
    num_epochs=10,
    scheduler_update_every_step=True,
    warmup_ratio=0.01
):
    # ================== 初始化训练组件 ==================
    
    # 定义可优化参数组
    warmup_iters = warmup_ratio * args.iter
    optimizer_fn = lambda model: torch.optim.Adam(model.get_params(args.lr), eps=1e-15)
    lr_scheduler_fn = lambda optimizer: torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lr_lambda=lambda iter: min(1.0, 0.01 + 0.99 * (iter/warmup_iters)) if iter <= warmup_iters 
                               else 0.1 ** ((iter - warmup_iters) / (args.iter - 500))
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    
    # 创建初始优化器和调度器
    optimizer = optimizer_fn(geometry)
    lr_scheduler = lr_scheduler_fn(optimizer)

    # 损失函数
    loss_fn = torch.nn.L1Loss(reduction='none')
    
    # 混合精度训练
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
    
    # 总迭代次数
    global_itr=0

    # ================== 训练循环 ==================
    for epoch in range(num_epochs):
        geometry.train()
        epoch_loss = 0.0
        epoch_loss_components = {
            'alpha': 0.0,
            'color': 0.0,
            'lpips': 0.0,
            'lap': 0.0,
            'normal': 0.0,
            'edgelen': 0.0,
            'offsets': 0.0
        }

        # 进度条
        progress_bar = tqdm(dataloader_train, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, batch in enumerate(progress_bar):
            # 数据准备
            target = prepare_batch(batch, args.background)
            global_itr = global_itr + 1
            # ========= 前向传播 + 损失计算 =========
            
            # 渲染预测图像
            # buffers = geometry.render(glctx, target)
            
            # 计算图像损失
            loss, buffers = geometry.tick(glctx, target, global_itr)
            
            
            # output_dir = "outtestmesh"
            # os.makedirs(output_dir, exist_ok=True)
            # image_np = image.detach().cpu().numpy()
            # image_uint8 = (image_np * 255).astype(np.uint8)
            # image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
            # filename = os.path.join(output_dir, f"batch_idx_{batch_idx}.png")  # 添加 .png 扩展名
            # cv2.imwrite(filename, image_bgr)

            # ========= 反向传播 + 优化 =========
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            scaler.step(optimizer)
            scaler.update()
            
            # 更新进度条
            epoch_loss += loss.item()
            for key in epoch_loss_components:
                epoch_loss_components[key] += geometry.loss_record[key]
            progress_bar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "LR": f"{optimizer.param_groups[0]['lr']:.2e}"
            })

            # ========= 学习率调度 =========
            if scheduler_update_every_step:
                lr_scheduler.step()

            if global_itr in args.refine_steps:
                progress_bar.write(f'\nrefine and decimate mesh at step {global_itr}')  # 通过tqdm专用通道输出
        
                # geometry.refine_and_decimate()
                geometry.refine_and_decimate()

                # 重新初始化优化系统
                optimizer = optimizer_fn(geometry)
                lr_scheduler = lr_scheduler_fn(optimizer)
                scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)  # 重置梯度缩放器
        
        # ========= epoch学习率调整 =========
        if not scheduler_update_every_step:
            lr_scheduler.step()
        # ========= 保存检查点 =========
        # if (epoch + 1) % 10 == 0:
        #     os.makedirs(checkpoint_dir, exist_ok=True)
        #     torch.save({
        #         'epoch': epoch,
        #         'model_state_dict': geometry.state_dict(),
        #         'optimizer_state_dict': optimizer.state_dict(),
        #         'loss': epoch_loss / len(dataloader_train),
        #     }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch+1}.pth"))
        
        # ========= 验证和可视化 =========
        if (epoch + 1) % 1 == 0:
            geometry.eval()
            with torch.no_grad():
                # 使用验证函数生成示例图像
                sample = next(iter(dataloader_train))
                target = prepare_batch(sample, args.background)
                result_image, result_dict = validate_itr(glctx, target, geometry, args, epoch)
            
                output_dir = os.path.join(args.outputpath, "train_vis")
                os.makedirs(output_dir, exist_ok=True)
                image_np = result_image.detach().cpu().numpy()
                image_uint8 = (image_np * 255).astype(np.uint8)
                image_bgr = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2BGR)
                filename = os.path.join(output_dir, f"epoch_{epoch + 1}.png")  # 添加 .png 扩展名
                cv2.imwrite(filename, image_bgr)
        avg_components = {k: v/len(dataloader_train) for k, v in epoch_loss_components.items()}
        progress_bar.write(f"\nEpoch {epoch+1} Summary:")  # 使用write代替print
        progress_bar.write(f"| Avg Loss: {epoch_loss/len(dataloader_train):.4f}")
        progress_bar.write(f"| Alpha: {avg_components['alpha']:.4f} "
                        f"| Color: {avg_components['color']:.4f}")
        if args.lambda_lpips > 0:
            progress_bar.write(f"  LPIPS Loss: {avg_components['lpips']:.4f}")
        if args.lambda_lap > 0:
            progress_bar.write(f"  Laplacian Loss: {avg_components['lap']:.4f}")
        if args.lambda_normal > 0:
            progress_bar.write(f"  Normal Loss: {avg_components['normal']:.4f}")
        if args.lambda_edgelen > 0:
            progress_bar.write(f"  Edge Length Loss: {avg_components['edgelen']:.4f}")
        if args.lambda_offsets > 0:
            progress_bar.write(f"  Offsets Loss: {avg_components['offsets']:.4f}")

    with torch.no_grad():
        save_path = os.path.join(args.outputpath, 'mesh')
        os.makedirs(save_path, exist_ok=True)
        geometry.export_stage1(glctx, save_path, 1070, 1907)
        # 新增：生成整个训练集的结果
        geometry.eval()
        output_dir_all = os.path.join(args.outputpath, "train_results")
        os.makedirs(output_dir_all, exist_ok=True)
        
        for batch_idx, batch in enumerate(dataloader_train):
            # 准备数据
            target = prepare_batch(batch, args.background)
            
            # 批量推理
            result_images, result_dict = validate_itr(glctx, target, geometry, args, num_epochs)

            img = result_images.detach().cpu().numpy()
            img_uint8 = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(
                os.path.join(output_dir_all, f"{batch_idx:03d}_res.png"),
                img_bgr
            )

    return geometry

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('--loss', default='logl1', choices=['logl1', 'logl2', 'mse', 'smape', 'relmse'])
    parser.add_argument('-o', '--out-dir', type=str, default=None)
    parser.add_argument('-rm', '--ref_mesh', type=str)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--isosurface', default='dmtet', choices=['dmtet', 'flexicubes'])

    parser.add_argument('--bound', type=float, default=2, help="assume the scene is bounded in box[-bound, bound]^3, if > 1, will invoke adaptive ray marching.")
    parser.add_argument('--tcnn', action='store_true', help="use tcnn's gridencoder")
    parser.add_argument('--ind_dim', type=int, default=0, help="individual code dim, 0 to turn off")
    parser.add_argument('--ssaa', type=int, default=2, help="super sampling anti-aliasing ratio")
    parser.add_argument('--contract', action='store_true', help="apply L-INF ray contraction as in mip-nerf, only work for bound > 1, will override bound to 2.")
    parser.add_argument('--pos_gradient_boost', type=float, default=1, help="nvdiffrast option")
    parser.add_argument('-O', action='store_true', help="recommended settings")
    parser.add_argument('--sdf', action='store_true', help="use sdf instead of density for nerf")
    parser.add_argument('--lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('--lr_vert', type=float, default=1e-4, help="initial learning rate for vert optimization")
    parser.add_argument("--refine_steps_ratio", type=float, action="append", default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.7])
    parser.add_argument('--refine_remesh_size', type=float, default=0.02, help="remesh trig length")
    
    parser.add_argument('--outputpath', type=str)

    parser.add_argument('--wo_smooth', action='store_true', help="disable all smoothness regularizations")
    parser.add_argument('--lambda_lpips', type=float, default=0, help="loss scale")
    parser.add_argument('--lambda_offsets', type=float, default=0.01, help="loss scale")
    parser.add_argument('--lambda_lap', type=float, default=0.01, help="loss scale")
    parser.add_argument('--lambda_normal', type=float, default=0.01, help="loss scale")
    parser.add_argument('--lambda_edgelen', type=float, default=0.01, help="loss scale")

    args = parser.parse_args()

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
    args.cam_near_far        = [0.1, 1000.0]
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

    if args.display_res is None:
        args.display_res = args.train_res
    if args.out_dir is None:
        args.out_dir = 'out/cube_%d' % (args.train_res)
    else:
        args.out_dir = os.path.join('out', args.out_dir)

    if args.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in args.__dict__.keys():
            print(key, args.__dict__[key])
        print("---------")

    if args.bound > 1:
        args.contract = True

    if args.O:
        args.fp16 = True
        args.preload = True
        args.mark_untrained = True
        args.random_image_batch = True
        args.mesh_visibility_culling = True
        args.adaptive_num_rays = True
        args.refine = True
    
    if args.sdf:
        # args.tcnn = True # tcnn supports 2nd order gradient, which is faster than finite difference.
        # args.lambda_tv = 0 # tcnn does not support inplace TV
        args.density_thresh = 0.001 # use smaller thresh to suit density scale from sdf

        # contract background
        if args.bound > 1:
            args.contract = True
        
        args.enable_offset_nerf_grad = True # lead to more sharp texture

        # just perform remesh periodically
        args.refine_decimate_ratio = 0 # disable decimation
        args.refine_size = 0 # disable subdivision
    
    args.refine_steps = [int(round(x * args.iter)) for x in args.refine_steps_ratio]

    if args.bound > 1:
        args.contract = True

    # os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.outputpath, exist_ok=True)

    glctx = dr.RasterizeGLContext() # 管理 ​可微分光栅化（differentiable rasterization）的GPU上下文

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_dir = args.ref_mesh
    mesh_dir = os.path.join(base_dir, "mesh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_path = args.base_mesh or os.path.join(base_dir, "mesh", "mesh.obj")
    
    # 初始化模型和数据集
    geometry = DMTetGeometry(args.dmtet_grid, mesh_path, args).to(device)
    center = geometry.get_center()
    dataset_train = DatasetMesh(args, base_dir, center=center)
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch, collate_fn=dataset_train.collate)

    max_epoch = np.ceil(args.iter / len(dataset_train)).astype(np.int32)
    
    # 启动训练
    trained_geometry = training_loop(
        glctx=glctx,
        geometry=geometry,
        dataloader_train=dataloader_train,
        args=args,
        num_epochs=max_epoch,
    )
    
    # 保存最终模型
    model_path = os.path.join(args.outputpath,"final_model.pth")
    torch.save(trained_geometry.state_dict(), model_path)
    
# 单纯优化mesh
    