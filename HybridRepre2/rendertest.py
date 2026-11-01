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


        
        # 调整张量格式并保存
        from PIL import Image
        output_dir = "outnew"
        os.makedirs(output_dir, exist_ok=True)  # 自动创建目录
        filename = os.path.join(output_dir, f"result_{index}.png")

        # 将PyTorch张量转换为PIL图像并保存
        image_np = result_image.cpu().numpy()        # 转为NumPy数组
        image_np = (image_np * 255).astype(np.uint8) # 从[0,1]映射到[0,255]
        Image.fromarray(image_np).save(filename)

        return result_image, result_dict

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
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
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

    os.makedirs(args.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext() # 管理 ​可微分光栅化（differentiable rasterization）的GPU上下文

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_dir = args.ref_mesh
    mesh_dir = os.path.join(base_dir, "mesh")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mesh_path = "data/pineapple_nolgt/mesh/mesh.obj"
    geometry = DMTetGeometry(args.dmtet_grid, mesh_path, args) # 加载预定义初始网格、初始化sdf和顶点形变参数
    geometry = geometry.to(device)
    
    dataset_train    = DatasetMesh(args, base_dir, examples=(args.iter+1)*args.batch)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate)
    
    
    index = 135  # 需要获取第5个样本

    # 1. 直接通过数据集获取样本
    sample = dataset_train[index]  # 调用 __getitem__(5)

    # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
    #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
    batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）

    # 3. 调用 prepare_batch 处理（与训练循环一致）
    target = prepare_batch(batch, args.background)
    result_image, result_dict = validate_itr(glctx, target, geometry, args, index)

    # # save and show
    np_img = result_dict['opt'].detach().cpu().numpy()
    util.save_image("../HybridRepre/output/nvdiffrec/pineapple2" + 
                    '/' + ('nvi_%06d_2.png' % (index)), np_img)
    print("finish")
    # 每个样本迭代
    progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc="Training", unit="batch")
    print("Running validation")
    
    
    # for it, target in progress_bar:

    #     # Mix validation background
    #     target = prepare_batch(target, args.background)

    #     result_image, result_dict = validate_itr(glctx, target, geometry, args)
        
    #     # Compute metrics
    #     opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 

    #     # # save and show
    #     np_img = result_dict['opt'].detach().cpu().numpy()
    #     util.save_image("../HybridRepre/output/nvdiffrec/pineapple3" + 
    #                     '/' + ('nvi_%06d_opt.png' % (it)), np_img)
    #     np_img2 = result_dict['ref'].detach().cpu().numpy()
    #     util.save_image("../HybridRepre/output/nvdiffrec/pineapple3" + 
    #                     '/' + ('nvi_%06d_ref.png' % (it)), np_img2)
    #     opt_np = opt.detach().cpu().numpy()
    #     # if opt_np.shape[0] == 3:  # 假设输入是 [C, H, W]
    #     #     opt_np = opt_np.transpose(1, 2, 0)  # 转为 [H, W, C]
    #     # import matplotlib.pyplot as plt
    #     # plt.figure(figsize=(8, 8))
    #     # plt.imshow(opt_np, vmin=0, vmax=1)  # 自动处理单通道/多通道
    #     # plt.axis('off')
    #     # plt.title("opt")
    #     # plt.show()
