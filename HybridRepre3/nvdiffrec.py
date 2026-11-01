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
from dataset.dataset_mesh import DatasetMesh
from dataset.dataset_nerf import DatasetNERF
from dataset.dataset_llff import DatasetLLFF

# Import topology / geometry trainers
from geometry.dmtet import DMTetGeometry
from geometry.dlmesh import DLMesh
from geometry.flexicubes_geo import FlexiCubesGeometry

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

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

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

        import matplotlib.pyplot as plt
        result_dict['depth'] = (buffers['depth'][..., 0].cpu().numpy()[0] * 255).astype(np.uint8)
        plt.imsave("zzdepth.png", result_dict['depth'], cmap='viridis')
        np.savetxt("zzz.txt", result_dict['depth'], fmt='%d', delimiter='\t')

        return result_image, result_dict


class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat # mlp预测
        self.optimize_geometry = optimize_geometry # true
        self.optimize_light = optimize_light # true
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it)# DMTetGeometry.tick() geometry定义了sdf和顶点形变 material由Mlp预测材质，输入xyz就可以知道材质，材质有9个通道

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
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for Laplacian regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = True

    FLAGS.local_rank = 0
    FLAGS.multi_gpu  = "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1
    if FLAGS.multi_gpu:
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = 'localhost'
        if "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = '23456'

        FLAGS.local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(FLAGS.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res
    if FLAGS.out_dir is None:
        FLAGS.out_dir = 'out/cube_%d' % (FLAGS.train_res)
    else:
        FLAGS.out_dir = os.path.join('out', FLAGS.out_dir)

    if FLAGS.local_rank == 0:
        print("Config / Flags:")
        print("---------")
        for key in FLAGS.__dict__.keys():
            print(key, FLAGS.__dict__[key])
        print("---------")

    os.makedirs(FLAGS.out_dir, exist_ok=True)

    glctx = dr.RasterizeGLContext() # 管理 ​可微分光栅化（differentiable rasterization）的GPU上下文

    # ==============================================================================================
    #  Create data pipeline
    # ==============================================================================================
    base_dir = FLAGS.ref_mesh
    mesh_dir = os.path.join(base_dir, "mesh")
    mesh_path = os.path.join(base_dir, "mesh/mesh.obj")
    ref_mesh         = mesh.load_mesh(mesh_path, FLAGS.mtl_override)
    geometry = DMTetGeometry(FLAGS.dmtet_grid, FLAGS.mesh_scale, FLAGS, ref_mesh) # 加载预定义初始网格、初始化sdf和顶点形变参数
    
    mtl_path = "../HybridRepre/data/china/mesh/mesh.mtl"
    mat = material.load_mtl(mtl_path,clear_ks=True)[0]
    lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
   
    dataset_train    = DatasetMesh(ref_mesh, glctx, RADIUS, FLAGS, base_dir, validate=False, examples=(FLAGS.iter+1)*FLAGS.batch)
    
    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=1, collate_fn=dataset_train.collate)
    
    
    index = 135  # 需要获取第5个样本

    # 1. 直接通过数据集获取样本
    sample = dataset_train[index]  # 调用 __getitem__(5)

    # 2. 手动应用 collate_fn（假设 dataset_train.collate 是数据集的整理函数）
    #    如果 collate_fn 是默认的，可以使用 torch.utils.data.default_collate
    batch = dataset_train.collate([sample])  # 包装成批次形式（batch_size=1）

    # 3. 调用 prepare_batch 处理（与训练循环一致）
    target = prepare_batch(batch, FLAGS.background)
    result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, FLAGS)

    # # save and show
    np_img = result_dict['opt'].detach().cpu().numpy()
    util.save_image("../HybridRepre/output/nvdiffrec/china" + 
                    '/' + ('nvi_%06d_2.png' % (index)), np_img)
    print("finish")
    # 每个样本迭代
    progress_bar = tqdm(enumerate(dataloader_train), total=len(dataloader_train), desc="Training", unit="batch")
    print("Running validation")
    for it, target in progress_bar:

        # Mix validation background
        target = prepare_batch(target, FLAGS.background)

        result_image, result_dict = validate_itr(glctx, target, geometry, mat, lgt, FLAGS)
        
        # Compute metrics
        opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 

        # # save and show
        np_img = result_dict['opt'].detach().cpu().numpy()
        util.save_image("../HybridRepre/output/nvdiffrec/gloves" + 
                        '/' + ('nvi_%06d_opt.png' % (it)), np_img)
        np_img2 = result_dict['ref'].detach().cpu().numpy()
        util.save_image("../HybridRepre/output/nvdiffrec/gloves" + 
                        '/' + ('nvi_%06d_ref.png' % (it)), np_img2)
        opt_np = opt.detach().cpu().numpy()
        # if opt_np.shape[0] == 3:  # 假设输入是 [C, H, W]
        #     opt_np = opt_np.transpose(1, 2, 0)  # 转为 [H, W, C]
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(8, 8))
        # plt.imshow(opt_np, vmin=0, vmax=1)  # 自动处理单通道/多通道
        # plt.axis('off')
        # plt.title("opt")
        # plt.show()
