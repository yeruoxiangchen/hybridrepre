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

def load_texture(image_path, resize=None, mode='RGB', normalize_range=None):
    from PIL import Image
    # 加载图像并转换为RGB numpy数组
    img = Image.open(image_path).convert(mode)
    if resize is not None:
        img = img.resize(resize, Image.LANCZOS)
    img = np.array(img) / 255.0  # 归一化到[0,1]
    
    # 数值范围调整（如法线贴图需要从[0,1]转[-1,1]）
    if normalize_range == 'sRGB_to_linear':
        img = np.where(img <= 0.04045, img/12.92, ((img+0.055)/1.055)**2.4)  # sRGB转线性
    elif normalize_range == 'normal_map':
        img = img * 2 - 1  # 法线贴图[0,1] -> [-1,1]
        
    return torch.tensor(img, dtype=torch.float32).cuda()

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda') # 漫反射颜色范围
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda') # 镜面反射参数范围
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda') # 法线贴图范围
    if mlp: # 目前是 true 输入xyz，输出材质属性，MLP 将材质表示为连续函数
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0) # 拼接最小值和最大值：将漫反射（只取前三个元素）、高光和法向量的范围拼接起来，分别得到 mlp_min 和 mlp_max。
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max]) # 创建 MLP 纹理：使用几何体的包围盒（geometry.getAABB()）和设定的通道数（9个通道），生成一个 MLPTexture3D 对象，并传入范围信息
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt}) # 构造材质：生成一个材质对象，其中键 'kd_ks_normal' 对应上面生成的 MLP 纹理对象。
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

    if 0:#init_mat is not None:
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

        return result_image, result_dict

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
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        render_pkg = render(view, gaussians, pipeline, background)
        image = render_pkg["render"]

        index = int(view.image_name) 
        sample = dataset_train[index]  # 调用 __getitem__(5)
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
        gt = view.original_image[0:3, :, :]
        gt = gt * view.mask + background[:, None, None] * (1 - view.mask).squeeze()
        # blender_image = (blender_image - blender_image.min()) / (blender_image.max() - blender_image.min())
        # rendering = (rendering - rendering.min()) / (rendering.max() - rendering.min())

        torchvision.utils.save_image(mesh_image, os.path.join(mesh_image_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(blender_image, os.path.join(blender_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

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