import os
import numpy as np # Added to avoid issue
import torch
import torch.nn.functional as F
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import imageio

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

def render_all_views_enhanced(dataset, opt, pipe, checkpoint, 
                            resolution_scale=1.0, 
                            save_depth=False,
                            device="cuda"):
    # 初始化
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    
    # 加载模型
    if checkpoint.endswith(".ply"):
        gaussians.load_ply(checkpoint)  # 直接加载PLY文件
    else:
        model_params, _ = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    
    # 配置参数
    bg_color = [1, 1, 1] 
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)
    
    # 获取所有视角
    cameras = scene.getTrainCameras().copy()
    
    # 准备输出
    output_dir = os.path.join(dataset.model_path, f"renders")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for cam in tqdm(cameras):
            # 调整分辨率
            orig_width = cam.image_width
            orig_height = cam.image_height
            cam.image_width = int(orig_width * resolution_scale)
            cam.image_height = int(orig_height * resolution_scale)
            
            # 渲染
            render_pkg = render(cam, gaussians, pipe, background)
            
            # 保存RGB
            rgb = render_pkg["render"].permute(1,2,0).cpu().numpy()
            print(f"{output_dir}/{cam.image_name}" + "_render.png")
            imageio.imwrite(f"{output_dir}/{cam.image_name}"+"_render.png", np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            
            gt_image = cam.original_image.to(render_pkg["render"].dtype).cuda()
            gt_image = gt_image * cam.mask + background[:, None, None] * (1 - cam.mask).squeeze()
            gt_np = gt_image.permute(1, 2, 0).detach().cpu().numpy()
            imageio.imwrite(f"{output_dir}/{cam.image_name}" + "_gt.png", np.clip(np.rint(gt_np * 255.0), 0, 255).astype(np.uint8), compress_level=3)
            # 释放显存
            del render_pkg
            torch.cuda.empty_cache()

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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    model_path = os.path.join(args.source_path, "point_cloud.ply")
    render_all_views_enhanced(lp.extract(args), op.extract(args), pp.extract(args), model_path)

    # All done
    print("\nTraining complete.")
