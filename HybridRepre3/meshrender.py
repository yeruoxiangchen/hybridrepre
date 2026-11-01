import numpy as np
import pyrender
import trimesh
import imageio
import os

# 创建输出目录
output_dir = "./out"
os.makedirs(output_dir, exist_ok=True)

# 加载模型和纹理
mesh_trimesh = trimesh.load('../HybridRepre/data/test/pineapple.obj')
mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

# 加载所有相机位姿
pose_bounds = np.load('../HybridRepre/data/test/poses_bounds.npy')

# 定义相机内参（需根据实际数据调整）
camera = pyrender.PerspectiveCamera(yfov=np.pi/3, aspectRatio=1.0)

# 创建离屏渲染器（复用提升性能）
renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)

# 遍历所有位姿并渲染
for i, camera_pose in enumerate(pose_bounds):
    # 创建新场景（白色背景）
    scene = pyrender.Scene(bg_color=[255, 255, 255, 255])
    
    # 添加模型
    scene.add(mesh)
    
    # 添加相机（应用当前位姿）
    scene.add(camera, pose=camera_pose)
    
    # 添加光源（避免渲染过暗）
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
    scene.add(light, pose=camera_pose)
    
    # 渲染图像
    color, _ = renderer.render(scene)
    
    # 保存结果到文件（格式：0000.png, 0001.png...）
    output_path = os.path.join(output_dir, f"{i:04d}.png")
    imageio.imwrite(output_path, color)

print(f"渲染完成！结果保存在 {output_dir}")

# xvfb-run -s "-screen 0 1024x768x24" python meshrender.py