import numpy as np
import torch
import torch.nn.functional as F
import math
import torch.nn as nn
import trimesh
import nvdiffrast.torch as dr

from .utils import safe_normalize
from pytorch3d.structures import Meshes
from pytorch3d.loss import mesh_laplacian_smoothing, mesh_normal_consistency, mesh_edge_loss

from encoding import get_encoder

from meshutils import decimate_and_refine_mesh

import os
import cv2
import xatlas
from sklearn.neighbors import NearestNeighbors
from scipy.ndimage import binary_dilation, binary_erosion
import json

TORCH_SCATTER = None # lazy import

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

def contract(xyzs):
    if isinstance(xyzs, np.ndarray):
        mag = np.max(np.abs(xyzs), axis=1, keepdims=True)
        xyzs = np.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    else:
        mag = torch.amax(torch.abs(xyzs), dim=1, keepdim=True)
        xyzs = torch.where(mag <= 1, xyzs, xyzs * (2 - 1 / mag) / mag)
    return xyzs

def scale_img_nhwc(x, size, mag='bilinear', min='bilinear'):
    assert (x.shape[1] >= size[0] and x.shape[2] >= size[1]) or (x.shape[1] < size[0] and x.shape[2] < size[1]), "Trying to magnify image in one dimension and minify in the other"
    y = x.permute(0, 3, 1, 2) # NHWC -> NCHW
    if x.shape[1] > size[0] and x.shape[2] > size[1]: # Minification, previous size was bigger
        y = torch.nn.functional.interpolate(y, size, mode=min)
    else: # Magnification
        if mag == 'bilinear' or mag == 'bicubic':
            y = torch.nn.functional.interpolate(y, size, mode=mag, align_corners=True)
        else:
            y = torch.nn.functional.interpolate(y, size, mode=mag)
    return y.permute(0, 2, 3, 1).contiguous() # NCHW -> NHWC

def scale_img_hwc(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ...], size, mag, min)[0]

def scale_img_nhw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[..., None], size, mag, min)[..., 0]

def scale_img_hw(x, size, mag='bilinear', min='bilinear'):
    return scale_img_nhwc(x[None, ..., None], size, mag, min)[0, ..., 0]

def laplacian_cot(verts, faces):
    """
    Compute the cotangent laplacian
    Inspired by https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/loss/mesh_laplacian_smoothing.html
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """

    # V = sum(V_n), F = sum(F_n)
    V, F = verts.shape[0], faces.shape[0]

    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Side lengths of each triangle, of shape (sum(F_n),)
    # A is the side opposite v1, B is opposite v2, and C is opposite v3
    A = (v1 - v2).norm(dim=1)
    B = (v0 - v2).norm(dim=1)
    C = (v0 - v1).norm(dim=1)

    # Area of each triangle (with Heron's formula); shape is (sum(F_n),)
    s = 0.5 * (A + B + C)
    # note that the area can be negative (close to 0) causing nans after sqrt()
    # we clip it to a small positive value
    area = (s * (s - A) * (s - B) * (s - C)).clamp_(min=1e-12).sqrt()

    # Compute cotangents of angles, of shape (sum(F_n), 3)
    A2, B2, C2 = A * A, B * B, C * C
    cota = (B2 + C2 - A2) / area
    cotb = (A2 + C2 - B2) / area
    cotc = (A2 + B2 - C2) / area
    cot = torch.stack([cota, cotb, cotc], dim=1)
    cot /= 4.0

    # Construct a sparse matrix by basically doing:
    # L[v1, v2] = cota
    # L[v2, v0] = cotb
    # L[v0, v1] = cotc
    ii = faces[:, [1, 2, 0]]
    jj = faces[:, [2, 0, 1]]
    idx = torch.stack([ii, jj], dim=0).view(2, F * 3)
    L = torch.sparse.FloatTensor(idx, cot.view(-1), (V, V))

    # Make it symmetric; this means we are also setting
    # L[v2, v1] = cota
    # L[v0, v2] = cotb
    # L[v1, v0] = cotc
    L += L.t()

    # Add the diagonal indices
    vals = torch.sparse.sum(L, dim=0).to_dense()
    indices = torch.arange(V, device='cuda')
    idx = torch.stack([indices, indices], dim=0)
    L = torch.sparse.FloatTensor(idx, vals, (V, V)) - L
    return L

def laplacian_uniform(verts, faces):
    """
    Compute the uniform laplacian
    Parameters
    ----------
    verts : torch.Tensor
        Vertex positions.
    faces : torch.Tensor
        array of triangle faces.
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Neighbor indices
    ii = faces[:, [1, 2, 0]].flatten()
    jj = faces[:, [2, 0, 1]].flatten()
    adj = torch.stack([torch.cat([ii, jj]), torch.cat([jj, ii])], dim=0).unique(dim=1)
    adj_values = torch.ones(adj.shape[1], device=verts.device, dtype=torch.float)

    # Diagonal indices
    diag_idx = adj[0]

    # Build the sparse matrix
    idx = torch.cat((adj, torch.stack((diag_idx, diag_idx), dim=0)), dim=1)
    values = torch.cat((-adj_values, adj_values))

    # The coalesce operation sums the duplicate indices, resulting in the
    # correct diagonal
    return torch.sparse_coo_tensor(idx, values, (V,V)).coalesce()

def laplacian_smooth_loss(verts, faces, cotan=False):
    with torch.no_grad():
        if cotan:
            L = laplacian_cot(verts, faces.long())
            norm_w = torch.sparse.sum(L, dim=1).to_dense().view(-1, 1)
            mask = norm_w > 0
            norm_w[mask] = 1.0 / norm_w[mask]
        else:
            L = laplacian_uniform(verts, faces.long())
    if cotan:
        loss = L.mm(verts) * norm_w - verts
    else:
        loss = L.mm(verts)
    loss = loss.norm(dim=1)
    loss = loss.mean()
    return loss

class MLP(nn.Module):
    def __init__(self, dim_in, dim_out, dim_hidden, num_layers, bias=True, geom_init=False, weight_norm=False):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_hidden = dim_hidden
        self.num_layers = num_layers
        self.geom_init = geom_init

        net = []
        for l in range(num_layers):

            in_dim = self.dim_in if l == 0 else self.dim_hidden
            out_dim = self.dim_out if l == num_layers - 1 else self.dim_hidden

            net.append(nn.Linear(in_dim, out_dim, bias=bias))
        
            if geom_init:
                if l == num_layers - 1:
                    torch.nn.init.normal_(net[l].weight, mean=math.sqrt(math.pi) / math.sqrt(in_dim), std=1e-4)
                    if bias: torch.nn.init.constant_(net[l].bias, -0.5) # sphere init (very important for hashgrid encoding!)
                    # nn.init.uniform_(net[l].weight, -1e-2, 1e-2)

                elif l == 0:
                    torch.nn.init.normal_(net[l].weight[:, :3], 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    torch.nn.init.constant_(net[l].weight[:, 3:], 0.0)
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)

                else:
                    torch.nn.init.normal_(net[l].weight, 0.0, math.sqrt(2) / math.sqrt(out_dim))
                    if bias: torch.nn.init.constant_(net[l].bias, 0.0)
            
            if weight_norm:
                net[l] = nn.utils.weight_norm(net[l])

        self.net = nn.ModuleList(net)
    
    def forward(self, x):
        for l in range(self.num_layers):
            x = self.net[l](x)
            if l != self.num_layers - 1:
                if self.geom_init:
                    x = F.softplus(x, beta=100)
                else:
                    x = F.relu(x, inplace=True)
        return x

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, mesh_path, opt=None, specular_dim=3):
        super(DMTetGeometry, self).__init__()
        self.grid_res      = grid_res
        self.opt = opt
        self.ssaa = opt.ssaa
        self.bound = opt.bound
        self.max_level = 16
        self.device = 'cuda'
        self.cascade = 1 + math.ceil(math.log2(self.bound))

        vertices = []
        triangles = []
        v_cumsum = [0]
        f_cumsum = [0]
        refmesh = trimesh.load(mesh_path, skip_material=True, process=False)
        self.mesh = refmesh
        self.center = self.cal_center()
        refmesh.vertices -= self.center
        print(f"vertices range: x [{refmesh.vertices[:,0].min()}, {refmesh.vertices[:,0].max()}]")
        print(f"vertices range: y [{refmesh.vertices[:,1].min()}, {refmesh.vertices[:,1].max()}]")
        print(f"vertices range: z [{refmesh.vertices[:,2].min()}, {refmesh.vertices[:,2].max()}]")
        # refmesh = self.normalize_mesh(refmesh)  
        vertices.append(refmesh.vertices)
        triangles.append(refmesh.faces + v_cumsum[-1])

        v_cumsum.append(v_cumsum[-1] + refmesh.vertices.shape[0])
        f_cumsum.append(f_cumsum[-1] + refmesh.faces.shape[0])

        vertices = np.concatenate(vertices, axis=0)
        triangles = np.concatenate(triangles, axis=0)
        self.v_cumsum = np.array(v_cumsum)
        self.f_cumsum = np.array(f_cumsum)

        # must put to cuda manually, we don't want these things in the model as buffers...
        self.vertices = torch.from_numpy(vertices).float().cuda() # [N, 3]
        self.triangles = torch.from_numpy(triangles).int().cuda()

        # learnable offsets for mesh vertex
        self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))

        # accumulate error for mesh face
        self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
        self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32).cuda()
        self.triangles_errors_id = None

        # color network
        self.encoder_color, self.in_dim_color = get_encoder("hashgrid_tcnn" if self.opt.tcnn else "hashgrid", level_dim=2, desired_resolution=2048 * self.bound, interpolation='linear')
        self.color_net = MLP(3 + self.in_dim_color + self.opt.ind_dim, 3 + specular_dim, 64, 3, bias=True, geom_init=True)

        self.encoder_dir, self.in_dim_dir = get_encoder("None")
        self.specular_net = MLP(specular_dim + self.in_dim_dir, 3, 32, 2, bias=True, geom_init=True)

        # optionally use LPIPS loss for patch-based training
        if self.opt.lambda_lpips > 0:
            import lpips
            self.criterion_lpips = lpips.LPIPS(net='vgg').to(self.device)
        
        self.loss_record ={
            'alpha': 0.0,
            'color': 0.0,
            'lpips': 0.0,
            'lap': 0.0,
            'normal': 0.0,
            'edgelen': 0.0,
            'offsets': 0.0,
        }

    def cal_center(self):
        center = (self.mesh.bounds[0] + self.mesh.bounds[1]) / 2.0
        return center
    def get_center(self):
        return self.center

    def normalize_mesh(self,refmesh):
        # 计算模型包围盒
        vertices = refmesh.vertices.astype(np.float32)
        # Step 1: 计算包围盒中心
        vmin = np.min(vertices, axis=0)
        vmax = np.max(vertices, axis=0)
        center = (vmin + vmax) / 2.0

        # Step 2: 计算缩放因子（使模型适配到[-1,1]^3）
        span = vmax - vmin
        max_span = np.max(span)
        scale = 2.0 / max_span if max_span > 0 else 1.0

        # Step 3: 平移并缩放顶点
        vertices_normalized = (vertices - center) * scale

        # Step 4: 更新网格顶点
        refmesh.vertices = vertices_normalized
        print(f"Normalized vertices range: x [{vertices_normalized[:,0].min()}, {vertices_normalized[:,0].max()}]")
        print(f"Normalized vertices range: y [{vertices_normalized[:,1].min()}, {vertices_normalized[:,1].max()}]")
        print(f"Normalized vertices range: z [{vertices_normalized[:,2].min()}, {vertices_normalized[:,2].max()}]")

        return refmesh
    def geo_feat(self, x, c=None):

        h = self.encoder_color(x, bound=self.bound, max_level=self.max_level)
        h = torch.cat([x, h], dim=-1)
        if c is not None:
            h = torch.cat([h, c.repeat(x.shape[0], 1) if c.shape[0] == 1 else c], dim=-1)
        h = self.color_net(h)
        geo_feat = torch.sigmoid(h)

        return geo_feat


    def rgb(self, x, d, c=None, shading='full'):

        # color
        geo_feat = self.geo_feat(x, c)
        diffuse = geo_feat[..., :3]

        if shading == 'diffuse':
            color = diffuse
            specular = None
        else: 
            d = self.encoder_dir(d)
            specular = self.specular_net(torch.cat([d, geo_feat[..., 3:]], dim=-1))
            specular = torch.sigmoid(specular)
            if shading == 'specular':
                color = specular
            else: # full
                # color = (specular + diffuse).clamp(0, 1) # specular + albedo 似乎specular + diffuse始终大于1
                # color = (specular + diffuse) / 2 # specular + albedo 似乎specular + diffuse始终大于1
                color = diffuse
                # hdr_color = diffuse + specular
                # color = hdr_color / (hdr_color + 1)  # Reinhard变体
                # color = torch.clamp(color, 0, 1)     # 二次保护
                # kd = 1 - specular.amax(dim=-1, keepdim=True)  # 动态调整diffuse系数
                # color = kd * diffuse + specular

        return color, specular

    @torch.no_grad()
    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def render_stage1(self, glctx, rays_o, rays_d, mvp, h0, w0, bg_color=None, shading='full', **kwargs):

        prefix = rays_d.shape[:-1]
        rays_d = rays_d.contiguous().view(-1, 3)

        N = rays_d.shape[0] # N = B * N, in fact
        device = rays_d.device
            
        # do super-sampling 超采样 将射线方向矩阵上采样到更高分辨率，提升边缘质量
        if self.ssaa > 1:
            h = int(h0 * self.ssaa)
            w = int(w0 * self.ssaa)
            # interpolate rays_d when ssaa > 1 ...
            dirs = rays_d.view(h0, w0, 3)
            dirs = scale_img_hwc(dirs, (h, w), mag='nearest').view(-1, 3).contiguous()
        else:
            h, w = h0, w0
            dirs = rays_d.contiguous()

        dirs = safe_normalize(dirs)

        # mix background color
        if bg_color is None:
            bg_color = 1

        # [N, 3] to [h, w, 3]
        if torch.is_tensor(bg_color) and len(bg_color.shape) == 2:
            bg_color = bg_color.view(h0, w0, 3)

        ind_code = None

        results = {}

        vertices = self.vertices + self.vertices_offsets # [N, 3]

        vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), torch.transpose(mvp, 0, 1)).float().unsqueeze(0) # [1, N, 4] # 模型->裁剪空间
        # vertices_clip = torch.matmul(F.pad(vertices, pad=(0, 1), mode='constant', value=1.0), mvp).float().unsqueeze(0)
        # from render import renderutils as ru
        # vertices_clip = ru.xfm_points(vertices[None,...], mvp.unsqueeze(0))
        
        # import matplotlib
        # matplotlib.use('Agg')  # 无GUI环境需要这行
        # import matplotlib.pyplot as plt
        # import os

        # # 创建输出目录
        # save_dir = "outnew"
        # os.makedirs(save_dir, exist_ok=True)

        # # 提取坐标和w分量
        # x = vertices_clip[0, :, 0].cpu().detach().numpy()
        # y = vertices_clip[0, :, 1].cpu().detach().numpy()
        # z = vertices_clip[0, :, 2].cpu().detach().numpy()
        # ww = vertices_clip[0, :, 3].cpu().detach().numpy()
        # print(f'x:{np.min(x)}~{np.max(x)}')
        # print(f'x:{np.min(y)}~{np.max(y)}')
        # print(f'x:{np.min(z)}~{np.max(z)}')
        # print(f'x:{np.min(ww)}~{np.max(ww)}')
        # # 计算归一化坐标（透视除法前）
        # x_normalized = x / ww
        # y_normalized = y / ww

        # # 创建可视化画布
        # plt.figure(figsize=(10, 10))

        # # 绘制有效顶点（在裁剪空间可见范围内）
        # valid_mask = (np.abs(x) <= np.abs(ww)) & (np.abs(y) <= np.abs(ww))
        # plt.scatter(x_normalized[valid_mask], y_normalized[valid_mask], 
        #             s=2, c='blue', alpha=0.5, label='Visible Points')

        # # 绘制裁剪边界（视锥体边界）
        # border_x = np.array([-1, 1, 1, -1, -1])
        # border_y = np.array([-1, -1, 1, 1, -1])
        # plt.plot(border_x, border_y, 'r-', linewidth=2, label='Clip Space Border')

        # # 设置坐标轴
        # plt.xlim(-1.2, 1.2)
        # plt.ylim(-1.2, 1.2)
        # plt.xlabel('X/W')
        # plt.ylabel('Y/W')
        # plt.title(f'Clip Space Projection ({len(x)} points)')
        # plt.grid(True)
        # plt.legend()

        # # 保存图像
        # save_path = os.path.join(save_dir, "vetclip.png")
        # plt.savefig(save_path, bbox_inches='tight', dpi=150)
        # plt.close()
        # print(f"Clip space visualization saved to: {save_path}")

        rast, _ = dr.rasterize(glctx, vertices_clip, self.triangles, (h, w)) # 使用Nvdiffrast生成光栅化信息rast，包含像素对应的三角形ID、深度等

        xyzs, _ = dr.interpolate(vertices.unsqueeze(0), rast, self.triangles) # [1, H, W, 3] 像素在空间中的三维坐标
        mask, _ = dr.interpolate(torch.ones_like(vertices[:, :1]).unsqueeze(0), rast, self.triangles) # [1, H, W, 1] 像素在空间中是否可见
        mask_flatten = (mask > 0).view(-1).detach()
        xyzs = xyzs.view(-1, 3)
        
        # mask_np = mask[0, ..., 0].detach().cpu().numpy()  # [H, W]
        # import matplotlib.pyplot as plt
        # # 创建图像
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mask_np, cmap='gray', vmin=0, vmax=1)
        # plt.title("Rasterization Mask (White=Covered)")
        # plt.axis('off')
        # plt.savefig("outnew/mask.png", bbox_inches='tight', pad_inches=0)
        
        # random noise to make appearance more robust
        # if self.training:
        #     xyzs = xyzs + torch.randn_like(xyzs) * 1e-3
        if self.opt.contract:
            xyzs = contract(xyzs)

        rgbs = torch.zeros(h * w, 3, device=device, dtype=torch.float32) # 颜色预测

        if mask_flatten.any():
            with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                mask_rgbs, masked_specular = self.rgb(
                    xyzs[mask_flatten], 
                    dirs[mask_flatten], ind_code, shading)

            rgbs[mask_flatten] = mask_rgbs.float()

        rgbs = rgbs.view(1, h, w, 3)
        alphas = mask.float()
        # 抗锯齿：通过Nvdiffrast的antialias减少几何边缘锯齿。
        # ​透明度混合：按alpha混合前景与背景
        alphas = dr.antialias(alphas, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)
        rgbs = dr.antialias(rgbs, rast, vertices_clip, self.triangles, pos_gradient_boost=self.opt.pos_gradient_boost).squeeze(0).clamp(0, 1)

        image = alphas * rgbs 
        depth = alphas * rast[0, :, :, [2]]
        T = 1 - alphas

        # trig_id for updating trig errors
        trig_id = rast[0, :, :, -1] - 1 # [h, w]

        # ssaa
        if self.opt.ssaa > 1:
            image = scale_img_hwc(image, (h0, w0))
            depth = scale_img_hwc(depth, (h0, w0))
            alphas = scale_img_hwc(alphas, (h0, w0))
            T = scale_img_hwc(T, (h0, w0))
            trig_id = scale_img_hw(trig_id.float(), (h0, w0), mag='nearest', min='nearest')
        
        self.triangles_errors_id = trig_id

        image = image + T * bg_color
        
        # image = image.view(*prefix, 3)
        depth = depth.view(*prefix)
        image = torch.cat([image, alphas.unsqueeze(0)], dim=-1)
        results['depth'] = depth
        results['image'] = image
        results['weights_sum'] = 1 - T
        
        # import os
        # import matplotlib.pyplot as plt
        # import numpy as np
        # from PIL import Image
        # output_dir = "outnew"
        # os.makedirs(output_dir, exist_ok=True)

        # # 将张量转为CPU numpy数组
        # def tensor_to_numpy(tensor):
        #     return tensor.detach().cpu().numpy().clip(0, 1)  # 确保数值在[0,1]

        # # 保存rgbs（HxWx3）
        # rgbs_np = tensor_to_numpy(rgbs.squeeze(0))
        # rgbs_uint8 = (rgbs_np * 255).astype(np.uint8)
        # Image.fromarray(rgbs_uint8).save(os.path.join(output_dir, "rgbs.png"))

        # # 保存alphas（HxWx1）
        # alphas_np = tensor_to_numpy(alphas.squeeze(0))
        # alphas_uint8 = (alphas_np * 255).astype(np.uint8)
        # Image.fromarray(alphas_uint8.squeeze(-1), mode='L').save(os.path.join(output_dir, "alphas.png"))

        # # 保存image（HxWx3）
        # image_np = tensor_to_numpy(image.squeeze(0))
        # image_uint8 = (image_np * 255).astype(np.uint8)
        # Image.fromarray(image_uint8).save(os.path.join(output_dir, "image.png"))

        return results
    
    def render(self, glctx, target):
        
        H, W = target['resolution'][0], target['resolution'][1]
        device = self.vertices.device
        # 将输入数据移动到对应设备
        rays_o = target['rays_o'].to(device)
        rays_d = target['rays_d'].to(device)
        mvp = target['mvp'].squeeze(0).to(device)
        bg_color = target['background'].to(device) if torch.is_tensor(target['background']) else target['background']
        res = self.render_stage1(glctx, rays_o, rays_d, mvp, H, W, bg_color)
        return res

    @torch.no_grad()
    def update_triangles_errors(self, loss):
        # loss: [H, W], detached! 通过像素级误差反向映射到三角形，识别高频误差区域
        # 累积每个三角形在渲染过程中的误差统计信息，记录每个三角形在图像平面上的累计损失值和出现次数，为后续的网格优化（如自适应细分或顶点调整）提供数据支持
        # always call after render_stage1, so self.triangles_errors_id is not None.
        indices = self.triangles_errors_id.view(-1).long() # 来自render_stage1的光栅化输出，形状 [H, W]，每个像素记录其覆盖的三角形ID（无效区域为-1）
        mask = (indices >= 0) 

        indices = indices[mask].contiguous() # 通过mask过滤掉无效ID（indices < 0），保留有效三角形索引
        values = loss.view(-1)[mask].contiguous()

        global TORCH_SCATTER

        if TORCH_SCATTER is None:
            import torch_scatter
            TORCH_SCATTER = torch_scatter

        TORCH_SCATTER.scatter_add(values, indices, out=self.triangles_errors) # 累加每个三角形覆盖的所有像素的损失总和
        TORCH_SCATTER.scatter_add(torch.ones_like(values), indices, out=self.triangles_errors_cnt) # 累加每个三角形覆盖的像素数量

        self.triangles_errors_id = None

    def tick(self, glctx, target, iteration, loss_fn = None):

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        buffers = self.render(glctx, target)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        # img_loss = torch.nn.functional.mse_loss(buffers['image'][..., 3:], color_ref[..., 3:], reduction='none') 
        # loss = img_loss + loss_fn(buffers['image'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        loss_acc = torch.nn.functional.mse_loss(buffers['image'][..., 0:3], color_ref[..., 0:3], reduction='none').mean(-1).view(-1)
        self.update_triangles_errors(loss_acc.detach())

        alpha_pred = buffers['image'][..., 3:]          # 预测的Alpha [H, W, 1]
        alpha_gt = color_ref[..., 3:]                    # 真实的Alpha [H, W, 1]
        alpha_loss = F.mse_loss(alpha_pred, alpha_gt)    # MSE损失

        # RGB通道的L1损失（仅在Alpha>0的区域加权）
        rgb_pred = buffers['image'][..., 0:3]           # 预测的RGB [H, W, 3]
        rgb_gt = color_ref[..., 0:3]                     # 真实的RGB [H, W, 3]
        mask = alpha_gt.detach()                         # 用真实Alpha作为遮罩 [H, W, 1]
        #color_loss = F.l1_loss(rgb_pred * mask, rgb_gt * mask)  # 只计算有效区域的L1
        
        valid_pixels = mask.sum()  # 或 (mask > 0).sum()
        # 方法一：手动计算加权均值
        l1_loss = torch.abs(rgb_pred - rgb_gt) * mask
        color_loss = l1_loss.sum() / (valid_pixels + 1e-6)  # 避免除零
        
        self.loss_record['color'] = color_loss.item()
        self.loss_record['alpha'] = alpha_loss.item()

        # 合并损失（可根据需要调整权重）
        loss = alpha_loss + color_loss
        # perceptual loss
        if self.opt.lambda_lpips > 0:
            pred_rgb = buffers['image'][..., 0:3] * color_ref[..., 3:]
            gt_rgb = color_ref[..., 0:3] * color_ref[..., 3:]
            # 获取图像尺寸
            H, W = pred_rgb.shape[0], pred_rgb.shape[1]
            # loss = loss + self.opt.lambda_lpips * self.criterion_lpips(pred_rgb.view(1, H, W, 3).permute(0,3,1,2).contiguous(), gt_rgb.view(1, H, W, 3).permute(0,3,1,2).contiguous())
            lpips_loss =  self.criterion_lpips(pred_rgb.view(1, H, W, 3).permute(0,3,1,2).contiguous(), gt_rgb.view(1, H, W, 3).permute(0,3,1,2).contiguous())
            loss = loss + self.opt.lambda_lpips * lpips_loss
            self.loss_record['lpips'] = self.opt.lambda_lpips * lpips_loss.item()
            
        _mesh = None
        speed = iteration / self.opt.iter
        def get_dynamic_factor(initial_lambda, speed, eps=1e-6):
            """计算动态乘数，使得最终权重接近1
            Args:
                initial_lambda: 初始lambda值（如0.01）
                speed: 训练进度[0,1]
            Returns:
                动态乘数因子，最终满足 initial_lambda * factor ≈ 1
            """
            # 计算理论最大乘数（使得最终权重=1）
            max_factor = (1.0 - eps) / initial_lambda  # 保留微小余量
            
            # 三阶段控制
            if speed < 0.4:
                # 阶段1：线性预热 (0-40%进度)
                factor = 1.0 + (max_factor - 1.0) * (speed / 0.4) * 0.3
            elif speed < 0.8:
                # 阶段2：二次加速 (40-80%进度)
                phase_speed = (speed - 0.4) / 0.4
                factor = 1.0 + (max_factor - 1.0) * (0.3 + 0.5 * phase_speed**2)
            else:
                # 阶段3：平方根饱和 (80-100%进度)
                phase_speed = (speed - 0.8) / 0.2
                factor = 1.0 + (max_factor - 1.0) * (0.8 + 0.2 * math.sqrt(phase_speed))
            
            return min(factor, max_factor)  # 确保不超过理论最大值
        
        if self.opt.lambda_lap > 0:
            dynamic_factor = get_dynamic_factor(self.opt.lambda_lap, speed)
            loss_lap = laplacian_smooth_loss(self.vertices + self.vertices_offsets, self.triangles)
            loss = loss + self.opt.lambda_lap * loss_lap * dynamic_factor
            self.loss_record['lap'] = loss_lap.item() * dynamic_factor * self.opt.lambda_lap

        if self.opt.lambda_normal > 0:
            if _mesh is None:
                _mesh = Meshes(verts=[self.vertices + self.vertices_offsets], faces=[self.triangles])
            dynamic_factor = get_dynamic_factor(self.opt.lambda_normal, speed)
            loss_normal = mesh_normal_consistency(_mesh)
            loss = loss + self.opt.lambda_normal * loss_normal * dynamic_factor
            self.loss_record['normal'] = loss_normal.item() * dynamic_factor * self.opt.lambda_normal
        
        if self.opt.lambda_edgelen > 0:
            dynamic_factor = get_dynamic_factor(self.opt.lambda_edgelen, speed)
            if _mesh is None:
                _mesh = Meshes(verts=[self.vertices + self.vertices_offsets], faces=[self.triangles])
            loss_edge = mesh_edge_loss(_mesh)
            loss = loss + self.opt.lambda_edgelen * loss_edge * dynamic_factor
            self.loss_record['edgelen'] = loss_edge.item() * dynamic_factor * self.opt.lambda_edgelen

        # L1 reg on vertices offsets
        if self.opt.lambda_offsets > 0:

            if self.opt.bound > 1:
                abs_offsets_inner = self.vertices_offsets[:self.model.v_cumsum[1]].abs()
                abs_offsets_outer = self.vertices_offsets[self.model.v_cumsum[1]:].abs()
            else:
                abs_offsets_inner = self.vertices_offsets.abs()

            # inner mesh
            # loss_offsets = torch.where(abs_offsets_inner > 0.01, abs_offsets_inner * 100, abs_offsets_inner * 0.01).sum(-1).mean()
            loss_offsets = (abs_offsets_inner ** 2).sum(-1).mean()

            # outer mesh
            if self.opt.bound > 1:
                # loss_offsets = loss_offsets + torch.where(abs_offsets_outer > 0.02, abs_offsets_outer * 100, abs_offsets_outer * 0.01).sum(-1).mean()
                loss_offsets = loss_offsets + 0.1 * (abs_offsets_outer ** 2).sum(-1).mean()
            dynamic_factor = get_dynamic_factor(self.opt.lambda_offsets, speed)
            loss = loss + self.opt.lambda_offsets * loss_offsets * dynamic_factor
            self.loss_record['offsets'] = loss_offsets.item() * dynamic_factor * self.opt.lambda_offsets

        return loss, buffers
    
   
    def get_params(self, lr):
        params = []
        params.extend([
            {'params': self.encoder_color.parameters(), 'lr': lr},
            {'params': self.color_net.parameters(), 'lr': lr}, 
            {'params': self.specular_net.parameters(), 'lr': lr}, 
            {'params': self.vertices_offsets, 'lr': self.opt.lr_vert, 'weight_decay': 0}
        ])

        return params
    
    def export_stage1(self, glctx, path, h0=2048, w0=2048, png_compression_level=3):
        # png_compression_level: 0 is no compression, 9 is max (default will be 3)

        device = self.vertices.device

        def _export_obj(v, f, h0, w0, ssaa=1, cas=0):
            # v, f: torch Tensor

            v_np = v.cpu().numpy() # [N, 3]
            f_np = f.cpu().numpy() # [M, 3]

            print(f'[INFO] running xatlas to unwrap UVs for mesh: v={v_np.shape} f={f_np.shape}')

            # unwrap uv in contracted space
            atlas = xatlas.Atlas()
            atlas.add_mesh(contract(v_np) if self.opt.contract else v_np, f_np)
            chart_options = xatlas.ChartOptions()
            chart_options.max_iterations = 0 # disable merge_chart for faster unwrap...
            pack_options = xatlas.PackOptions()
            # pack_options.blockAlign = True
            # pack_options.bruteForce = False
            atlas.generate(chart_options=chart_options, pack_options=pack_options)
            vmapping, ft_np, vt_np = atlas[0] # [N], [M, 3], [N, 2]

            # vmapping, ft_np, vt_np = xatlas.parametrize(v_np, f_np) # [N], [M, 3], [N, 2]

            vt = torch.from_numpy(vt_np.astype(np.float32)).float().to(device)
            ft = torch.from_numpy(ft_np.astype(np.int64)).int().to(device)

            # render uv maps
            uv = vt * 2.0 - 1.0 # uvs to range [-1, 1]
            uv = torch.cat((uv, torch.zeros_like(uv[..., :1]), torch.ones_like(uv[..., :1])), dim=-1) # [N, 4]

            if ssaa > 1:
                h = int(h0 * ssaa)
                w = int(w0 * ssaa)
            else:
                h, w = h0, w0

            rast, _ = dr.rasterize(glctx, uv.unsqueeze(0), ft, (h, w)) # [1, h, w, 4]
            xyzs, _ = dr.interpolate(v.unsqueeze(0), rast, f) # [1, h, w, 3]
            mask, _ = dr.interpolate(torch.ones_like(v[:, :1]).unsqueeze(0), rast, f) # [1, h, w, 1]

            # masked query 
            xyzs = xyzs.view(-1, 3)
            mask = (mask > 0).view(-1)

            if self.opt.contract:
                xyzs = contract(xyzs)
            
            feats = torch.zeros(h * w, 6, device=device, dtype=torch.float32)

            if mask.any():
                xyzs = xyzs[mask] # [M, 3]

                # check individual codes
                if self.opt.ind_dim > 0:
                    ind_code = self.individual_codes[[0]]
                else:
                    ind_code = None

                # batched inference to avoid OOM
                all_feats = []
                head = 0
                while head < xyzs.shape[0]:
                    tail = min(head + 640000, xyzs.shape[0])
                    with torch.cuda.amp.autocast(enabled=self.opt.fp16):
                        all_feats.append(self.geo_feat(xyzs[head:tail], ind_code).float())
                    head += 640000

                feats[mask] = torch.cat(all_feats, dim=0)
            
            feats = feats.view(h, w, -1) # 6 channels
            mask = mask.view(h, w)
            from render import util
            albedo = feats[..., :3]
            albedo_srgb = util.rgb_to_srgb(albedo)
            feats[..., :3] = albedo_srgb
            # quantize [0.0, 1.0] to [0, 255]
            feats = feats.cpu().detach().numpy()
            feats = (feats * 255).astype(np.uint8)

            ### NN search as a queer antialiasing ...
            mask = mask.cpu().numpy()

            inpaint_region = binary_dilation(mask, iterations=32) # pad width
            inpaint_region[mask] = 0

            search_region = mask.copy()
            not_search_region = binary_erosion(search_region, iterations=3)
            search_region[not_search_region] = 0

            search_coords = np.stack(np.nonzero(search_region), axis=-1)
            inpaint_coords = np.stack(np.nonzero(inpaint_region), axis=-1)

            knn = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(search_coords)
            _, indices = knn.kneighbors(inpaint_coords)

            feats[tuple(inpaint_coords.T)] = feats[tuple(search_coords[indices[:, 0]].T)]

            # do ssaa after the NN search, in numpy
            feats0 = cv2.cvtColor(feats[..., :3], cv2.COLOR_RGB2BGR) # albedo
            feats1 = cv2.cvtColor(feats[..., 3:], cv2.COLOR_RGB2BGR) # visibility features

            if ssaa > 1:
                feats0 = cv2.resize(feats0, (w0, h0), interpolation=cv2.INTER_LINEAR)
                feats1 = cv2.resize(feats1, (w0, h0), interpolation=cv2.INTER_LINEAR)

            # cv2.imwrite(os.path.join(path, f'feat0_{cas}.png'), feats0, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            # cv2.imwrite(os.path.join(path, f'feat1_{cas}.png'), feats1, [int(cv2.IMWRITE_PNG_COMPRESSION), png_compression_level])
            cv2.imwrite(os.path.join(path, f'feat0_{cas}.jpg'), feats0)
            cv2.imwrite(os.path.join(path, f'feat1_{cas}.jpg'), feats1)

            # save obj (v, vt, f /)
            obj_file = os.path.join(path, f'mesh_{cas}.obj')
            mtl_file = os.path.join(path, f'mesh_{cas}.mtl')

            print(f'[INFO] writing obj mesh to {obj_file}')
            with open(obj_file, "w") as fp:

                fp.write(f'mtllib mesh_{cas}.mtl \n')
                
                print(f'[INFO] writing vertices {v_np.shape}')
                for v in v_np:
                    fp.write(f'v {v[0]} {v[1]} {v[2]} \n')
            
                print(f'[INFO] writing vertices texture coords {vt_np.shape}')
                for v in vt_np:
                    fp.write(f'vt {v[0]} {1 - v[1]} \n') 

                print(f'[INFO] writing faces {f_np.shape}')
                fp.write(f'usemtl defaultMat \n')
                for i in range(len(f_np)):
                    fp.write(f"f {f_np[i, 0] + 1}/{ft_np[i, 0] + 1} {f_np[i, 1] + 1}/{ft_np[i, 1] + 1} {f_np[i, 2] + 1}/{ft_np[i, 2] + 1} \n")

            with open(mtl_file, "w") as fp:
                fp.write(f'newmtl defaultMat \n')
                fp.write(f'Ka 1 1 1 \n')
                fp.write(f'Kd 1 1 1 \n')
                fp.write(f'Ks 0 0 0 \n')
                fp.write(f'Tr 1 \n')
                fp.write(f'illum 1 \n')
                fp.write(f'Ns 0 \n')
                fp.write(f'map_Kd feat0_{cas}.jpg \n')
        
        v = (self.vertices + self.vertices_offsets).detach()
        f = self.triangles.detach()

        for cas in range(self.cascade):
            cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
            cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]
            _export_obj(cur_v, cur_f, h0, w0, self.opt.ssaa, cas)

            # half the texture resolution for remote area.
            if not self.opt.sdf and h0 > 2048 and w0 > 2048:
                h0 //= 2
                w0 //= 2

        # save mlp as json
        params = dict(self.specular_net.named_parameters())

        mlp = {}
        for k, p in params.items():
            p_np = p.detach().cpu().numpy().T
            print(f'[INFO] wrting MLP param {k}: {p_np.shape}')
            mlp[k] = p_np.tolist()

        mlp['bound'] = self.bound
        mlp['cascade'] = self.cascade

        mlp_file = os.path.join(path, f'mlp.json')
        with open(mlp_file, 'w') as fp:
            json.dump(mlp, fp, indent=2)

    @torch.no_grad()
    def refine_and_decimate(self):

        device = self.vertices.device
        # 获取当前顶点坐标（原始顶点 + 偏移量）和面数据
        v = (self.vertices + self.vertices_offsets).detach().cpu().numpy()
        f = self.triangles.detach().cpu().numpy()
        # 提取三角形误差及计数
        errors = self.triangles_errors.cpu().numpy()

        cnt = self.triangles_errors_cnt.cpu().numpy()
        cnt_mask = cnt > 0
        errors[cnt_mask] = errors[cnt_mask] / cnt[cnt_mask] # 计算平均误差（总误差 / 出现次数）

        # only care about the inner mesh # 仅处理内部网格（主物体）
        errors = errors[:self.f_cumsum[1]]
        cnt_mask = cnt_mask[:self.f_cumsum[1]]

        # find a threshold to decide whether we perform subdivision / decimation.
        thresh_refine = np.percentile(errors[cnt_mask], 90) # 被细分的面片数 取误差前10%作为细分阈值
        thresh_decimate = np.percentile(errors[cnt_mask], 50) # 被简化的面片数 取误差后50%作为简化阈值

        mask = np.zeros_like(errors)
        mask[(errors > thresh_refine) & cnt_mask] = 2 # 被细分的面片数 需要细分以提升局部精度
        mask[(errors < thresh_decimate) & cnt_mask] = 1 # 被简化的面片数 可简化以降低冗余

        print(f'[INFO] faces to decimate {(mask == 1).sum()}, faces to refine {(mask == 2).sum()}')
        # 单级联处理（主物体）​
        if self.bound <= 1:

            # mesh = trimesh.Trimesh(v, f, process=False)
            # mesh.export(os.path.join(self.opt.workspace, 'mesh_stage0', 'mesh_0_before_updated.ply'))
            # 简化和优化mesh
            # v, f = decimate_and_refine_mesh(v, f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)
            v, f = decimate_and_refine_mesh(v, f, mask,refine_size=1e-4)
            
            import os 
            mesh = trimesh.Trimesh(v, f, process=False)
            mesh_dir = os.path.join(self.opt.outputpath, "mesh")
            os.makedirs(mesh_dir,exist_ok=True)
            mesh.export(os.path.join(mesh_dir, f'mesh_updated.ply'))

            # fix counters # 更新顶点/面计数器
            self.v_cumsum[1:] += v.shape[0] - self.v_cumsum[1]
            self.f_cumsum[1:] += f.shape[0] - self.f_cumsum[1]
        else:
            
            vertices = []
            triangles = []
            v_cumsum = [0]
            f_cumsum = [0]

            for cas in range(self.cascade):

                cur_v = v[self.v_cumsum[cas]:self.v_cumsum[cas+1]]
                cur_f = f[self.f_cumsum[cas]:self.f_cumsum[cas+1]] - self.v_cumsum[cas]

                if cas == 0:
                    cur_v, cur_f = decimate_and_refine_mesh(cur_v, cur_f, mask, decimate_ratio=self.opt.refine_decimate_ratio, refine_size=self.opt.refine_size, refine_remesh_size=self.opt.refine_remesh_size)

                import os 
                mesh = trimesh.Trimesh(cur_v, cur_f, process=False)
                mesh_dir = os.path.join(self.opt.outputpath, "mesh", f'mesh_{cas}_updated.ply')
                os.makedirs(mesh_dir,exist_ok=True)
                mesh.export(mesh_dir)

                vertices.append(mesh.vertices)
                triangles.append(mesh.faces + v_cumsum[-1])

                v_cumsum.append(v_cumsum[-1] + mesh.vertices.shape[0])
                f_cumsum.append(f_cumsum[-1] + mesh.faces.shape[0])
            
            v = np.concatenate(vertices, axis=0)
            f = np.concatenate(triangles, axis=0)
            self.v_cumsum = np.array(v_cumsum)
            self.f_cumsum = np.array(f_cumsum)

        self.vertices = torch.from_numpy(v).float().contiguous().to(device) # [N, 3] # 更新顶点和面数据到GPU
        self.triangles = torch.from_numpy(f).int().contiguous().to(device)
        self.vertices_offsets = nn.Parameter(torch.zeros_like(self.vertices))
        # 重置顶点偏移量（重新开始优化）
        self.triangles_errors = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)
        self.triangles_errors_cnt = torch.zeros_like(self.triangles[:, 0], dtype=torch.float32)

        print(f'[INFO] update stage1 mesh: {self.vertices.shape}, {self.triangles.shape}')
