# Third-party code

Each bundled component retains its own copyright notices and license. This repository does not replace those licenses with a blanket license.

| Component | Upstream | Local source |
| --- | --- | --- |
| nvdiffrec | https://github.com/NVlabs/nvdiffrec | Mesh rendering/material code in each version; license in `licenses/` |
| 3D Gaussian Splatting | https://github.com/graphdeco-inria/gaussian-splatting | `HybridRepre*/gaussian_splatting/`, Gaussian scene/rendering modules; `LICENSE-3DGS.md` |
| Gaussian rasterizer / simple-knn | https://github.com/graphdeco-inria/diff-gaussian-rasterization | `HybridRepre*/submodules/` and baseline submodules; component licenses retained |
| nvdiffrast | https://github.com/NVlabs/nvdiffrast | `third_party/nvdiffrast/`, revision `729261dc64c4241ea36efda84fbf532cc8b425b8` |
| PyTorch3D | https://github.com/facebookresearch/pytorch3d | `third_party/pytorch3d/`, revision `06a76ef8ddd00b6c889768dfc990ae8cb07c6f2f` |
| tiny-cuda-nn | https://github.com/NVlabs/tiny-cuda-nn | `third_party/tiny-cuda-nn/`, revision `075158a70b87dba8729188a9cadc9411cfa4b71d` |
| fvcore / iopath | https://github.com/facebookresearch/fvcore , https://github.com/facebookresearch/iopath | `third_party/fvcore/`, `third_party/iopath/` |
| torch-ngp components | https://github.com/ashawkey/torch-ngp | Neural encoders and utilities in `HybridRepre2/`; license in `licenses/` |
| Frosting | https://github.com/Anttwo/Frosting | Historical alternate renderers in `HybridRepre*/rendering_scene/`; upstream license in `licenses/` |

Build products and pretrained weights are excluded. Third-party components are source snapshots; custom rasterization changes are kept with their corresponding experiment version.
