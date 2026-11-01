#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
version="${1:-HybridRepre3}"
case "$version" in HybridRepre|HybridRepre2|HybridRepre3) ;; *) echo 'Choose HybridRepre, HybridRepre2 or HybridRepre3' >&2; exit 2;; esac
source "${project_root}/scripts/env.sh" "${HYBRIDREPRE_WORKDIR:-/data/hybridrepre}"
python -m pip install "${project_root}/third_party/fvcore" "${project_root}/third_party/iopath"
python -m pip install --no-build-isolation "${project_root}/third_party/nvdiffrast" "${project_root}/third_party/pytorch3d"
python -m pip install --no-build-isolation "${project_root}/third_party/tiny-cuda-nn/bindings/torch"
python -m pip install --no-build-isolation "${project_root}/${version}/submodules/diff-gaussian-rasterization" "${project_root}/${version}/submodules/simple-knn"
