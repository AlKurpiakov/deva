#!/usr/bin/env bash
# Builds and installs the MultiScaleDeformableAttention CUDA extension into the
# active conda environment. Run it after `module load cuda` and
# `conda activate hush`: nvcc comes from the module, the host compiler (CC/CXX)
# from the conda environment. See INSTALL.md.
#
# Optional environment variables:
#   TORCH_CUDA_ARCH_LIST  GPU architectures to compile for, e.g. "8.0" or "8.0;9.0".
#                         If unset, the architectures of the visible GPUs are used.
#   FORCE_CUDA=1          build without a visible GPU, e.g. on a login node
#                         (TORCH_CUDA_ARCH_LIST is required then).
#   MAX_JOBS              number of parallel compile jobs (ninja).
set -euo pipefail

cd "$(dirname "$0")"

if [[ "${FORCE_CUDA:-0}" == "1" && -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
    echo "ERROR: FORCE_CUDA=1 requires TORCH_CUDA_ARCH_LIST, e.g. TORCH_CUDA_ARCH_LIST=\"8.0\"" >&2
    exit 1
fi

python - <<'PY'
import torch
from torch.utils.cpp_extension import CUDA_HOME
print(f"torch {torch.__version__} | built with CUDA {torch.version.cuda} | "
      f"GPU available: {torch.cuda.is_available()} | CUDA_HOME: {CUDA_HOME}")
PY

rm -rf build ./*.egg-info

# --no-build-isolation: setup.py imports torch, so the build must run in the
# current environment and not in an isolated one without torch.
python -m pip install --no-build-isolation -v .

# Sanity check, run from a neutral directory so the installed module is used.
(cd / && python -c "import torch, MultiScaleDeformableAttention as m; print('OK:', m.__file__)")
