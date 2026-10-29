#!/bin/bash
# =============================================================================
# HUSH training on Stanford2D3D with SLURM.
#
# Submit from the repository root:
#     sbatch train_sf2d3d.sbatch
# Settings can be overridden without editing the file, e.g.:
#     DATA_ROOT=/scratch/$USER/SF2D3D/ CUDA_MODULE=cuda/12.4 sbatch train_sf2d3d.sbatch
#
# What the job does:
#   1. loads the cuda module and activates the conda environment;
#   2. builds the MultiScaleDeformableAttention op on the first run;
#   3. checks that the Swin-L weights are available (compute nodes often have
#      no internet access, see INSTALL.md);
#   4. trains; the full state is saved to $CKPT_DIR/last.pth after every epoch;
#   5. 10 minutes before the time limit SLURM sends SIGUSR1, the job stops
#      training and requeues itself; the new run continues from the last
#      finished epoch.
# A job that was cancelled or crashed is continued by submitting it again.
# =============================================================================
#SBATCH --job-name=hush-sf2d3d
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
##SBATCH --partition=gpu
##SBATCH --account=my_account

set -o pipefail

# ------------------------------- settings -----------------------------------
CUDA_MODULE="${CUDA_MODULE:-cuda}"           # see "module avail cuda"; major version must match torch.version.cuda
CONDA_DIR="${CONDA_DIR:-$HOME/miniforge3}"   # where Miniforge is installed
CONDA_ENV="${CONDA_ENV:-hush}"               # conda env name or full path

DATA_ROOT="${DATA_ROOT:-/path/to/SF2D3D/}"
TRAIN_LIST="${TRAIN_LIST:-./filenames/train_stanford2d3d.txt}"
VAL_LIST="${VAL_LIST:-./filenames/test_stanford2d3d.txt}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
CKPT_DIR="${CKPT_DIR:-./checkpoints/sf2d3d}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                 # e.g. "--keep_all"
MAX_RESTARTS="${MAX_RESTARTS:-30}"           # protection against endless requeueing
# TORCH_CUDA_ARCH_LIST (optional): GPU architectures the op is built for,
# e.g. "8.0;9.0" for a cluster with A100 and H100 nodes. By default the op is
# built for the GPU of the node the first job runs on.
# ----------------------------------------------------------------------------

die() { echo "ERROR: $*"; exit 1; }

cd "${SLURM_SUBMIT_DIR:-$PWD}" || die "cannot cd to the submit directory"
[[ -f train_sf2d3d.py ]] || die "$PWD is not the HUSH repository; submit the job from the repository root"

echo "================================================================"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), restart #${SLURM_RESTART_COUNT:-0}, $(date)"
echo "================================================================"

# --- cuda module ------------------------------------------------------------
if ! type module >/dev/null 2>&1; then
    for f in /etc/profile.d/modules.sh /etc/profile.d/lmod.sh /usr/share/lmod/lmod/init/bash; do
        # shellcheck disable=SC1090
        [[ -f "$f" ]] && source "$f" && break
    done
fi
module load "$CUDA_MODULE" || die "module load $CUDA_MODULE failed"
NVCC="$(command -v nvcc)" || die "nvcc not found after 'module load $CUDA_MODULE'"
export CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$NVCC")")}"

# --- conda environment (provides python, PyTorch and gcc for the op) -------
if [[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$CONDA_DIR/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
    eval "$(conda shell.bash hook)"
else
    die "conda not found; set CONDA_DIR to your Miniforge directory"
fi
conda activate "$CONDA_ENV" || die "conda activate $CONDA_ENV failed"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "CUDA_HOME=$CUDA_HOME"; nvcc --version | tail -n 1; "${CXX:-g++}" --version | head -n 1
python - <<'PY' || exit 1
import subprocess, sys, torch
from torch.utils.cpp_extension import CUDA_HOME
print(f"python {sys.version.split()[0]} | torch {torch.__version__} | torch CUDA {torch.version.cuda}")
if not torch.cuda.is_available():
    sys.exit("ERROR: torch.cuda.is_available() is False on this node")
out = subprocess.check_output([f"{CUDA_HOME}/bin/nvcc", "--version"]).decode()
nvcc_major = out.split("release ")[1].split(".")[0]
if nvcc_major != torch.version.cuda.split(".")[0]:
    sys.exit(f"ERROR: nvcc is CUDA {nvcc_major}.x, but torch is built with CUDA {torch.version.cuda}. "
             "Load a matching module (CUDA_MODULE=...) or recreate the environment "
             "for this CUDA version (see INSTALL.md).")
print("GPU:", torch.cuda.get_device_name(0))
PY

# --- MultiScaleDeformableAttention op ---------------------------------------
if ! python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
    echo "MultiScaleDeformableAttention is not installed, building it"
    # The lock prevents several jobs from building into the same env at once.
    (
        command -v flock >/dev/null && flock -x 9
        python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null || bash models/ops/make.sh
    ) 9>"models/ops/.build.lock" || die "building the op failed"
fi
# The op must have kernels for this GPU, otherwise training fails much later.
python - <<'PY' || die "the op does not run on this GPU; rebuild it with a suitable TORCH_CUDA_ARCH_LIST (see INSTALL.md)"
import torch
from models.ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch
shapes = torch.as_tensor([(6, 4), (3, 2)], dtype=torch.long).cuda()
start = torch.cat((shapes.new_zeros((1,)), shapes.prod(1).cumsum(0)[:-1]))
S = int(shapes.prod(1).sum())
v = torch.rand(2, S, 2, 8).cuda()
loc = torch.rand(2, 5, 2, 2, 2, 2).cuda()
w = torch.rand(2, 5, 2, 2, 2).cuda()
w = w / w.sum(-1, keepdim=True).sum(-2, keepdim=True)
ok = torch.allclose(MSDeformAttnFunction.apply(v, shapes, start, loc, w, 2),
                    ms_deform_attn_core_pytorch(v, shapes, loc, w), rtol=1e-2, atol=1e-3)
print("MultiScaleDeformableAttention check:", "OK" if ok else "MISMATCH")
raise SystemExit(0 if ok else 1)
PY

# --- pretrained Swin-L weights ----------------------------------------------
SWIN_URL="https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth"
python - "$SWIN_URL" <<'PY' || exit 1
import os, sys, torch
url = sys.argv[1]
path = os.path.join(torch.hub.get_dir(), "checkpoints", os.path.basename(url))
if os.path.isfile(path):
    print("Swin-L weights:", path)
    raise SystemExit(0)
print("Swin-L weights not found, downloading to", path)
try:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.hub.download_url_to_file(url, path)
except Exception as e:
    sys.exit(f"ERROR: download failed ({e}). Compute nodes may have no internet: download the file "
             f"on a login node into {os.path.dirname(path)} (see INSTALL.md) and resubmit.")
PY

# --- training ---------------------------------------------------------------
[[ -d "$DATA_ROOT" ]] || die "DATA_ROOT=$DATA_ROOT does not exist"
mkdir -p "$CKPT_DIR"

# shellcheck disable=SC2086
python -u train_sf2d3d.py \
    --rootpath "$DATA_ROOT" \
    --trainfile "$TRAIN_LIST" \
    --valfile "$VAL_LIST" \
    --epoch "$EPOCHS" \
    --bs "$BATCH_SIZE" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" \
    --ckpt_dir "$CKPT_DIR" \
    --resume auto \
    $EXTRA_ARGS &
TRAIN_PID=$!

on_time_limit() {
    echo "[$(date)] Time limit is close: stopping training and requeueing the job."
    kill -TERM "$TRAIN_PID" 2>/dev/null
    wait "$TRAIN_PID"
    if (( ${SLURM_RESTART_COUNT:-0} < MAX_RESTARTS )); then
        scontrol requeue "${SLURM_JOB_ID}" \
            && echo "Requeued, training will continue from $CKPT_DIR/last.pth" \
            || echo "WARNING: requeue failed; submit the script again to continue."
    else
        echo "MAX_RESTARTS=$MAX_RESTARTS reached; submit the script again to continue."
    fi
    exit 0
}
trap on_time_limit USR1

wait "$TRAIN_PID"
RC=$?
echo "[$(date)] Training process exited with code $RC"
exit $RC
