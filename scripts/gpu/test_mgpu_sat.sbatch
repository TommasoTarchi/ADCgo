#!/bin/bash
# Multi-GPU parity for the per-device DIP satellite apply (TestSatelliteMatFreePerDeviceParity).
#
# Gates the -mgpu fast path: each GPU recomputing only its own output row band, reading a
# full-height slab gathered over NVLink, must reproduce BOTH the dense operator and the host
# gather-apply-scatter path it replaces — bit-exact (<=1e-10).
#
# Needs >= 4 devices (at 2 the partition-resolution and row-band clamping are trivial) AND peer
# access between every pair, else the test skips. The workload is h2o-sized, so any GPU model
# will do; the interconnect is what matters, hence the topology dump below.
#
#SBATCH --account=bw22i008
#SBATCH --job-name=mgpu-sat-test
#SBATCH --partition=gpu-single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --mem=64gb
#SBATCH --time=00:30:00
#SBATCH --output=mgpu-sat-test-%j.out
set -euo pipefail

self="$0"
if [ -n "${SLURM_JOB_ID:-}" ]; then
    self="$(scontrol show job "$SLURM_JOB_ID" | sed -n 's/^[[:space:]]*Command=//p' | head -1 | awk '{print $1}')"
fi
root="$(cd "$(dirname "$self")/../.." && pwd)"
cd "$root"

if command -v module >/dev/null 2>&1; then
    module load compiler/gnu/11.3 devel/cuda/13.2 || true
fi
: "${CUDA_HOME:=${CUDA_INSTALL_PATH:-${CUDA_ROOT:-/usr/local/cuda}}}"
export PATH="/home/hd/hd_hd/hd_hh323/opt/go/bin:$CUDA_HOME/bin:$PATH"
export CGO_CFLAGS="-I$CUDA_HOME/include ${CGO_CFLAGS:-}"
export CGO_LDFLAGS="-L$CUDA_HOME/lib64 ${CGO_LDFLAGS:-}"

echo "host=$(hostname)  go=$(go version)"
nvidia-smi -L || true
echo "--- interconnect (peer path must be NVLink/NVSwitch, not PHB, for the gather to be fast) ---"
nvidia-smi topo -m 2>/dev/null || true

echo "#begin<nvcc>"
# Kernel objects and the workspace file a cuda build links against: scripts/gpu/cuda_kernels.sh.
. "$root/scripts/gpu/cuda_kernels.sh" && adcgo_cuda_kernels || exit 1
echo "#end<nvcc>"

# -v so a SKIP (too few devices / not peered) is visible rather than passing silently.
echo "#begin<gotest>"
go test -tags cuda -v -run 'TestSatelliteMatFreePerDeviceParity|TestSatBatchedPerDeviceParity' ./internal/adc/dip/
echo "#end<gotest>"
