#!/usr/bin/env bash
# Build the project's Python environment on /scratch1. Run on a Discovery login
# node. Safe to rerun; expect about 15 GB and 10 to 20 minutes the first time.
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$repo/configs/carc.env"

if ! command -v sbatch >/dev/null; then
    echo "ERROR: run this on a CARC login node" >&2
    exit 1
fi

mkdir -p "$repo/logs" "$PIP_CACHE_DIR" "$CONDA_PKGS_DIRS" "$PB_SCRATCH/tmp"
export TMPDIR="$PB_SCRATCH/tmp"
pb_login_limits

if [[ ! -x "$PB_ENV/bin/python" ]]; then
    rm -rf "$PB_ENV"  # leftovers from an interrupted create
    pb_modules conda
    # shellcheck disable=SC2086  # the package list is meant to split
    mamba create --yes --prefix "$PB_ENV" $PB_CONDA_PACKAGES
fi

# shellcheck disable=SC2086  # the module list is meant to split
pb_modules $PB_MPI_MODULES
MPICC="$(command -v mpicc)"
export MPICC

"$PB_ENV/bin/python" -m pip install \
    -r "$repo/requirements.txt" \
    -r "$repo/requirements-gpu.txt" \
    -e "$repo"

# PyPI's torchcodec needs CUDA 13 and the newest cu129 build is too old for this
# torch. vLLM uses it only for audio and runs without it, but with it installed
# every vLLM command dies at import, because the load error is not an ImportError.
"$PB_ENV/bin/python" -m pip uninstall --yes torchcodec

# Every vLLM command, even download-kernels, needs a GPU, so slurm/smoke.sbatch
# installs the prebuilt kernels instead.

"$PB_ENV/bin/python" "$repo/scripts/check_env.py"
"$PB_ENV/bin/python" -m pip freeze --exclude-editable > "$repo/configs/requirements-carc.lock"

echo "Environment ready at $PB_ENV"
echo "Lock file written to configs/requirements-carc.lock; commit it."
echo "Next: sbatch --account=<project_id> slurm/smoke.sbatch, which installs vLLM's kernels."
