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

if [[ ! -x "$PB_ENV/bin/python" ]]; then
    pb_modules conda
    mamba create --yes --prefix "$PB_ENV" "python=$PB_PYTHON_VERSION" pip
fi

# shellcheck disable=SC2086  # the module list is meant to split
pb_modules $PB_MPI_MODULES
MPICC="$(command -v mpicc)"
export MPICC

"$PB_ENV/bin/python" -m pip install \
    -r "$repo/requirements.txt" \
    -r "$repo/requirements-gpu.txt" \
    -e "$repo"

# Prebuilt kernels only save compile time at server startup, so a failure here
# (for example no GPU on the login node) does not stop the setup.
"$PB_ENV/bin/vllm" download-kernels || echo "WARNING: vllm download-kernels failed; continuing" >&2

"$PB_ENV/bin/python" "$repo/scripts/check_env.py"
"$PB_ENV/bin/python" -m pip freeze --exclude-editable > "$repo/configs/requirements-carc.lock"

echo "Environment ready at $PB_ENV"
echo "Lock file written to configs/requirements-carc.lock; commit it."
