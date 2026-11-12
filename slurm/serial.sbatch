#!/bin/bash
# Gate 1 on one 40 GB A100: serial baselines (check 9) and, if asked, the cost
# model (check 8) and the greedy reference. Submit from the repo root:
#
#   sbatch --account=<project_id> --export=ALL,PB_N=256 slurm/serial.sbatch
#   sbatch --account=<project_id> --export=ALL,PB_N=64,PB_CALIBRATE=1,PB_GREEDY=1 slurm/serial.sbatch
#
# Results go to $PB_RUNS (default ~/parabeam/runs), one line per problem as it
# finishes. Resubmitting the same command resumes where a job stopped.
#
#SBATCH --job-name=pb-serial
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=a100:1
#SBATCH --constraint=a100-40gb
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail

cd "${SLURM_SUBMIT_DIR:-.}" || exit 1
# shellcheck source=configs/carc.env  # PB_ENV_FILE lets a dry run use another
source "${PB_ENV_FILE:-configs/carc.env}"
export PATH="$PB_ENV/bin:$PATH"
# Run this directory's code, also inside the vLLM servers that load the plugin.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
source slurm/launch_servers.sh

job="${SLURM_JOB_ID:-local}"
runs="${PB_RUNS:-$HOME/parabeam/runs}"
subset="${PB_SUBSET:-weak}"
mkdir -p logs "$runs"
echo "job $job on $(hostname) at $(date -u +%FT%TZ)"
cat SNAPSHOT.json 2>/dev/null || git rev-parse HEAD

trap pb_stop_servers EXIT
pb_start_servers "logs/pb-serial-$job" || exit 1

# A small search first, so a client or server problem fails in minutes.
python -m parabeam.eval.run serial --subset pilot --limit 2 --N 8 --M 4 \
    --out "$runs/smoke-$job" || { echo "FAIL  smoke search"; exit 1; }

status=0
if [[ "${PB_CALIBRATE:-0}" == 1 ]]; then
    python -m parabeam.tools.calibrate --out "$runs/calibrate-$job/cost_model.json" || status=1
fi
if [[ "${PB_GREEDY:-0}" == 1 ]]; then
    python -m parabeam.eval.run greedy --subset "$subset" --out "$runs/greedy-$subset" || status=1
fi
for n in ${PB_N:-256}; do
    python -m parabeam.eval.run serial --subset "$subset" --N "$n" --M 4 \
        --out "$runs/serial-$subset-n$n" || status=1
done
echo "finished at $(date -u +%FT%TZ) with status $status"
exit "$status"
