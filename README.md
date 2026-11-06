# parabeam

Scaling PRM-guided beam search across GPUs with asynchronous work stealing. EE/CSCI 451 course project, USC, Fall 2026.

- Generator: Qwen2.5-Math-1.5B-Instruct
- PRM: Qwen2.5-Math-PRM-7B
- Runtime: vLLM servers on each GPU, MPI ranks via mpi4py, USC CARC Discovery A100s

## Layout

- parabeam/core: beams, seeding, selection, voting, serial search
- parabeam/backends: backend interface, mock backend, vLLM client
- parabeam/dist: MPI runtime, level-sync and async schedulers, placement, stealing
- parabeam/eval: datasets, grading, metrics
- parabeam/tools: calibration, GPU budget, plots
- configs, slurm, scripts, tests

## Setup on CARC Discovery

Run these on a login node, from the repo root. Connecting from outside USC needs the VPN.

```bash
bash scripts/setup_env.sh                                            # Python environment on /scratch1
nohup bash scripts/download_models.sh > logs/download_models.log 2>&1 &  # about 25 GB; rerun to resume
myaccount                                                            # shows your project ID
sbatch --account=<project_id> slurm/preflight.sbatch                 # checks one 40 GB A100
```

`logs/pb-preflight-<job_id>.out` ends in `PREFLIGHT PASSED` or `PREFLIGHT FAILED`. `noderes -c -g` and `noderes -f -g` show the configured and free GPUs.

The environment, models and caches live in `/scratch1/$USER/parabeam` (see `configs/carc.env`). Scratch has no backups and can be purged, but both scripts rebuild it. Versions are pinned in `requirements-gpu.txt`, `configs/models.tsv` and the generated `configs/requirements-carc.lock`.
