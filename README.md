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
