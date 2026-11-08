#!/usr/bin/env python
"""Check the Python environment against the project's pins.

Run with no arguments on a login node. Run with --gpu inside a GPU job, which
also checks that torch can use the GPU.
"""
import argparse
import sys

# Keep in step with requirements-gpu.txt and configs/carc.env.
VLLM_VERSION = "0.32.0"
TORCH_CUDA = "12.9"
OPEN_MPI_VERSION = "5.0.5"
PRM_ARCH = "Qwen2ForProcessRewardModel"


def check_env(gpu: bool) -> list[str]:
    import torch
    import vllm
    import vllm.engine.arg_utils  # what `vllm serve` loads; `import vllm` alone is lazy
    from mpi4py import MPI
    from vllm import ModelRegistry
    from vllm.plugins import load_general_plugins

    from parabeam.backends.vllm_plugin import PRM_LAST_ARCH

    mpi_library = MPI.Get_library_version().splitlines()[0]
    print(f"python {sys.version.split()[0]}")
    print(f"vllm   {vllm.__version__}")
    print(f"torch  {torch.__version__} (CUDA {torch.version.cuda})")
    print(f"mpi    {mpi_library}")

    errors = []
    if not vllm.__version__.startswith(VLLM_VERSION):
        errors.append(f"expected vLLM {VLLM_VERSION}")
    if torch.version.cuda != TORCH_CUDA:
        errors.append(f"expected a torch build for CUDA {TORCH_CUDA}")
    if f"Open MPI v{OPEN_MPI_VERSION}" not in mpi_library:
        errors.append(f"mpi4py is not linked to Open MPI {OPEN_MPI_VERSION}")
    if PRM_ARCH not in ModelRegistry.get_supported_archs():
        errors.append(f"vLLM does not register {PRM_ARCH}")
    load_general_plugins()
    if PRM_LAST_ARCH not in ModelRegistry.get_supported_archs():
        errors.append(f"the parabeam vLLM plugin is missing; rerun pip install -e . ({PRM_LAST_ARCH})")

    if gpu:
        if torch.cuda.is_available():
            x = torch.ones(1024, 1024, device="cuda")
            print(f"gpu    {torch.cuda.get_device_name(0)}, matmul sum {(x @ x).sum().item():.0f}")
        else:
            errors.append(f"torch cannot use the GPU; driver may be too old for CUDA {TORCH_CUDA}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", action="store_true", help="also run a GPU computation")
    errors = check_env(parser.parse_args().gpu)
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
