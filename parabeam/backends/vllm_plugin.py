"""vLLM plugin hook, registered as an entry point in pyproject.toml.

vLLM calls register() in every process it starts, so it must be cheap and safe
to repeat. The model class is named by string so vLLM imports it only when a
server asks for this architecture.
"""

PRM_LAST_ARCH = "ParabeamQwen2PRMLast"


def register() -> None:
    from vllm import ModelRegistry

    if PRM_LAST_ARCH not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            PRM_LAST_ARCH, "parabeam.backends.vllm_prm:Qwen2ForProcessRewardModelLast"
        )
