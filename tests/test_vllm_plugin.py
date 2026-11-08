"""The vLLM plugin; runs only where vLLM is installed, as on CARC."""
import pytest

vllm = pytest.importorskip("vllm")


def test_the_plugin_registers_the_last_pooling_prm():
    from vllm import ModelRegistry

    from parabeam.backends.vllm_plugin import PRM_LAST_ARCH, register

    register()
    register()  # vLLM calls it once per process
    assert PRM_LAST_ARCH in ModelRegistry.get_supported_archs()


def test_vllm_finds_the_plugin_through_its_entry_point():
    from importlib.metadata import entry_points

    names = {ep.name: ep.value for ep in entry_points(group="vllm.general_plugins")}
    assert names.get("parabeam_prm") == "parabeam.backends.vllm_plugin:register"


def test_the_model_class_pools_the_last_token():
    from parabeam.backends.vllm_prm import Qwen2ForProcessRewardModelLast

    assert Qwen2ForProcessRewardModelLast.default_seq_pooling_type == "LAST"
    assert Qwen2ForProcessRewardModelLast.default_tok_pooling_type != "STEP"
