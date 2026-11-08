"""Qwen2.5-Math-PRM served so that vLLM can use its prefix cache.

The stock class answers token_classify requests, and vLLM 0.32 never reads the
prefix cache for those (pooling_params.py), so every level would recompute every
beam's whole prefix. This class keeps the same network and weights but pools
the last token for classify requests. The last token of a beam's PRM input is
its newest step's <extra_0>, so the result is exactly the reward the search
needs, and vLLM reads the cache for it.

Select it with --hf-overrides '{"architectures": ["ParabeamQwen2PRMLast"]}'.
"""
from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.seqwise import pooler_for_classify
from vllm.model_executor.models.interfaces_base import default_pooling_type
from vllm.model_executor.models.qwen2_rm import Qwen2ForProcessRewardModel


@default_pooling_type(seq_pooling_type="LAST", tok_pooling_type="ALL")
class Qwen2ForProcessRewardModelLast(Qwen2ForProcessRewardModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        # forward() already applies the score head, so the pooler only picks the
        # last token's two logits and applies softmax.
        self.pooler = pooler_for_classify(vllm_config.model_config.pooler_config)
