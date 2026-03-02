"""Patch transformers for qwen35moe GGUF architecture."""
from transformers.integrations import ggml
for a in dir(ggml):
    o = getattr(ggml, a)
    if isinstance(o, dict):
        for s in ('qwen3_5_moe', 'qwen3_moe'):
            if s in o and 'qwen35moe' not in o:
                o['qwen35moe'] = o[s]
                break
from transformers.models.auto import configuration_auto as ca
from transformers.models.qwen3_5_moe import configuration_qwen3_5_moe as c
ca.CONFIG_MAPPING._extra_content['qwen35moe'] = c.Qwen3_5MoeConfig
