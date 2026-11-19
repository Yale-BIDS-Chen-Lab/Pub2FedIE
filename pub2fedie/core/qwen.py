"""Qwen3.5 compatibility for the study's deterministic PyTorch configuration."""
import torch
from torch.overrides import TorchFunctionMode
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5GatedDeltaNet, torch_chunk_gated_delta_rule


class _ChunkPrefixSum(TorchFunctionMode):
    def __torch_function__(self, func, types, args=(), kwargs=None):
        # The pinned Transformers chunk rule calls g.cumsum(dim=-1) on FP32 gates.
        if func is torch.Tensor.cumsum and kwargs == {"dim": -1} and args[0].dtype == torch.float32:
            g = args[0]
            width = g.shape[-1]
            return g @ g.new_ones((width, width)).triu()
        return func(*args, **(kwargs or {}))


def _chunk_rule(*args, **kwargs):
    with _ChunkPrefixSum():
        return torch_chunk_gated_delta_rule(*args, **kwargs)


def install_qwen35_deterministic_prefix_sum(model):
    layers = [m for m in model.modules() if isinstance(m, Qwen3_5GatedDeltaNet)]
    if not layers:
        raise ValueError("Expected Qwen3.5 GatedDeltaNet layers")
    for layer in layers:
        layer.chunk_gated_delta_rule = _chunk_rule
