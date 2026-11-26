"""Weight initialisation shared by the trunk and the multi-token-prediction blocks."""

from __future__ import annotations

from torch import nn

from hale_llm.nn.blocks.feed_forward import FeedForward
from hale_llm.nn.blocks.transformer_block import TransformerBlock


def init_linear_and_embedding(root: nn.Module, std: float = 0.02) -> None:
    """GPT-2 style: every Linear and Embedding starts at N(0, std), biases at 0."""
    for module in root.modules():
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)


def scale_residual_outputs(block: TransformerBlock, std: float) -> None:
    """Shrink the layers that write into the residual stream (attention output, feed-forward output).

    Each block adds two such terms, so without this the stream grows with depth.
    """
    nn.init.normal_(block.attn.proj.weight, std=std)
    ffn = block.ffn
    if isinstance(ffn, FeedForward):
        nn.init.normal_(ffn.down.weight, std=std)
    else:  # mixture of experts: every expert's output matrix
        for expert in [*ffn.shared_experts, *ffn.routed_experts]:
            nn.init.normal_(expert.w2.weight, std=std)
