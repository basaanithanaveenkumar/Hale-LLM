"""[MASK] is added only for the variants that corrupt text with it."""

import pytest

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import RunConfig
from hale_llm.core.registry import get_variant
from hale_llm.dataloader.tokenizer import build_tokenizer_for

GPT2_VOCAB = 50_257


def test_autoregressive_keeps_the_plain_gpt2_vocabulary():
    tok = build_tokenizer_for(RunConfig(variant="autoregressive"))
    assert len(tok) == GPT2_VOCAB
    assert tok.mask_token is None


@pytest.mark.parametrize("variant", ["diffusion", "block_diffusion", "flow_matching"])
def test_diffusion_variants_get_one_extra_mask_token(variant):
    tok = build_tokenizer_for(RunConfig(variant=variant))
    assert len(tok) == GPT2_VOCAB + 1
    assert tok.mask_token == "[MASK]"


def test_only_masking_variants_declare_the_flag():
    flagged = {
        v for v in ("autoregressive", "diffusion", "block_diffusion", "flow_matching", "mtp")
        if getattr(get_variant(v), "uses_mask_token", False)
    }
    assert flagged == {"diffusion", "block_diffusion", "flow_matching"}


def test_tokenizer_is_loaded_once_per_configuration():
    cfg = RunConfig(variant="autoregressive")
    assert build_tokenizer_for(cfg) is build_tokenizer_for(cfg)
