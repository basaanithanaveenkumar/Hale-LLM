"""Context-extension stages: validation, step boundaries, batch sizing, long-document mixing."""

import pytest
import torch

from hale_llm.config.schema import RunConfig
from hale_llm.dataloader.long_document_mix import mix_long_documents
from hale_llm.training.context_schedule import ContextSchedule

EOS = 9


def _cfg(**train):
    return RunConfig(
        variant="autoregressive",
        data={"context_window": 4096},
        model={"rope_theta": 10_000.0},
        train={
            "batch_size": 32,
            "context_stages": [{"name": "extend", "start_fraction": 0.95, "context_window": 32768, "rope_theta": 1_000_000.0}],
            **train,
        },
    )


def test_plans_follow_the_guidance_numbers():
    schedule = ContextSchedule.from_config(_cfg(), total_steps=1000)
    base, extend = schedule.plans
    assert (base.start_step, base.context_window, base.rope_theta, base.batch_size) == (0, 4096, 10_000.0, 32)
    # Tokens per step stay constant: 32 x 4096 = 4 x 32768.
    assert (extend.start_step, extend.context_window, extend.batch_size) == (950, 32768, 4)
    assert extend.long_doc_fraction == 0.5


def test_plan_at_switches_exactly_at_the_boundary():
    schedule = ContextSchedule.from_config(_cfg(), total_steps=1000)
    assert schedule.plan_at(0).name == "pretrain"
    assert schedule.plan_at(949).name == "pretrain"
    assert schedule.plan_at(950).name == "extend"
    assert schedule.plan_at(999).name == "extend"


def test_no_stages_means_one_plan():
    cfg = RunConfig(variant="autoregressive", data={"context_window": 128})
    assert len(ContextSchedule.from_config(cfg, 100).plans) == 1


def test_stages_need_rope_positions():
    cfg = _cfg()
    cfg.model.pos_encoding = "learned"
    with pytest.raises(ValueError, match="rope"):
        ContextSchedule.from_config(cfg, 100)


def test_stage_must_grow_and_be_ordered():
    shorter = _cfg()
    shorter.train.context_stages[0].context_window = 2048
    with pytest.raises(ValueError, match="must exceed"):
        ContextSchedule.from_config(shorter, 1000)
    lower_theta = _cfg()
    lower_theta.train.context_stages[0].rope_theta = 100.0
    with pytest.raises(ValueError, match="rope_theta"):
        ContextSchedule.from_config(lower_theta, 1000)
    too_early = _cfg()
    too_early.train.context_stages[0].start_fraction = 0.0001
    with pytest.raises(ValueError, match="not after"):
        ContextSchedule.from_config(too_early, 1000)


# ----- long-document mixing -----

def _docs():
    short = [[1] * 5 for _ in range(60)]  # 60 x 6 tokens
    long = [[2] * 40 for _ in range(4)]  # 4 x 41 tokens
    return short + long


def test_mix_keeps_the_window_count_and_adds_long_windows():
    plain = mix_long_documents(_docs(), EOS, 16, long_fraction=0.0, min_long_tokens=30)
    mixed = mix_long_documents(_docs(), EOS, 16, long_fraction=0.2, min_long_tokens=30)
    assert mixed.shape == plain.shape
    long_rows = (mixed == 2).sum(dim=1) >= 15  # windows made of nearly only long-document tokens
    assert long_rows.sum().item() >= int(0.2 * len(plain)) - 1


def test_mix_uses_all_long_windows_when_supply_is_short():
    mixed = mix_long_documents(_docs(), EOS, 16, long_fraction=0.9, min_long_tokens=30)
    assert mixed.shape[0] == mix_long_documents(_docs(), EOS, 16, 0.0, 30).shape[0]


def test_mix_without_long_documents_is_the_plain_packing():
    docs = [[1] * 5 for _ in range(40)]
    assert torch.equal(
        mix_long_documents(docs, EOS, 16, 0.5, min_long_tokens=100),
        mix_long_documents(docs, EOS, 16, 0.0, min_long_tokens=100),
    )


def test_mix_rejects_a_fraction_of_one():
    with pytest.raises(ValueError):
        mix_long_documents(_docs(), EOS, 16, 1.0, 30)
