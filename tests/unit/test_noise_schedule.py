"""The extracted diffusion schedule and flow path must equal the expressions they replaced."""

from __future__ import annotations

import torch

from hale_llm.models.models_diffusion import LogLinearSchedule, NoiseSchedule
from hale_llm.models.models_flow_matching.vector_field import mask_prob


def test_log_linear_schedule_is_bit_identical_to_the_inline_math():
    sched = LogLinearSchedule()
    assert isinstance(sched, NoiseSchedule)
    t = torch.rand(64) * 0.999 + 1e-3
    t_next = t * torch.rand(64)
    eps = 1e-3
    assert torch.equal(sched.mask_probability(t), t)
    assert torch.equal(sched.alpha(t), 1.0 - t)
    assert torch.equal(sched.loss_weight(t, eps), 1.0 / t.clamp(min=eps))
    expected = ((t - t_next) / t.clamp(min=1e-8)).clamp(0, 1)
    assert torch.equal(sched.reveal_probability(t, t_next), expected)


def test_flow_matching_path_is_linear():
    t = torch.rand(16)
    assert torch.equal(mask_prob(t), 1.0 - t) and mask_prob(0.25) == 0.75
