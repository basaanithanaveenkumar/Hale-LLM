"""Unit tests for hale_llm.rl (online GRPO) and the RL films."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from hale_llm.rl import (
    ArithmeticTask,
    ExactMatchReward,
    GroupRelativeAdvantage,
    GRPOObjective,
    MajorityVoteReward,
    OnlineRLConfig,
    OnlineRLError,
    OnlineRLExperiment,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs" / "rl"


def test_group_advantage_centres_each_group_and_zeroes_flat_groups():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0])
    adv = GroupRelativeAdvantage()(rewards, 4)
    assert adv[:4].sum().abs() < 1e-5 and adv[0] > 0 > adv[1]
    assert torch.all(adv[4:] == 0)
    assert GroupRelativeAdvantage.zero_variance_fraction(rewards, 4) == 0.5
    with pytest.raises(OnlineRLError):
        GroupRelativeAdvantage()(rewards, 3)


def test_grpo_objective_clips_and_kl_is_zero_at_reference():
    lp = torch.log(torch.tensor([[1.5, 0.5]]))  # ratios 1.5 and 0.5 vs old = 0
    terms = GRPOObjective(clip_eps=0.2, kl_coef=0.1)(
        lp, torch.zeros_like(lp), torch.tensor([1.0]), ref_logp=lp.clone()
    )
    assert terms.clip_fraction == 1.0 and terms.kl == pytest.approx(0.0)
    # positive advantage: gain capped at 1.2 for the first token, unclipped 0.5 for the second
    assert terms.loss.item() == pytest.approx(-(1.2 + 0.5) / 2)


def test_rewards_exact_match_and_majority_vote():
    prompts, comps = ["1+1="], ["02", "03", "03", "02 "]
    assert ExactMatchReward({"1+1=": "02"})(prompts, comps, 4).tolist() == [1, 0, 0, 1]
    assert MajorityVoteReward()(prompts, ["03", "03", "02", "05"], 4).tolist() == [1, 1, 0, 0]
    assert MajorityVoteReward.pseudo_labels(["03", "03", "02", "05"], 4) == ["03"]
    with pytest.raises(OnlineRLError):
        ExactMatchReward({"1+1=": "02"})(prompts, comps, 3)


def test_arithmetic_task_split():
    task = ArithmeticTask(max_operand=9, sft_fraction=0.4, seed=0)
    assert len(task.prompts) == 100 and len(task.seen_prompts) == 40
    assert set(task.seen_prompts).isdisjoint(task.unseen_prompts)
    assert all(len(t) == 6 for t in task.sft_texts) and task.answers["8+4="] == "12"


def test_config_inheritance_overrides_and_cycle(tmp_path):
    cfg = OnlineRLConfig.from_yaml(CONFIGS / "arithmetic_self_reward.yaml")
    assert cfg.reward.type == "majority_vote" and cfg.variant == "autoregressive"
    assert cfg.train.steps == 600 and cfg.rollout.group_size == 8  # inherited from base
    assert cfg.with_overrides({"rollout": {"temperature": 1.0}}).rollout.temperature == 1.0
    (tmp_path / "a.yaml").write_text("inherits: b.yaml\n")
    (tmp_path / "b.yaml").write_text("inherits: a.yaml\n")
    with pytest.raises(OnlineRLError, match="cycle"):
        OnlineRLConfig.from_yaml(tmp_path / "a.yaml")
    with pytest.raises(ValueError):
        OnlineRLConfig.model_validate({"variant": "mtp"})


@pytest.mark.parametrize("variant", ["autoregressive", "diffusion"])
def test_log_prob_shapes_and_answer_distribution(variant):
    exp = OnlineRLExperiment(
        OnlineRLConfig.model_validate(
            {
                "variant": variant,
                "model": {"d_model": 32, "n_layers": 1, "d_ff": 64, "n_heads": 2},
                "sft": {"steps": 2},
            }
        )
    )
    policy = exp.warm_start()
    est = exp.log_prob()
    ids = torch.tensor([exp.tokenizer.encode("1+2=")] * 3)
    comp = torch.tensor([exp.tokenizer.encode("03")] * 3)
    lp = est(policy, ids, comp, stochastic=False)
    assert lp.shape == (3, 2) and torch.all(lp <= 0)
    dist = est.answer_distribution(policy, "1+2=", exp.task.candidates)
    assert dist.shape == (100,) and dist.sum().item() == pytest.approx(1.0, abs=1e-4)


def test_grpo_gradient_equals_reinforce_at_ratio_one():
    exp = OnlineRLExperiment(
        OnlineRLConfig.model_validate(
            {"model": {"d_model": 32, "n_layers": 1, "d_ff": 64, "n_heads": 2}, "sft": {"steps": 2}}
        )
    )
    policy = exp.warm_start()
    batch = exp.trainer(copy.deepcopy(policy)).collect(exp.task.prompts[:2])
    batch.advantages = torch.randn_like(batch.advantages)
    lp = exp.log_prob()(policy, batch.prompt_ids, batch.completion_ids)
    loss = GRPOObjective(kl_coef=0.0)(lp, lp.detach(), batch.advantages).loss
    g1 = torch.autograd.grad(loss, list(policy.parameters()), retain_graph=True)
    g2 = torch.autograd.grad(-(batch.advantages[:, None] * lp).mean(), list(policy.parameters()))
    assert max((a - b).abs().max().item() for a, b in zip(g1, g2)) < 1e-6


def test_short_run_records_history_and_improves_reward():
    exp = OnlineRLExperiment.from_yaml(
        CONFIGS / "arithmetic_autoregressive.yaml",
        overrides={"train": {"eval_every": 50, "track_prompts": 1}},
    )
    history = exp.run(exp.warm_start(), steps=150)
    assert len(history.stats) == 150 and [e["step"] for e in history.evals][:2] == [0, 50]
    assert (
        set(history.groups) >= {0, 50}
        and len(history.distributions[0][history.tracked_prompts[0]]) == 100
    )
    early, late = history.series("reward_mean")[:30], history.series("reward_mean")[-30:]
    assert sum(late) / 30 > sum(early) / 30


def test_unknown_reward_raises():
    exp = OnlineRLExperiment(OnlineRLConfig.model_validate({"reward": {"type": "nope"}}))
    with pytest.raises(OnlineRLError, match="unknown reward"):
        exp.reward()


def test_films_render(tmp_path):
    from PIL import Image

    from hale_llm.visualization.rl_films import GRPOStepFilm, PolicyEvolutionFilm

    path = GRPOStepFilm("1+1=", ["02", "03"], [1.0, 0.0], [0.7, -0.7]).render(tmp_path / "s.gif")
    with Image.open(path) as gif:
        assert gif.is_animated
    cands = [f"{n:02d}" for n in range(100)]
    flat = [1 / 100] * 100
    film = PolicyEvolutionFilm(
        [0, 10], [flat, flat], cands, "02", [{"step": 0, "seen": 0, "unseen": 0}], "1+1=", tweens=1
    )
    assert film.render(tmp_path / "e.gif").stat().st_size > 0
    with pytest.raises(ValueError):
        GRPOStepFilm("1+1=", ["02"], [1.0, 0.0], [0.0])
