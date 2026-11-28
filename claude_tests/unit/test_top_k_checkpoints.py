"""Best-N checkpoint keeping: ranking, eviction and the trainer hook."""

from __future__ import annotations

import math

import pytest

from hale_llm.training.top_k_checkpoints import TopKCheckpoints


def _files(tmp_path):
    return sorted(p.name for p in tmp_path.glob("*.pt"))


def test_keeps_the_lowest_losses_and_deletes_the_rest(tmp_path):
    best = TopKCheckpoints(tmp_path, keep=3, score_name="loss", higher_is_better=False)
    for step, loss in [(1, 5.0), (2, 4.0), (3, 6.0), (4, 3.0), (5, 4.5)]:
        best.offer(loss, step, model_state_dict={})
    assert [p.name for p in best.paths] == ["step0000004_loss_3.0000.pt", "step0000002_loss_4.0000.pt", "step0000005_loss_4.5000.pt"]
    assert _files(tmp_path) == sorted(p.name for p in best.paths)  # worse ones are gone from disk


def test_keeps_the_highest_accuracies(tmp_path):
    best = TopKCheckpoints(tmp_path, keep=2, score_name="token_accuracy", higher_is_better=True)
    for step, acc in [(1, 0.2), (2, 0.5), (3, 0.4), (4, 0.1)]:
        best.offer(acc, step, model_state_dict={})
    assert [p.name for p in best.paths] == ["step0000002_token_accuracy_0.5000.pt", "step0000003_token_accuracy_0.4000.pt"]


def test_a_worse_score_is_not_saved(tmp_path):
    best = TopKCheckpoints(tmp_path, keep=1, score_name="loss", higher_is_better=False)
    assert best.offer(1.0, 1, model_state_dict={}) is not None
    assert best.offer(2.0, 2, model_state_dict={}) is None
    assert len(_files(tmp_path)) == 1


@pytest.mark.parametrize("score", [math.nan, math.inf])
def test_non_finite_scores_are_ignored(tmp_path, score):
    best = TopKCheckpoints(tmp_path, keep=3, score_name="loss", higher_is_better=False)
    assert best.offer(score, 1, model_state_dict={}) is None


def test_keep_zero_disables_it(tmp_path):
    best = TopKCheckpoints(tmp_path, keep=0, score_name="loss", higher_is_better=False)
    assert best.offer(1.0, 1, model_state_dict={}) is None and not list(tmp_path.glob("*.pt"))


def test_saved_file_records_step_and_score(tmp_path):
    import torch
    best = TopKCheckpoints(tmp_path, keep=1, score_name="loss", higher_is_better=False)
    path = best.offer(1.25, 7, model_state_dict={"w": 1})
    state = torch.load(path, weights_only=False)
    assert state["step"] == 7 and state["loss"] == 1.25 and state["model_state_dict"] == {"w": 1}
