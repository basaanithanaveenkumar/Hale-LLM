"""Litmus test: sample data -> real windows/collate/trainer -> checkpoint -> model -> decoding.

The corpus is a counting stream ("5 6 7 8 ..." modulo 20) so the right answer for every head is known
without running any model: depth k at a position holding the number x must predict (x + k + 1) % 20, i.e.
the token k + 1 steps ahead. If any index were off by one anywhere between the dataset and the decoder,
some head would be wrong by exactly one step, and this test would say so.
"""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.dataloader.data_module import DataModule
from hale_llm.inference import load_session
from hale_llm.models.models_autoregressive.sample import sample_autoregressive
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.training.lean_trainer import train_lean

MOD, EOS, DOC_LEN, DEPTH, WINDOW = 20, 23, 37, 4, 24


class CountingTokenizer:
    """The word "7" is token 7. EOS is 23. No pad token, like the real tokenizer."""

    eos_token_id, pad_token_id, mask_token_id = EOS, None, None

    def __len__(self):
        return 24

    def __call__(self, text, add_special_tokens=False, truncation=False, padding=False, return_tensors=None, **_):
        return {"input_ids": [int(word) for word in text.split()]}

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(i) for i in ids if not (skip_special_tokens and i == EOS))


def _documents(count: int, seed: int) -> list[dict]:
    starts = torch.randint(0, MOD, (count,), generator=torch.Generator().manual_seed(seed)).tolist()
    return [{"text": " ".join(str((start + i) % MOD) for i in range(DOC_LEN))} for start in starts]


@pytest.fixture
def counting_pipeline(monkeypatch):
    """The real dataset/windows/collate code, fed from memory instead of the Hugging Face Hub."""
    tokenizer = CountingTokenizer()
    splits = {"train": _documents(400, seed=0), "validation": _documents(60, seed=1)}

    def fake_load_split(cfg, split):
        return splits["train" if split == cfg.data.train_split else "validation"]

    monkeypatch.setattr("hale_llm.dataloader.sources.huggingface.load_hf_split", fake_load_split)
    monkeypatch.setattr("hale_llm.dataloader.loaders.build_tokenizer_for", lambda cfg, **_: tokenizer)
    monkeypatch.setattr("hale_llm.dataloader.data_module.build_tokenizer_for", lambda cfg, **_: tokenizer)
    return tokenizer


def _cfg(tmp_path, variant="autoregressive_mtp"):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.variant, cfg.model.ffn_type, cfg.model.moe_hidden_dim = variant, "moe" if "moe" in variant else "mlp", 48
    cfg.data.source = "huggingface"
    cfg.data.train_size = cfg.data.val_size = None
    cfg.set_context_window(WINDOW)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 48, 4, 2, 96
    cfg.model.dropout, cfg.model.mtp_depth, cfg.model.mtp_loss_weight = 0.0, DEPTH, 1.0
    cfg.model.mtp_loss_weight_final = None  # a constant lambda: this test is about the indices, not the schedule
    cfg.train.steps, cfg.train.epochs, cfg.train.batch_size, cfg.train.lr, cfg.train.log_every = 500, None, 16, 4e-3, 100
    cfg.train.lean.accum_steps, cfg.train.lean.warmup_steps = 1, 10
    cfg.train.lean.eval_every, cfg.train.lean.save_every = 0, 0
    cfg.train.checkpoint_path = str(tmp_path / "ckpt" / "last.pt")
    cfg.device = "cpu"
    cfg.logging.tensorboard, cfg.logging.log_file, cfg.logging.level = False, None, "WARNING"
    cfg.experiment.enabled, cfg.train.resume = False, False
    return cfg


@pytest.mark.parametrize("variant", ["autoregressive_mtp", "autoregressive_moe_mtp"])
def test_every_head_predicts_the_right_step_ahead_from_data_to_decoder(variant, counting_pipeline, tmp_path):
    cfg = _cfg(tmp_path, variant)
    tokenizer = counting_pipeline
    train_lean(cfg, tokenizer=tokenizer)  # real windows, real collate, real loss, real optimizer, real checkpoint
    session = load_session(cfg, tokenizer=tokenizer, require_checkpoint=True)  # checkpoint -> model
    model = session.model.double().eval()
    assert model.mtp.depth == DEPTH

    # ---- (a) the data pipeline: windows are consecutive pieces of the stream and labels are shifted by one ----
    evaluated = [0] * DEPTH
    correct = [0] * DEPTH
    for batch in DataModule(cfg, tokenizer).val_loader():
        ids, labels = batch["input_ids"], batch["labels"]
        assert ids.shape[1] == WINDOW
        assert torch.equal(labels[:, :-1], ids[:, 1:]) and bool((labels[:, -1] == -100).all())
        no_eos = ids != EOS
        stream_ok = no_eos[:, :-1] & no_eos[:, 1:]
        assert torch.equal(ids[:, 1:][stream_ok], ((ids[:, :-1] + 1) % MOD)[stream_ok])  # counting is intact inside windows

        # ---- (b) the training-side heads, on real windows, judged by arithmetic ----
        with torch.no_grad():
            logits = model.mtp(model.backbone.encode(ids), ids, None, model.backbone.token_emb, model.backbone.lm_head)
        for k, depth_logits in enumerate(logits, start=1):  # depth k: position j predicts t_{j+k+1}
            predicted = depth_logits[:, :-1].argmax(-1)  # [B, T-k-1]
            starts = ids[:, : WINDOW - k - 1]
            span_has_eos = torch.stack([(ids[:, j : j + k + 2] == EOS).any(dim=1) for j in range(WINDOW - k - 1)], dim=1)
            valid = ~span_has_eos
            expected = (starts + k + 1) % MOD
            assert torch.equal(expected[valid], ids[:, k + 1 :][valid]), "the data itself disagrees with the arithmetic"
            evaluated[k - 1] += int(valid.sum())
            correct[k - 1] += int((predicted == expected)[valid].sum())
    for k in range(DEPTH):
        assert evaluated[k] > 100, f"depth {k + 1}: too few positions checked to mean anything"
        assert correct[k] / evaluated[k] > 0.97, f"depth {k + 1} accuracy {correct[k] / evaluated[k]:.3f}"

    # ---- (c) the inference side, from a prompt: the chain must draft the next numbers, and be accepted ----
    prompt = torch.tensor([[5, 6, 7, 8, 9, 10]])
    expected_text = [(10 + i) % MOD for i in range(1, 15)]  # 11 12 13 ... : what counting says comes next
    decoder = SpeculativeDecoder(model, tokenizer, 0.0)
    output = decoder.generate(prompt, 14)
    assert output[0, 6:].tolist() == expected_text
    assert torch.equal(output, sample_autoregressive(model, tokenizer, prompt, 14, temperature=0.0))
    assert all(rate > 0.9 for rate in decoder.acceptance_by_depth()), decoder.acceptance_by_depth()
    assert decoder.rounds <= 4  # about 5 tokens per trunk pass instead of 1: 14 tokens in ~3 rounds, not 13 passes
