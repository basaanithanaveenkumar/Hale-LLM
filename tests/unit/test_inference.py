import hale_llm.models  # noqa: F401
from hale_llm.config.schema import RunConfig
from hale_llm.registry import get_variant
from hale_llm.inference import session_from_model


def test_session_generate_returns_text(tiny_tokenizer):
    cfg = RunConfig(
        variant="autoregressive",
        train={"batch_size": 2, "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        sample={"max_new_tokens": 3, "prompt": "hi", "temperature": 1.0},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    session = session_from_model(model, tiny_tokenizer, cfg, device="cpu")
    result = session.generate("hi")
    assert result.prompt == "hi"
    assert result.token_ids.ndim == 2
    assert result.token_ids.size(1) > 0
    assert isinstance(result.text, str)
    hist = session.generate("hi", return_history=True)
    assert hist.history is not None
    assert len(hist.history) >= 1


def test_load_session_falls_back_when_checkpoint_missing(tiny_tokenizer, tmp_path):
    from hale_llm.inference import load_session

    cfg = RunConfig(
        variant="autoregressive",
        train={"checkpoint_path": str(tmp_path / "missing.pt"), "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    session = load_session(cfg, tokenizer=tiny_tokenizer, require_checkpoint=False)
    assert session.model is not None
    result = session.generate("hi")
    assert isinstance(result.text, str)


def test_load_session_explicit_missing_checkpoint_raises(tiny_tokenizer, tmp_path):
    from hale_llm.inference import load_session
    import pytest

    cfg = RunConfig(
        variant="autoregressive",
        train={"checkpoint_path": str(tmp_path / "also_missing.pt"), "steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    with pytest.raises(FileNotFoundError):
        load_session(cfg, str(tmp_path / "explicit.pt"), tokenizer=tiny_tokenizer)


def test_autoregressive_sampler_stops_at_the_end_of_text_token(tiny_tokenizer):
    """A stub model that always predicts EOS: the sampler adds one token and stops."""
    import torch

    from hale_llm.models.models_autoregressive.sample import sample_autoregressive

    class AlwaysEos(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = type("B", (), {"max_length": 16})()

        def forward(self, batch):
            ids = batch["input_ids"]
            logits = torch.full((ids.size(0), ids.size(1), 64), -1e4)
            logits[:, :, tiny_tokenizer.eos_token_id] = 1e4
            return logits

    prompt = torch.tensor([[5, 6]])
    out = sample_autoregressive(AlwaysEos(), tiny_tokenizer, prompt, max_new_tokens=8)
    assert out.tolist() == [[5, 6, tiny_tokenizer.eos_token_id]]


def test_session_cuts_text_after_the_stop_token(tiny_tokenizer, monkeypatch):
    """A sampler that fills the whole length still ends the text at EOS."""
    import torch

    from hale_llm.inference import session as session_module

    cfg = RunConfig(variant="autoregressive", model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 16}, device="cpu")
    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    session = session_from_model(model, tiny_tokenizer, cfg, device="cpu")
    eos = tiny_tokenizer.eos_token_id
    monkeypatch.setattr(
        session_module, "get_sampler", lambda variant: lambda *a, **k: torch.tensor([[40, 41, 50, eos, 51, 52]])
    )
    result = session.generate("hi")  # prompt "hi" = 2 tokens
    assert result.token_ids.size(1) == 6  # raw sampler output is untouched
    assert result.text == tiny_tokenizer.decode([40, 41, 50])  # text ends before EOS
