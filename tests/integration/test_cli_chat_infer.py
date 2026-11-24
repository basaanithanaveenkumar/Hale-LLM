"""CLI-level checks for hale-llm-infer and hale-llm-chat with a stubbed model loader."""

import json

import pytest

import hale_llm.models  # noqa: F401
from hale_llm.chat import ChatSession
from hale_llm.cli import chat as chat_cli
from hale_llm.cli import inference as infer_cli
from hale_llm.config.schema import RunConfig
from hale_llm.core.registry import get_variant
from hale_llm.inference import session_from_model


@pytest.fixture
def session(tiny_tokenizer):
    cfg = RunConfig(
        variant="autoregressive",
        train={"steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 32},
        sample={"max_new_tokens": 4},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    return session_from_model(model, tiny_tokenizer, cfg, device="cpu")


def test_infer_json_output_for_each_prompt(session, monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(infer_cli, "open_session", lambda args: session)
    prompt_file = tmp_path / "prompts.txt"
    prompt_file.write_text("second\n\nthird\n")
    infer_cli.main(["--config", "x.yaml", "--prompt", "first", "--prompt-file", str(prompt_file), "--json"])
    rows = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [r["prompt"] for r in rows] == ["first", "second", "third"]


def test_infer_missing_prompt_file_exits_cleanly(session, monkeypatch):
    monkeypatch.setattr(infer_cli, "open_session", lambda args: session)
    with pytest.raises(SystemExit) as exc:
        infer_cli.main(["--config", "x.yaml", "--prompt-file", "/no/such/file.txt"])
    assert "hale-llm-infer" in str(exc.value)


def test_chat_loop_handles_messages_commands_and_exit(session):
    scripted = iter(["hello", "/temp 0.7", "", "/history", "/quit"])
    output: list[str] = []
    chat_cli.run_loop(ChatSession(session), read=lambda _: next(scripted), write=output.append)
    text = "\n".join(output)
    assert "model>" in text
    assert "temperature = 0.7" in text
    assert "you> hello" in text  # printed by /history
    assert output[-1] == "bye"


def test_chat_loop_ends_on_eof(session):
    def eof(_):
        raise EOFError

    chat_cli.run_loop(ChatSession(session), read=eof, write=lambda _: None)
