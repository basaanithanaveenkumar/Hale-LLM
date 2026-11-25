"""Unit tests for the chat package (window trimming, session, slash commands)."""

import pytest

import hale_llm.models  # noqa: F401
from hale_llm.chat import ChatCommands, ChatSession, PromptWindow
from hale_llm.config.schema import RunConfig
from hale_llm.registry import get_variant
from hale_llm.inference import session_from_model


@pytest.fixture
def chat(tiny_tokenizer):
    cfg = RunConfig(
        variant="autoregressive",
        train={"steps": 1, "epochs": None},
        model={"d_model": 32, "n_heads": 4, "n_layers": 1, "d_ff": 64, "max_length": 32},
        sample={"max_new_tokens": 4, "temperature": 1.0},
        logging={"tensorboard": False, "log_file": None, "level": "WARNING"},
        device="cpu",
    )
    model = get_variant("autoregressive")(vocab_size=len(tiny_tokenizer), cfg=cfg)
    return ChatSession(session_from_model(model, tiny_tokenizer, cfg, device="cpu"))


def test_prompt_window_keeps_newest_tokens(tiny_tokenizer):
    # Tiny tokenizer maps one character to one token.
    window = PromptWindow(tiny_tokenizer, max_tokens=5)
    assert window.fit("abc") == "abc"
    assert len(window.fit("abcdefghij")) == 5


def test_prompt_window_rejects_empty_budget(tiny_tokenizer):
    with pytest.raises(ValueError):
        PromptWindow(tiny_tokenizer, max_tokens=0)


def test_send_records_turn_and_grows_context(chat):
    turn = chat.send("hello there")
    assert turn.user == "hello there"
    assert len(chat.turns) == 1
    chat.send("again")
    assert len(chat.turns) == 2


def test_long_conversation_stays_inside_context(chat):
    # 40 turns of 20 characters is far beyond the 32-token context; must not raise.
    for _ in range(40):
        chat.send("x" * 20)
    assert len(chat.turns) == 40


def test_send_rejects_blank_message(chat):
    with pytest.raises(ValueError, match="empty"):
        chat.send("   ")


def test_setters_validate(chat):
    with pytest.raises(ValueError):
        chat.temperature = 0
    with pytest.raises(ValueError):
        chat.max_new_tokens = 32  # equals the context length, no room for a prompt


def test_reset_clears_turns(chat):
    chat.send("hi")
    chat.reset()
    assert chat.turns == ()


def test_commands_change_settings_and_report_errors(chat):
    cmds = ChatCommands(chat)
    assert "0.5" in cmds.execute("/temp 0.5").message
    assert chat.temperature == 0.5
    assert "invalid" in cmds.execute("/temp abc").message
    assert "invalid" in cmds.execute("/tokens 0").message
    assert "unknown" in cmds.execute("/nope").message


def test_quit_requests_exit(chat):
    assert ChatCommands(chat).execute("/quit").exit_requested is True
    assert ChatCommands.is_command("  /help")
    assert not ChatCommands.is_command("hello")
