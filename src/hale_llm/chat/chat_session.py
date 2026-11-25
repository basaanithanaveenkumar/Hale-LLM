"""Multi-turn chat on top of an :class:`InferenceSession`."""

from __future__ import annotations

from loguru import logger

from hale_llm.chat.chat_turn import ChatTurn
from hale_llm.chat.prompt_window import PromptWindow
from hale_llm.inference.encode import encode_prompt
from hale_llm.inference.stop import cut_at_stop
from hale_llm.inference.session import InferenceSession


class ChatSession:
    """Keeps a running transcript and asks the model to continue it.

    The checkpoints in this repo are base language models trained on TinyStories,
    not instruction-tuned assistants. "Chat" therefore means: append the user's
    message to the transcript, let the sampler continue it, and treat the new
    tokens as the reply. Every paradigm works because only the registered
    sampler is used.

    Steps of one :meth:`send` call:

    1. add the user's message to the transcript,
    2. trim the transcript from the left to fit the context window,
    3. run the sampler on the trimmed transcript,
    4. cut the prompt tokens off the output and decode what is left.
    """

    def __init__(
        self,
        session: InferenceSession,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
    ) -> None:
        self._session = session
        sample_cfg = session.cfg.sample
        self.max_new_tokens = max_new_tokens or sample_cfg.max_new_tokens
        self.temperature = temperature if temperature is not None else sample_cfg.temperature
        self._turns: list[ChatTurn] = []
        self._context = ""

    # ----- settings (validated so the CLI cannot push the sampler into a bad state) -----

    @property
    def max_new_tokens(self) -> int:
        return self._max_new_tokens

    @max_new_tokens.setter
    def max_new_tokens(self, value: int) -> None:
        if value < 1:
            raise ValueError(f"max_new_tokens must be >= 1, got {value}")
        if value >= self._context_length():
            raise ValueError(
                f"max_new_tokens ({value}) must be smaller than the model context "
                f"({self._context_length()}) so some room is left for the prompt"
            )
        self._max_new_tokens = value

    @property
    def temperature(self) -> float:
        return self._temperature

    @temperature.setter
    def temperature(self, value: float) -> None:
        if value <= 0:
            raise ValueError(f"temperature must be > 0, got {value}")
        self._temperature = value

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        return tuple(self._turns)

    # ----- conversation -----

    def reset(self) -> None:
        """Forget the transcript."""
        self._turns.clear()
        self._context = ""
        logger.info("chat transcript cleared")

    def send(self, user_text: str) -> ChatTurn:
        """Add ``user_text`` to the transcript and return the model's reply."""
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("message is empty")

        # Step 1-2: transcript + new message, trimmed to leave room for generation.
        budget = self._context_length() - self._max_new_tokens
        window = PromptWindow(self._session.tokenizer, budget)
        prompt = window.fit(f"{self._context} {user_text}".strip())

        # Step 3: run the registered sampler through the shared inference session.
        result = self._session.generate(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=self._temperature,
        )

        # Step 4: the sampler returns prompt + continuation; keep only the continuation.
        prompt_len = encode_prompt(
            self._session.tokenizer, prompt, self._session.model, self._session.device
        ).size(1)
        kept = cut_at_stop(result.token_ids[0], prompt_len, getattr(self._session.tokenizer, "eos_token_id", None))
        new_ids = kept[prompt_len:].tolist()
        reply = self._session.tokenizer.decode(new_ids, skip_special_tokens=True).strip()

        turn = ChatTurn(user=user_text, reply=reply)
        self._turns.append(turn)
        self._context = f"{prompt} {reply}".strip()
        return turn

    def _context_length(self) -> int:
        """Tokens the backbone can see; falls back to the config value."""
        backbone = getattr(self._session.model, "backbone", None)
        return getattr(backbone, "max_length", None) or self._session.cfg.model.max_length
