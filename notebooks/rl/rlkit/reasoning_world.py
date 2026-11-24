"""A tiny "pre-training corpus" and tokenizer for the DeepSeek-R1 notebook (15).

R1's story starts from a *base model*: a network that has read lots of text, written by people of very
different carefulness, and has never been told to "think step by step". We cannot pre-train on the internet on a
laptop, but we can write a miniature internet whose authors behave like that:

* the task is ``Q a+b+c=`` with ``a, b, c`` in 0..49, answered in the R1 template
  ``<think> ... </think><answer> N </answer>``;
* **three writing habits** appear in the corpus (this is the only place where we decide what the base model "knows"):

  ============  ==========================================================  ======================
  habit         what the author writes                                      how reliable
  ============  ==========================================================  ======================
  ``direct``    skips the work: ``<think></think><answer> 71 </answer>``   wrong 40 % of the time
  ``steps``     ``a+b=s1 then s1+c=s2`` (two easy additions)                each step wrong 12 %
  ``checked``   like ``steps`` but writes ``wait`` and redoes a step        notices its slip 85 %
                after noticing a slip
  ============  ==========================================================  ======================

* two more imperfections that real text has: **sloppy formatting** (some authors forget the tags) and **language
  mixing** (the connector word between steps is sometimes French: ``puis`` instead of ``then``).

Nothing here is RL. The *algorithms* of the notebook (rewards, GRPO, rejection sampling, distillation) are written inline
in the notebook itself.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import torch

from .toy_lm import CharTokenizer

THINK, END_THINK, ANSWER, END_ANSWER = "<think>", "</think>", "<answer>", "</answer>"
EN, FR, WAIT = "then", "puis", "wait"  # English connector, French connector, the reflection marker
WORDS = [THINK, END_THINK, ANSWER, END_ANSWER, EN, FR, WAIT, "Q", "+", "=", " "]
MAX_OPERAND = 49


class ReasoningTokenizer(CharTokenizer):
    """Word-level tokenizer: ten digits, ``+ =``, the four template tags and three connector words.

    Same special ids as :class:`CharTokenizer` (``<pad>`` 0, ``<bos>`` 1, ``<eos>`` 2) so ``rlkit.toy_lm.generate`` works.
    """

    def __init__(self) -> None:
        self.itos = ["<pad>", "<bos>", "<eos>"] + list("0123456789") + [w for w in WORDS if w != " "]
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    def ids(self, names: Sequence[str]) -> List[int]:
        return [self.stoi[n] for n in names]

    def pretty(self, ids: Sequence[int]) -> str:
        """Readable text: digits glued into numbers, words separated by spaces, stops at ``<eos>``."""
        out: List[str] = []
        for i in ids:
            i = int(i)
            if i == self.EOS:
                break
            if i in (self.PAD, self.BOS):
                continue
            t = self.itos[i]
            if t.isdigit() and out and out[-1][-1:].isdigit():
                out[-1] += t
            else:
                out.append(t)
        return " ".join(out).replace(" = ", "=").replace(" + ", "+")


def digits(n: int, pad: int = 0) -> List[str]:
    return list(str(n).zfill(pad))


@dataclass(frozen=True)
class Problem:
    a: int
    b: int
    c: int

    @property
    def gold(self) -> int:
        return self.a + self.b + self.c

    def prompt_tokens(self) -> List[str]:
        return ["Q"] + digits(self.a, 2) + ["+"] + digits(self.b, 2) + ["+"] + digits(self.c, 2) + ["="]


class ProblemSet:
    """Disjoint train / test problems (a hash of ``(a, b, c)`` decides the split, 80 / 20)."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = random.Random(seed)
        all_p = [Problem(a, b, c) for a in range(MAX_OPERAND + 1) for b in range(MAX_OPERAND + 1) for c in range(MAX_OPERAND + 1)]
        self.train = [p for p in all_p if (p.a * 31 + p.b * 17 + p.c * 7) % 5 != 0]
        self.test = [p for p in all_p if (p.a * 31 + p.b * 17 + p.c * 7) % 5 == 0]

    def sample(self, n: int, split: str = "train") -> List[Problem]:
        pool = self.train if split == "train" else self.test
        return [self.rng.choice(pool) for _ in range(n)]


def encode_prompts(tok: ReasoningTokenizer, problems: Sequence[Problem]) -> torch.Tensor:
    """``(n, P)`` prompt tensor: all prompts have the same length, so no padding is needed."""
    return torch.tensor([[tok.BOS] + tok.ids(p.prompt_tokens()) for p in problems])


class CorpusWriter:
    """Writes documents the way the three kinds of authors would (see the module docstring)."""

    def __init__(self, seed: int = 0, p_direct: float = 0.40, p_steps: float = 0.35, p_sloppy_format: float = 0.12,
                 p_french: float = 0.25, direct_err: float = 0.40, step_err: float = 0.12, notice: float = 0.85) -> None:
        self.rng = random.Random(seed)
        self.p_direct, self.p_steps, self.p_sloppy = p_direct, p_steps, p_sloppy_format
        self.p_french, self.direct_err, self.step_err, self.notice = p_french, direct_err, step_err, notice

    def slip(self, x: int) -> int:
        """A wrong version of ``x`` (an author's arithmetic slip)."""
        return max(0, x + self.rng.choice([-10, -2, -1, 1, 2, 10]))

    def connector(self) -> str:
        return FR if self.rng.random() < self.p_french else EN

    def step(self, x: int, y: int, may_slip: float, checked: bool) -> List[str]:
        """``x+y=z`` possibly with a slip; a ``checked`` author who notices writes ``wait`` and the right sum."""
        z = x + y
        shown = self.slip(z) if self.rng.random() < may_slip else z
        out = digits(x, 2) + ["+"] + digits(y, 2) + ["="] + digits(shown)
        if shown != z and checked and self.rng.random() < self.notice:
            out += [WAIT] + digits(x, 2) + ["+"] + digits(y, 2) + ["="] + digits(z)
            shown = z
        self._last = shown
        return out

    def document(self, p: Problem) -> Tuple[List[str], str]:
        """``(response tokens, habit)``: everything after the ``=`` of the prompt, ending with ``<eos>`` implied by the caller."""
        r = self.rng.random()
        habit = "direct" if r < self.p_direct else ("steps" if r < self.p_direct + self.p_steps else "checked")
        if habit == "direct":
            ans = self.slip(p.gold) if self.rng.random() < self.direct_err else p.gold
            body = [THINK, END_THINK]
            final = ans
        else:
            chk = habit == "checked"
            s1 = self.step(p.a, p.b, self.step_err * (2.5 if chk else 1.0), chk)
            s1v = self._last
            s2 = self.step(s1v, p.c, self.step_err * (2.5 if chk else 1.0), chk)
            body = [THINK] + s1 + [self.connector()] + s2 + [END_THINK]
            final = self._last
        resp = body + [ANSWER] + digits(final) + [END_ANSWER]
        if self.rng.random() < self.p_sloppy:  # an author who forgot the template
            resp = [t for t in resp if t not in (THINK, END_THINK)] if self.rng.random() < 0.5 else digits(final)
        return resp, habit

    def batch(self, tok: ReasoningTokenizer, problems: Sequence[Problem]) -> torch.Tensor:
        """Right-padded ``prompt + response + <eos>`` rows for next-token training."""
        rows = []
        for p in problems:
            resp, _ = self.document(p)
            rows.append([tok.BOS] + tok.ids(p.prompt_tokens()) + tok.ids(resp) + [tok.EOS])
        L = max(map(len, rows))
        return torch.tensor([r + [tok.PAD] * (L - len(r)) for r in rows])


# --------------------------------------------------------------------------------------------- #
# Harness: sampling, parsing, diagnostics (the *infrastructure* of the experiments, not the R1 ideas)
# --------------------------------------------------------------------------------------------- #
import collections
import re

import numpy as np

from .toy_lm import GPTConfig, TinyGPT, generate

MAX_NEW = 46  # longest response we allow (the longest habit needs ~40 tokens)


def make_model(tok: ReasoningTokenizer, d_model: int = 96, n_layer: int = 2) -> TinyGPT:
    return TinyGPT(GPTConfig(vocab_size=tok.vocab_size, max_len=72, d_model=d_model, n_layer=n_layer, n_head=4))


def parse_response(tok: ReasoningTokenizer, resp_ids) -> dict:
    """Read a sampled response. ``format_ok`` is the strict R1 template; ``answer`` is read leniently (tags only)."""
    toks: List[str] = []
    eos = False
    for i in resp_ids:
        i = int(i)
        if i == tok.EOS:
            eos = True
            break
        if i != tok.PAD:
            toks.append(tok.itos[i])
    fmt_ok, ans, think = False, None, []
    if (eos and len(toks) >= 5 and toks[0] == THINK and toks[-1] == END_ANSWER
            and all(toks.count(t) == 1 for t in (THINK, END_THINK, ANSWER, END_ANSWER))):
        i1, i2, i3 = toks.index(END_THINK), toks.index(ANSWER), toks.index(END_ANSWER)
        if i1 + 1 == i2 and i2 + 1 < i3 and all(x.isdigit() for x in toks[i2 + 1:i3]):
            fmt_ok, ans, think = True, int("".join(toks[i2 + 1:i3])), toks[1:i1]
    if ans is None and ANSWER in toks and END_ANSWER in toks and toks.index(ANSWER) < toks.index(END_ANSWER):
        d = toks[toks.index(ANSWER) + 1:toks.index(END_ANSWER)]
        if d and all(x.isdigit() for x in d):
            ans = int("".join(d))
    return dict(format_ok=fmt_ok, answer=ans, think=think, n_en=toks.count(EN), n_fr=toks.count(FR), n_wait=toks.count(WAIT),
                length=len(toks) + int(eos), eos=eos, text=tok.pretty(resp_ids), ids=[int(i) for i in resp_ids if int(i) != tok.PAD])


def rollout(model: TinyGPT, tok: ReasoningTokenizer, problems: Sequence[Problem], temperature: float = 1.0, greedy: bool = False):
    """Sample one response per problem. Returns ``(seq, response_mask, prompt_len, parsed_list)``."""
    ids = encode_prompts(tok, problems)
    seq, mask = generate(model, ids, MAX_NEW, temperature=temperature, greedy=greedy)
    P = ids.shape[1]
    return seq, mask, P, [parse_response(tok, s[P:]) for s in seq.tolist()]


def step_correct(think_tokens: Sequence[str]):
    """Yield ``(is_correct, next_token)`` for every ``x+y=z`` segment inside a think block."""
    cur: List[str] = []
    for t in list(think_tokens) + ["END"]:
        if t in (EN, FR, WAIT, "END"):
            m = re.fullmatch(r"(\d+)\+(\d+)=(\d+)", "".join(cur))
            if m:
                yield int(m.group(1)) + int(m.group(2)) == int(m.group(3)), t
            cur = []
        else:
            cur.append(t)


def habit_of(par: dict) -> str:
    """Which writing habit a sampled response resembles: nofmt / direct / wait / steps."""
    if not par["format_ok"]:
        return "nofmt"
    if not par["think"]:
        return "direct"
    return "wait" if par["n_wait"] else "steps"


def summarize(parsed: Sequence[dict], problems: Sequence[Problem]) -> dict:
    """The dashboard numbers used everywhere: accuracy, format, length, habit shares, language mixing."""
    acc = np.mean([float(o["answer"] == p.gold) for o, p in zip(parsed, problems)])
    return dict(acc=float(acc), fmt=float(np.mean([o["format_ok"] for o in parsed])), length=float(np.mean([o["length"] for o in parsed])),
                wait=float(np.mean([o["n_wait"] > 0 for o in parsed])), french=float(np.mean([o["n_fr"] > 0 for o in parsed])),
                direct=float(np.mean([habit_of(o) == "direct" for o in parsed])))


def evaluate_policy(model: TinyGPT, tok: ReasoningTokenizer, n: int = 300, temperature: float = 1.0, greedy: bool = False, seed: int = 123) -> dict:
    """Held-out dashboard (test split, fixed seed so every method sees the same prompts)."""
    probs = ProblemSet(seed).sample(n, "test")
    out: List[dict] = []
    for i in range(0, n, 150):
        out += rollout(model, tok, probs[i:i + 150], temperature, greedy)[3]
    return summarize(out, probs)


def slip_detection(model: TinyGPT, tok: ReasoningTokenizer, n: int = 1500, seed: int = 5) -> dict:
    """How often does the model write ``wait`` right after a *wrong* sum vs after a *right* one?"""
    probs = ProblemSet(seed).sample(n, "test")
    cnt = collections.Counter()
    for i in range(0, n, 250):
        for o in rollout(model, tok, probs[i:i + 250])[3]:
            if o["format_ok"]:
                for ok, nxt in step_correct(o["think"]):
                    cnt[(ok, nxt == WAIT)] += 1
    rate = lambda ok: cnt[(ok, True)] / max(1, cnt[(ok, True)] + cnt[(ok, False)])  # noqa: E731
    return dict(p_wait_after_wrong=rate(False), p_wait_after_right=rate(True), n_wrong=cnt[(False, True)] + cnt[(False, False)],
                n_right=cnt[(True, True)] + cnt[(True, False)])
