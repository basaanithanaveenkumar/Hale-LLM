"""A tiny GPT-style language model + toy tasks, small enough to RL-train on a laptop CPU in seconds.

Why a toy? Every LLM RL algorithm (PPO-RLHF, DPO, GRPO, ...) only needs three things from the
model:

1. ``generate``: sample a response token by token from the policy pi(y | x);
2. ``sequence_logprobs``: the log-probability of every response token under some model
   (policy, old policy, reference) - this is what the RL losses are built from;
3. a ``reward`` for a finished response.

A 2-layer transformer on a 20-symbol vocabulary exercises exactly the same code paths as a 70B
model, so you can see an algorithm *learn* in a few hundred steps instead of a few thousand GPUs.
Notebook ``04`` builds this model from scratch and explains every line; later notebooks import it
from here so they can focus on the RL algorithm itself.

Conventions used everywhere:

* a batch is a right-padded ``LongTensor`` of shape ``(B, T)`` holding ``prompt + response``;
* ``response_mask`` (``(B, T)`` float) is 1 on response tokens (incl. ``<eos>``), 0 on prompt/pad;
* ``sequence_logprobs`` returns per-token log-probs aligned with ``input_ids`` (position ``t`` holds
  ``log p(x_t | x_<t)``; position 0 is 0), so ``(logp * response_mask).sum(-1)`` is log pi(y|x).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------------------------- #
# Tokenizer
# --------------------------------------------------------------------------------------------- #
class CharTokenizer:
    """Character-level tokenizer with ``<pad>`` (0), ``<bos>`` (1) and ``<eos>`` (2) specials."""

    PAD, BOS, EOS = 0, 1, 2

    def __init__(self, chars: str = "0123456789+-*=?abcdefghijklmnopqrstuvwxyz ,.:"):
        self.itos = ["<pad>", "<bos>", "<eos>"] + list(dict.fromkeys(chars))
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, bos: bool = True) -> List[int]:
        return ([self.BOS] if bos else []) + [self.stoi[c] for c in text]

    def decode(self, ids: Sequence[int], stop_at_eos: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.EOS and stop_at_eos:
                break
            if i in (self.PAD, self.BOS):
                continue
            out.append(self.itos[i] if i != self.EOS else "")
        return "".join(out)


# --------------------------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------------------------- #
@dataclass
class GPTConfig:
    vocab_size: int
    max_len: int = 48
    d_model: int = 64
    n_layer: int = 2
    n_head: int = 4
    dropout: float = 0.0  # RL wants a deterministic forward pass: old/new log-probs must match


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention with a causal mask (token t only sees tokens <= t)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=-1)
        # (B, T, C) -> (B, heads, T, head_dim)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    """Pre-norm transformer block: x + Attn(LN(x)), then x + MLP(LN(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(nn.Linear(cfg.d_model, 4 * cfg.d_model), nn.GELU(),
                                 nn.Linear(4 * cfg.d_model, cfg.d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    """Decoder-only transformer. ``forward`` returns ``(logits, hidden)``.

    ``prefix_embeds`` (``(B, P, d_model)``) lets a vision encoder prepend image tokens - that is all a
    (LLaVA-style) VLM is from the language model's point of view; see :mod:`rlkit.toy_vlm`.
    ``hidden`` is exposed so a value head / reward head can sit on top of the same trunk.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos = nn.Embedding(cfg.max_len + 32, cfg.d_model)  # +32 leaves room for image prefixes
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        # GPT-2 style init: small normal weights keep the initial policy close to uniform.
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor, prefix_embeds: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.tok(idx)
        if prefix_embeds is not None:
            x = torch.cat([prefix_embeds, x], dim=1)
        T = x.shape[1]
        x = x + self.pos(torch.arange(T, device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        h = self.ln_f(x)
        if prefix_embeds is not None:  # only return text positions
            h = h[:, prefix_embeds.shape[1]:]
        return self.head(h), h


class ValueHead(nn.Module):
    """Scalar head on the trunk's hidden states: used as PPO critic V(s_t) or as a reward model."""

    def __init__(self, d_model: int):
        super().__init__()
        self.v = nn.Linear(d_model, 1)
        nn.init.zeros_(self.v.weight)
        nn.init.zeros_(self.v.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.v(hidden).squeeze(-1)


# --------------------------------------------------------------------------------------------- #
# Sampling + log-probs (the two primitives every RL algorithm needs)
# --------------------------------------------------------------------------------------------- #
@torch.no_grad()
def generate(model: TinyGPT, prompts: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             eos_id: int = CharTokenizer.EOS, pad_id: int = CharTokenizer.PAD,
             prefix_embeds: Optional[torch.Tensor] = None, greedy: bool = False
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample responses for equal-length ``prompts`` (``(B, P)``).

    Returns ``(sequences, response_mask)`` with shape ``(B, P + max_new_tokens)``. Tokens after a
    sequence's ``<eos>`` are ``<pad>`` and masked out; the ``<eos>`` itself is part of the response
    (the policy *chose* to stop, so it gets credit/blame for that decision).
    """
    model.eval()
    seq = prompts.clone()
    B = seq.shape[0]
    done = torch.zeros(B, dtype=torch.bool, device=seq.device)
    mask = torch.zeros(B, prompts.shape[1], device=seq.device)
    masks = [mask]
    for _ in range(max_new_tokens):
        logits, _ = model(seq, prefix_embeds=prefix_embeds)
        logits = logits[:, -1, :]
        if greedy:
            nxt = logits.argmax(-1)
        else:
            nxt = torch.multinomial(F.softmax(logits / temperature, dim=-1), 1).squeeze(-1)
        nxt = torch.where(done, torch.full_like(nxt, pad_id), nxt)
        masks.append((~done).float().unsqueeze(1))
        seq = torch.cat([seq, nxt.unsqueeze(1)], dim=1)
        done = done | (nxt == eos_id)
        if done.all():
            break
    model.train()
    return seq, torch.cat(masks, dim=1)


def sequence_logprobs(model: TinyGPT, seq: torch.Tensor, prefix_embeds: Optional[torch.Tensor] = None,
                      return_hidden: bool = False):
    """Per-token ``log p(x_t | x_<t)`` aligned with ``seq`` (position 0 is filled with 0).

    One forward pass scores the whole sequence - this "teacher-forced" scoring is why RL on LLMs is
    affordable: sampling is sequential, but re-scoring a finished sample is a single parallel pass.
    """
    logits, hidden = model(seq, prefix_embeds=prefix_embeds)
    logp = F.log_softmax(logits[:, :-1], dim=-1).gather(-1, seq[:, 1:].unsqueeze(-1)).squeeze(-1)
    logp = torch.cat([torch.zeros_like(logp[:, :1]), logp], dim=1)
    return (logp, hidden) if return_hidden else logp


def token_entropy(model: TinyGPT, seq: torch.Tensor, response_mask: torch.Tensor,
                  prefix_embeds: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Mean entropy of the next-token distribution over response positions (exploration gauge)."""
    with torch.no_grad():
        logits, _ = model(seq, prefix_embeds=prefix_embeds)
        logp = F.log_softmax(logits[:, :-1], dim=-1)
        ent = -(logp.exp() * logp).sum(-1)
        m = response_mask[:, 1:]
        return (ent * m).sum() / m.sum().clamp(min=1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: Optional[int] = None) -> torch.Tensor:
    """Mean of ``x`` over positions where ``mask == 1`` (the workhorse of every token-level loss)."""
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


# --------------------------------------------------------------------------------------------- #
# Toy tasks
# --------------------------------------------------------------------------------------------- #
class AdditionTask:
    """Prompts ``"a+b="`` -> answer ``"c"`` then ``<eos>``. A *verifiable* reward: exact match.

    ``max_operand`` controls difficulty. All prompts have the same length (operands are zero-padded
    to ``digits`` characters) so a batch needs no left-padding - one less thing to explain.
    """

    def __init__(self, tokenizer: CharTokenizer, digits: int = 1, seed: int = 0):
        self.tok = tokenizer
        self.digits = digits
        self.max_operand = 10 ** digits - 1
        self.rng = random.Random(seed)
        self.answer_len = digits + 1

    def sample_problem(self) -> Tuple[str, str]:
        a, b = self.rng.randint(0, self.max_operand), self.rng.randint(0, self.max_operand)
        return f"{a:0{self.digits}d}+{b:0{self.digits}d}=", str(a + b)

    def all_problems(self) -> List[Tuple[str, str]]:
        n = self.max_operand + 1
        return [(f"{a:0{self.digits}d}+{b:0{self.digits}d}=", str(a + b)) for a in range(n) for b in range(n)]

    def batch(self, n: int, problems: Optional[List[Tuple[str, str]]] = None
              ) -> Tuple[torch.Tensor, List[str]]:
        """``n`` random prompts as a ``(n, P)`` tensor plus their gold answers."""
        probs = [self.rng.choice(problems) if problems else self.sample_problem() for _ in range(n)]
        ids = torch.tensor([self.tok.encode(p) for p, _ in probs])
        return ids, [a for _, a in probs]

    def reward(self, seq: torch.Tensor, prompt_len: int, answers: List[str]) -> torch.Tensor:
        """1.0 if the decoded response equals the gold answer, else 0.0."""
        outs = [self.tok.decode(s[prompt_len:]) for s in seq.tolist()]
        return torch.tensor([float(o == a) for o, a in zip(outs, answers)])

    def sft_batch(self, n: int, problems: Optional[List[Tuple[str, str]]] = None) -> torch.Tensor:
        """Teacher-forcing batch ``prompt + answer + <eos>`` (right-padded) for supervised warm-up."""
        probs = [self.rng.choice(problems) if problems else self.sample_problem() for _ in range(n)]
        rows = [self.tok.encode(p) + self.tok.encode(a, bos=False) + [self.tok.EOS] for p, a in probs]
        L = max(map(len, rows))
        return torch.tensor([r + [self.tok.PAD] * (L - len(r)) for r in rows])


def sft_train(model: TinyGPT, batches, steps: int, lr: float = 3e-3, prompt_len: Optional[int] = None,
              log_every: int = 0) -> List[float]:
    """Plain next-token cross-entropy ("supervised fine-tuning"). ``batches()`` yields ``(B, T)`` ids.

    If ``prompt_len`` is given, prompt tokens are excluded from the loss (we only teach the answer).
    Returns the loss history.
    """
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    hist = []
    for step in range(steps):
        x = batches()
        logits, _ = model(x)
        targets = x[:, 1:].clone()
        targets[targets == CharTokenizer.PAD] = -100
        if prompt_len is not None:
            targets[:, : prompt_len - 1] = -100
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1),
                               ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        hist.append(loss.item())
        if log_every and step % log_every == 0:
            print(f"sft step {step:4d}  loss {loss.item():.3f}")
    return hist


@torch.no_grad()
def evaluate(model: TinyGPT, task: AdditionTask, problems: List[Tuple[str, str]], greedy: bool = True,
             temperature: float = 1.0) -> float:
    """Accuracy of the model on ``problems`` (greedy decoding by default)."""
    ids = torch.tensor([task.tok.encode(p) for p, _ in problems])
    seq, _ = generate(model, ids, max_new_tokens=task.answer_len + 1, greedy=greedy, temperature=temperature)
    return task.reward(seq, ids.shape[1], [a for _, a in problems]).mean().item()


def make_policy(tokenizer: CharTokenizer, seed: int = 0, **cfg_kwargs) -> TinyGPT:
    """A freshly initialised :class:`TinyGPT` for ``tokenizer`` (seeded for reproducibility)."""
    torch.manual_seed(seed)
    return TinyGPT(GPTConfig(vocab_size=tokenizer.vocab_size, **cfg_kwargs))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def kl_k3(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's unbiased, always-positive KL(pi || ref) estimator: exp(r) - r - 1, r = ref - logp."""
    r = ref_logp - logp
    return torch.exp(r) - r - 1


__all__ = [
    "CharTokenizer", "GPTConfig", "TinyGPT", "ValueHead", "generate", "sequence_logprobs", "token_entropy",
    "masked_mean", "AdditionTask", "sft_train", "evaluate", "make_policy", "count_parameters", "kl_k3",
]
