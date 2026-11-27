"""Speculative decoding that uses the MTP depths as the draft model."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from loguru import logger

GREEDY_TEMPERATURE = 1e-5  # at or below this the decoder is greedy (argmax)


class SpeculativeDecoder:
    """Generate text with fewer trunk passes by letting the MTP depths guess ahead.

    One round, for committed tokens ``t_0 .. t_{L-1}`` and a *pending* token ``t_L`` that the trunk
    already chose::

        1. draft    depth 1 reads the trunk hidden states and t_L and guesses t_{L+1};
                    depth 2 reads depth 1 and t_{L+1} and guesses t_{L+2}; ... up to D drafts, D = model.mtp_draft_tokens
                    (offset-conditioned MTP: offset k reads the SAME trunk hidden states and the previous draft, so
                    draft k is "x(t + k) with offset k"; there is one module, run D times)
        2. verify   ONE trunk pass over  t_0 .. t_L, d_1 .. d_D  gives the trunk's own distribution
                    for every drafted position (and for the token after the last draft)
        3. accept   draft d_j is kept with probability min(1, p_j(d_j) / q_j(d_j)); the first rejected
                    one is replaced by a sample of norm(max(0, p_j - q_j)); if all D are kept, the
                    trunk's next distribution supplies a bonus token
        4. commit   the pending token and the accepted drafts become committed; the replacement (or
                    bonus) token is the next pending token, and the trunk hidden states of the
                    verify pass are reused, so a round costs one trunk pass

    This is exact: the tokens have the same distribution as sampling from the trunk alone (Leviathan
    et al., 2023), and at temperature 0 they are identical to plain greedy decoding. Every round yields
    between 1 and D + 1 tokens. There is no KV cache in this code base, so the saving is in trunk passes.
    The context window slides: if the text outgrows ``max_length`` the oldest tokens are dropped (the
    exact-equivalence with plain decoding holds while prompt + output fits in the window).
    """

    def __init__(self, model, tokenizer, temperature: float) -> None:
        self.model = model
        self.backbone = model.backbone
        # "depth" is the number of drafted tokens per round (model.mtp_draft_tokens, at most mtp_depth): draft j uses
        # MTP depth j (DeepSeek) or offset j (offset-conditioned).
        self.depth = model.cfg.model.mtp_draft_tokens
        self.eos_id = getattr(tokenizer, "eos_token_id", None)
        self.greedy = temperature <= GREEDY_TEMPERATURE
        self.temperature = max(temperature, GREEDY_TEMPERATURE)
        # Room in the window for the pending token and D drafts (the model guarantees max_length >= D + 2).
        self.window = self.backbone.max_length - self.depth - 1
        self.rounds = 0
        self.drafts_proposed = 0
        self.drafts_accepted = 0
        # Per depth: how often it was reached (every earlier draft accepted) and how often it was then accepted.
        self.depth_reached = [0] * self.depth
        self.depth_accepted = [0] * self.depth

    # ----- public -----
    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int, return_history: bool = False):
        """``prompt_ids`` is ``[1, n]``. Returns ``[1, n + new]`` (and a history list if asked)."""
        if prompt_ids.size(0) != 1:
            raise ValueError("speculative decoding handles one sequence at a time")
        self.model.eval()
        # 1. Trunk pass over the prompt, and the first token (exact: it comes from the trunk).
        tokens = prompt_ids[:, -self.window :]  # committed tokens [1, L]
        hidden = self.backbone.encode(tokens)  # [1, L, D]
        pending = self._sample(self._distribution(self.backbone.lm_head(hidden[:, -1])[0]))
        generated = [int(pending)]  # every generated token, the pending one included
        history = [(0.0, prompt_ids.clone())] if return_history else None

        # 2. Rounds until enough tokens exist or the text ended.
        while len(generated) < max_new_tokens and not self._has_ended(generated):
            tokens, hidden, pending, new_tokens = self._round(tokens, hidden, pending)
            generated.extend(new_tokens)
            if return_history:
                so_far = torch.tensor([generated[:max_new_tokens]], device=prompt_ids.device)
                history.append(
                    (
                        min(len(generated) / max_new_tokens, 1.0),
                        torch.cat([prompt_ids, so_far], dim=1),
                    )
                )

        # 3. Cut at the first end-of-text token, then at the requested length.
        if self.eos_id is not None and self.eos_id in generated:
            generated = generated[: generated.index(self.eos_id) + 1]
        generated = generated[:max_new_tokens]
        output = torch.cat([prompt_ids, torch.tensor([generated], device=prompt_ids.device)], dim=1)
        self._log_summary(len(generated))
        return (output, history) if return_history else output

    # ----- one round -----
    def _round(self, tokens: torch.Tensor, hidden: torch.Tensor, pending: torch.Tensor):
        # 0. Keep the window small enough for the pending token and the drafts (recompute if trimmed).
        if tokens.size(1) > self.window:
            tokens = tokens[:, -self.window :]
            hidden = self.backbone.encode(tokens)
        L = tokens.size(1)

        # 1. Draft D tokens with the MTP depths.
        drafts, draft_dists = self._draft(tokens, hidden, pending)  # D scalars, D distributions [V]

        # 2. Verify: one trunk pass over the committed tokens, the pending token and the drafts.
        pending_col = pending.view(1, 1)  # t_L
        draft_col = torch.stack(drafts).view(1, -1)  # d_1 .. d_D, [1, D]
        candidate = torch.cat([tokens, pending_col, draft_col], dim=1)  # [1, L + 1 + D]
        candidate_hidden = self.backbone.encode(candidate)  # [1, L + 1 + D, Dm]
        # Row j-1 of this is the trunk's distribution for d_j (j = 1..D); the last row is for the token after d_D.
        trunk_logits = self.backbone.lm_head(candidate_hidden[:, L:])[0]  # [D + 1, V]

        # 3. Accept a prefix of the drafts; get one more token from the trunk.
        accepted, extra = self._verify(drafts, draft_dists, trunk_logits)

        # 4. Commit the pending token and the accepted drafts. Their trunk hidden states are already known.
        tokens = torch.cat(
            [tokens, pending_col, draft_col[:, :accepted]], dim=1
        )  # [1, L + 1 + accepted]
        hidden = candidate_hidden[
            :, : L + 1 + accepted
        ]  # causal: unaffected by the rejected drafts
        new_tokens = [int(d) for d in drafts[:accepted]] + [int(extra)]

        self.rounds += 1
        self.drafts_proposed += len(drafts)
        self.drafts_accepted += accepted
        for depth_index in range(self.depth):
            self.depth_reached[depth_index] += int(
                accepted >= depth_index
            )  # drafts before this one were all kept
            self.depth_accepted[depth_index] += int(accepted > depth_index)
        return tokens, hidden, extra, new_tokens

    def _draft(self, tokens: torch.Tensor, hidden: torch.Tensor, pending: torch.Tensor):
        """Draft ``self.depth`` tokens: step k guesses t_{L+k} from the guesses before it (algorithm below)."""
        mtp, token_emb, lm_head = self.model.mtp, self.backbone.token_emb, self.backbone.lm_head
        known = torch.cat([tokens, pending.view(1, 1)], dim=1)  # t_0 .. t_L, [1, L + 1]
        previous = hidden  # what step 1 reads: the trunk's hidden states, [1, L, Dm]
        drafts, dists = [], []
        for k in range(1, self.depth + 1):
            # 1. `known` holds t_0 .. t_{L+k-1}, which is exactly what step k reads.
            new_hidden, logits = mtp.step(
                k, previous, known, token_emb, lm_head
            )  # [1, L, Dm], [1, V]
            # 2. DeepSeek depths are chained: the next depth reads this one's output. The offset-conditioned module
            #    reads the trunk's hidden states every time, so `previous` stays the same.
            if mtp.chained:
                previous = new_hidden
            dist = self._distribution(logits[0])
            draft = self._sample(dist)
            known = torch.cat([known, draft.view(1, 1)], dim=1)
            drafts.append(draft)
            dists.append(dist)
        return drafts, dists

    def _verify(self, drafts, draft_dists, trunk_logits: torch.Tensor):
        """Speculative sampling: returns how many drafts were kept and the next token (a sample from the trunk)."""
        for j, (draft, q) in enumerate(zip(drafts, draft_dists)):
            p = self._distribution(trunk_logits[j])
            accept_probability = torch.clamp(p[draft] / q[draft].clamp_min(1e-30), max=1.0)
            if torch.rand((), device=p.device) < accept_probability:
                continue
            # Rejected: draw from what the trunk wanted beyond what the draft already covered.
            residual = (p - q).clamp_min(0.0)
            total = residual.sum()
            return j, self._sample(residual / total if total > 0 else p)
        # Every draft was kept: the last row of the verify pass gives a free token.
        return len(drafts), self._sample(self._distribution(trunk_logits[len(drafts)]))

    # ----- helpers -----
    def _distribution(self, logits: torch.Tensor) -> torch.Tensor:
        """Probabilities over the vocabulary: a one-hot at the argmax when greedy, else softmax(logits / T)."""
        if self.greedy:
            return F.one_hot(logits.argmax(), logits.numel()).to(torch.float32)
        return F.softmax(logits.float() / self.temperature, dim=-1)

    def _sample(self, distribution: torch.Tensor) -> torch.Tensor:
        """One token id (a 0-d tensor) from a distribution."""
        if self.greedy:
            return distribution.argmax()
        return torch.multinomial(distribution, 1)[0]

    def _has_ended(self, generated: list[int]) -> bool:
        return self.eos_id is not None and self.eos_id in generated

    def acceptance_by_depth(self) -> list[float]:
        """For each depth, the share of its drafts that were kept, counting only rounds that reached it."""
        return [
            kept / reached if reached else 0.0
            for kept, reached in zip(self.depth_accepted, self.depth_reached)
        ]

    def _log_summary(self, produced: int) -> None:
        if self.rounds:
            logger.info(
                "speculative decoding: {} tokens in {} rounds after the first, {:.0%} of {} drafts accepted "
                "(per depth: {})",
                produced,
                self.rounds,
                self.drafts_accepted / max(self.drafts_proposed, 1),
                self.drafts_proposed,
                ", ".join(f"{rate:.0%}" for rate in self.acceptance_by_depth()),
            )
