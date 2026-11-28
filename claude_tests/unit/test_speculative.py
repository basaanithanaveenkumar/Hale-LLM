"""Speculative decoding with the MTP depths: it must change the speed, never the output."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import hale_llm.models  # noqa: F401  (registers the variants)
from hale_llm.config.schema import load_config
from hale_llm.models.models_autoregressive.sample import sample_autoregressive
from hale_llm.models.models_mtp.sample import sample_mtp
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.registry import get_variant

NO_EOS = SimpleNamespace(eos_token_id=None, pad_token_id=None)


def _model(depth=2, vocab=24, max_length=64, ffn_type="mlp", seed=0, sharpen=8.0, speculative=True, mtp_layers=1):
    """A random float64 model; the embedding is scaled up so its distributions are peaked and distinct."""
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.set_context_window(max_length)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.ffn_type, cfg.model.moe_hidden_dim, cfg.model.moe_router = 0.0, ffn_type, 48, "topk"
    cfg.model.mtp_depth, cfg.model.mtp_n_layers = depth, mtp_layers
    cfg.sample.speculative = speculative
    torch.manual_seed(seed)
    model = get_variant(cfg.variant)(vocab_size=vocab, cfg=cfg)
    model.backbone.token_emb.weight.data *= sharpen
    return model.double().eval()


def _prompt(model, n=5, seed=1):
    return torch.randint(0, model.vocab_size, (1, n), generator=torch.Generator().manual_seed(seed))


PATTERN = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]  # digits of pi: 1 and 5 repeat with different successors,
PERIOD = len(PATTERN)  # so the next token depends on more than the last token and a corrupted context shows


def _cycle(offset: int, length: int, pattern=PATTERN) -> torch.Tensor:
    """The pattern repeated: a small model learns it exactly, with 7 distinct tokens."""
    return torch.tensor([[pattern[(offset + i) % len(pattern)] for i in range(length)]])


_TRAINED: dict = {}


def _trained(depth=2, ffn_type="mlp", pattern=PATTERN, mtp_layers=1):
    """A small model trained on the pattern: its greedy output is varied and its MTP drafts are good.

    Random models are useless here: they always collapse to one repeated token, so wrong context, wrong
    rows and wrong bookkeeping all give the same output and no test can tell them apart.
    """
    key = (depth, ffn_type, tuple(pattern), mtp_layers)
    if key not in _TRAINED:
        from hale_llm.registry import get_loss

        model = _model(depth=depth, ffn_type=ffn_type, sharpen=1.0, mtp_layers=mtp_layers).float().train()
        model.cfg.model.moe_router = "topk"
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
        generator = torch.Generator().manual_seed(0)
        loss_fn = get_loss(model.cfg.variant)
        for _ in range(500):
            offsets = torch.randint(0, len(pattern), (16,), generator=generator)
            ids = torch.cat([_cycle(int(o), 24, pattern) for o in offsets])
            labels = ids.clone()
            labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
            optimizer.zero_grad()
            loss_fn(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}).backward()
            optimizer.step()
        _TRAINED[key] = model.double().eval()
    return _TRAINED[key]


def _plain(model, prompt, n):
    return sample_autoregressive(model, NO_EOS, prompt, n, temperature=0.0)


def _assert_varied(output, prompt):
    assert len(set(output[0, prompt.size(1):].tolist())) >= 5, "the model's output collapsed, the test would prove nothing"


# ----- exactness at temperature 0 -----
CASES = [(1, "mlp"), (2, "mlp"), (3, "mlp"), (2, "moe")]


@pytest.mark.parametrize("depth,ffn_type", CASES)
@pytest.mark.parametrize("offset", [0, 3, 8])
def test_greedy_speculative_decoding_equals_plain_greedy_decoding(depth, ffn_type, offset):
    model = _trained(depth, ffn_type)
    prompt = _cycle(offset, 6)
    plain = _plain(model, prompt, 30)
    _assert_varied(plain, prompt)
    assert torch.equal(plain, SpeculativeDecoder(model, NO_EOS, 0.0).generate(prompt, 30))


@pytest.mark.parametrize("offset", [0, 4])
def test_greedy_speculative_decoding_is_exact_with_several_blocks_per_mtp_depth(offset):
    model, prompt = _trained(depth=2, mtp_layers=2), _cycle(offset, 6)
    plain = _plain(model, prompt, 30)
    _assert_varied(plain, prompt)
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    assert torch.equal(plain, decoder.generate(prompt, 30))  # exactness does not depend on how good the drafts are
    assert decoder.drafts_accepted / decoder.drafts_proposed > 0.7  # the tiny two-block depth trains a little less well


def test_the_public_sampler_gives_the_same_tokens_as_the_decoder():
    model, prompt = _trained(2), _cycle(2, 6)
    assert torch.equal(sample_mtp(model, NO_EOS, prompt, 25, temperature=0.0), _plain(model, prompt, 25))


@pytest.mark.parametrize("depth,ffn_type", CASES)
def test_drafts_from_the_real_mtp_depths_are_mostly_accepted_and_save_trunk_passes(depth, ffn_type):
    model = _trained(depth, ffn_type)
    prompt = _cycle(4, 6)
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    output = decoder.generate(prompt, 40)
    assert torch.equal(output, _plain(model, prompt, 40))
    assert decoder.drafts_accepted / decoder.drafts_proposed > (0.9 if ffn_type == "mlp" else 0.7)  # the tiny MoE trains less well
    assert decoder.rounds <= math.ceil(39 / (depth + 1)) + 2  # about D + 1 tokens per trunk pass instead of 1


class WrongSometimes(SpeculativeDecoder):
    """Trunk-greedy drafts, but every other draft is replaced by a wrong token: accepts, then a rejection mid-chain."""

    def _draft(self, tokens, hidden, pending):
        sequence = torch.cat([tokens, pending.view(1, 1)], dim=1)
        drafts, dists = [], []
        for _ in range(self.depth):
            logits = self.backbone.lm_head(self.backbone.encode(sequence))[0, -1]
            token = logits.argmax()
            if (self.rounds + len(drafts)) % 2 == 1:
                token = (token + 1) % logits.numel()
            drafts.append(token)
            dists.append(F.one_hot(token, logits.numel()).to(torch.float32))
            sequence = torch.cat([sequence, token.view(1, 1)], dim=1)
        return drafts, dists


class AlwaysRightDraft(SpeculativeDecoder):
    """Drafts exactly what the trunk would choose, so every draft must be accepted (and the bonus token is used)."""

    def _draft(self, tokens, hidden, pending):
        sequence = torch.cat([tokens, pending.view(1, 1)], dim=1)
        drafts, dists = [], []
        for _ in range(self.depth):
            logits = self.backbone.lm_head(self.backbone.encode(sequence))[0, -1]
            token = logits.argmax()
            drafts.append(token)
            dists.append(F.one_hot(token, logits.numel()).to(torch.float32))
            sequence = torch.cat([sequence, token.view(1, 1)], dim=1)
        return drafts, dists


class AlwaysWrongDraft(SpeculativeDecoder):
    """Drafts a token the trunk would never choose, so every draft must be rejected."""

    def _draft(self, tokens, hidden, pending):
        sequence = torch.cat([tokens, pending.view(1, 1)], dim=1)
        drafts, dists = [], []
        for _ in range(self.depth):
            logits = self.backbone.lm_head(self.backbone.encode(sequence))[0, -1]
            token = (logits.argmax() + 1) % logits.numel()
            drafts.append(token)
            dists.append(F.one_hot(token, logits.numel()).to(torch.float32))
            sequence = torch.cat([sequence, token.view(1, 1)], dim=1)
        return drafts, dists


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_perfect_drafts_are_all_accepted(depth):
    model, prompt = _trained(depth), _cycle(1, 6)
    decoder = AlwaysRightDraft(model, NO_EOS, 0.0)
    output = decoder.generate(prompt, 21)
    _assert_varied(output, prompt)
    assert torch.equal(output, _plain(model, prompt, 21))
    assert decoder.drafts_accepted == decoder.drafts_proposed and decoder.rounds <= math.ceil(20 / (depth + 1))


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_terrible_drafts_are_all_rejected_and_the_output_is_still_exact(depth):
    model, prompt = _trained(depth), _cycle(1, 6)
    decoder = AlwaysWrongDraft(model, NO_EOS, 0.0)
    output = decoder.generate(prompt, 21)
    _assert_varied(output, prompt)
    assert torch.equal(output, _plain(model, prompt, 21))
    assert decoder.drafts_accepted == 0 and decoder.rounds == 20  # one token per round, like plain decoding


@pytest.mark.parametrize("depth", [2, 3])
def test_partly_wrong_drafts_still_give_the_exact_output(depth):
    model, prompt = _trained(depth), _cycle(5, 6)
    decoder = WrongSometimes(model, NO_EOS, 0.0)
    output = decoder.generate(prompt, 40)
    _assert_varied(output, prompt)
    assert torch.equal(output, _plain(model, prompt, 40))
    assert 0 < decoder.drafts_accepted < decoder.drafts_proposed  # both paths ran


class CheckedRounds(SpeculativeDecoder):
    """After every round, the carried state must be what a fresh computation would give."""

    def _round(self, tokens, hidden, pending):
        tokens, hidden, extra, new_tokens = super()._round(tokens, hidden, pending)
        plain = _plain(self.model, self.prompt, 50)[0]  # 6 + 50 tokens stay inside the 64-token window
        n = self.prompt.size(1)
        assert hidden.shape[1] == tokens.size(1), "one trunk hidden state per committed token"
        assert torch.allclose(hidden, self.backbone.encode(tokens), atol=1e-9), "carried hidden states are stale"
        assert torch.equal(tokens[0, n:], plain[n : tokens.size(1)]), "a rejected draft was committed"
        assert int(extra) == int(plain[tokens.size(1)]), "the pending token is not the trunk's next token"
        return tokens, hidden, extra, new_tokens


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_state_carried_between_rounds_is_exactly_what_a_fresh_pass_would_give(depth):
    model, prompt = _trained(depth), _cycle(2, 6)
    decoder = CheckedRounds(model, NO_EOS, 0.0)
    decoder.prompt = prompt
    decoder.generate(prompt, 30)
    assert decoder.rounds > 3


# ----- the draft chain at inference is the same computation as at training -----
@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_draft_chain_gives_what_the_training_forward_pass_gives(depth):
    """Teacher forcing: feed the chain the true tokens and compare with MTPs forward over the whole sequence."""
    model = _trained(depth)
    ids = _cycle(3, 16)
    L = 9
    hidden_full = model.backbone.encode(ids)
    expected = model.mtp(hidden_full, ids, None, model.backbone.token_emb, model.backbone.lm_head)
    previous = model.backbone.encode(ids[:, :L])
    for k in range(1, depth + 1):
        previous, logits = model.mtp.step(k, previous, ids[:, : L + k], model.backbone.token_emb, model.backbone.lm_head)
        # Depth k, position L-1 predicts t_{L+k}: the same value the training pass produces there.
        assert torch.allclose(logits[0], expected[k - 1][0, L - 1], atol=1e-8), k


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_the_decoders_draft_distributions_match_the_training_pass_on_the_drafted_tokens(depth):
    model = _trained(depth)
    decoder = SpeculativeDecoder(model, NO_EOS, 1.0)  # temperature 1: the distributions are real softmaxes
    tokens, pending = _cycle(0, 8), torch.tensor(PATTERN[8 % PERIOD])
    torch.manual_seed(0)
    drafts, dists = decoder._draft(tokens, model.backbone.encode(tokens), pending)
    extended = torch.cat([tokens, pending.view(1, 1), torch.stack(drafts).view(1, -1)], dim=1)  # t_0 .. t_{L+D}
    L = tokens.size(1)
    expected = model.mtp(model.backbone.encode(extended), extended, None, model.backbone.token_emb, model.backbone.lm_head)
    for k in range(1, depth + 1):
        assert torch.allclose(dists[k - 1].double(), F.softmax(expected[k - 1][0, L - 1], dim=-1), atol=1e-6), k


def test_it_stops_at_the_end_of_text_token_exactly_like_plain_decoding():
    model, prompt = _trained(2), _cycle(0, 6)
    free_run = _plain(model, prompt, 30)
    eos = int(free_run[0, prompt.size(1) + 6])  # a token the model emits at generation step 7
    tokenizer = SimpleNamespace(eos_token_id=eos, pad_token_id=None)
    plain = sample_autoregressive(model, tokenizer, prompt, 30, temperature=0.0)
    speculative = SpeculativeDecoder(model, tokenizer, 0.0).generate(prompt, 30)
    assert torch.equal(plain, speculative)
    assert int(speculative[0, -1]) == eos and eos not in speculative[0, prompt.size(1):-1].tolist()


@pytest.mark.parametrize("new_tokens", [1, 2, 3, 4, 5])
def test_the_requested_length_is_respected(new_tokens):
    model = _model(depth=3)
    prompt = _prompt(model)
    assert SpeculativeDecoder(model, NO_EOS, 0.0).generate(prompt, new_tokens).shape == (1, prompt.size(1) + new_tokens)


def test_the_context_window_slides_without_breaking():
    model = _model(depth=2, max_length=16)
    prompt = _prompt(model, n=10)
    output = SpeculativeDecoder(model, NO_EOS, 0.0).generate(prompt, 25)  # 35 tokens in a 16-token window
    assert output.shape == (1, 35) and torch.equal(output[:, :10], prompt) and int(output.max()) < model.vocab_size


def test_history_tracks_progress():
    model = _model()
    prompt = _prompt(model)
    output, history = SpeculativeDecoder(model, NO_EOS, 0.0).generate(prompt, 12, return_history=True)
    fractions = [fraction for fraction, _ in history]
    assert fractions[0] == 0.0 and fractions == sorted(fractions) and fractions[-1] == 1.0
    assert torch.equal(history[0][1], prompt) and torch.equal(history[-1][1], output)


# ----- exactness in distribution at temperature > 0 -----
def test_sampling_has_the_same_distribution_as_sampling_from_the_trunk():
    """Joint distribution of 3 sampled tokens: the exact trunk product against 4000 speculative samples."""
    vocab, n_new, samples = 4, 3, 4000
    model = _model(depth=2, vocab=vocab, sharpen=6.0, seed=3)
    prompt = _prompt(model, n=4)

    def trunk_probs(ids):
        return F.softmax(model.backbone.lm_head(model.backbone.encode(ids))[0, -1], dim=-1)

    exact = torch.zeros(vocab, vocab, vocab, dtype=torch.float64)
    for a in range(vocab):
        pa = trunk_probs(prompt)[a]
        ids_a = torch.cat([prompt, torch.tensor([[a]])], dim=1)
        for b in range(vocab):
            pb = trunk_probs(ids_a)[b]
            ids_ab = torch.cat([ids_a, torch.tensor([[b]])], dim=1)
            exact[a, b] = pa * pb * trunk_probs(ids_ab)

    torch.manual_seed(0)
    counts = torch.zeros(vocab, vocab, vocab, dtype=torch.float64)
    for _ in range(samples):
        a, b, c = SpeculativeDecoder(model, NO_EOS, 1.0).generate(prompt, n_new)[0, prompt.size(1):].tolist()
        counts[a, b, c] += 1
    total_variation = 0.5 * (counts / samples - exact).abs().sum().item()
    assert total_variation < 0.08, f"total variation {total_variation:.3f}"


# ----- fallbacks -----
def test_several_prompts_fall_back_to_plain_sampling():
    model = _model()
    prompts = torch.randint(0, model.vocab_size, (3, 5))
    assert sample_mtp(model, NO_EOS, prompts, 6, temperature=0.0).shape == (3, 11)


def test_speculative_can_be_switched_off_in_the_config():
    model = _model(speculative=False)
    prompt = _prompt(model)
    torch.manual_seed(5)
    off = sample_mtp(model, NO_EOS, prompt, 10, temperature=1.0)
    torch.manual_seed(5)
    assert torch.equal(off, sample_autoregressive(model, NO_EOS, prompt, 10, temperature=1.0))  # literally the plain sampler


def test_a_context_window_too_short_for_the_depth_is_rejected_when_the_model_is_built():
    with pytest.raises(ValueError, match="context window of at least 5"):
        _model(depth=3, max_length=4)
    _model(depth=3, max_length=5)  # the smallest legal window: room for exactly one committed token


def test_the_decoder_works_in_the_smallest_legal_window():
    model = _model(depth=3, max_length=5)
    output = SpeculativeDecoder(model, NO_EOS, 0.0).generate(_prompt(model, n=3), 8)
    assert output.shape == (1, 11)


def test_acceptance_is_reported_per_depth():
    model, prompt = _trained(3), _cycle(4, 6)
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    decoder.generate(prompt, 40)
    rates = decoder.acceptance_by_depth()
    assert len(rates) == 3 and all(0.0 <= rate <= 1.0 for rate in rates)
    assert decoder.depth_reached[0] == decoder.rounds  # depth 1 is reached in every round
    assert sum(decoder.depth_accepted) == decoder.drafts_accepted
    assert all(later <= earlier for earlier, later in zip(decoder.depth_reached, decoder.depth_reached[1:]))  # reached only after the earlier ones were kept


# ----- the worked example from docs/mtp.md: "The cat sat on" -----
WORDS = ["The", "cat", "sat", "on", "the", "mat", "."]  # token ids 0..6; every word has a single successor
SENTENCE = list(range(len(WORDS)))


class _CountPasses:
    """Counts the main model's passes: every call of ``backbone.encode`` reads the whole text once."""

    def __init__(self, model):
        self.texts: list[list[str]] = []
        original = model.backbone.encode
        model.backbone.encode = lambda ids, *a, **k: (self.texts.append([WORDS[i] for i in ids[0].tolist()]), original(ids, *a, **k))[1]


class ConstantWrongDraft(SpeculativeDecoder):
    """Always guesses "." (id 6), which is never the right next word in this sentence, and needs no extra pass."""

    def _draft(self, tokens, hidden, pending):
        wrong = torch.tensor(6)
        vocab = self.backbone.token_emb.num_embeddings  # the model's vocabulary is larger than the 7 words
        return [wrong] * self.depth, [F.one_hot(wrong, vocab).to(torch.float32)] * self.depth


def _fresh_copy(model):
    """The trained model as a new object, so a spy installed on one test never leaks into another."""
    import copy

    return copy.deepcopy(model)


def test_the_cat_sat_on_example_takes_two_main_model_passes_instead_of_three():
    model = _fresh_copy(_trained(depth=1, pattern=SENTENCE))
    passes = _CountPasses(model)
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    output = decoder.generate(torch.tensor([[0]]), 3)  # the prompt is "The"
    assert [WORDS[i] for i in output[0].tolist()] == ["The", "cat", "sat", "on"]
    assert passes.texts == [
        ["The"],  # pass 1: the main model reads "The" and says "cat" (the MTP module then guesses "sat")
        ["The", "cat", "sat"],  # pass 2: checks the guess at "cat", and gives "on" at "sat" as a bonus
    ]
    assert decoder.drafts_proposed == 1 and decoder.drafts_accepted == 1


def test_a_wrong_guess_costs_no_extra_passes_and_never_changes_the_text():
    model = _fresh_copy(_trained(depth=1, pattern=SENTENCE))
    passes = _CountPasses(model)
    decoder = ConstantWrongDraft(model, NO_EOS, 0.0)  # every guess is "." and is wrong here
    output = decoder.generate(torch.tensor([[0]]), 3)
    assert [WORDS[i] for i in output[0].tolist()] == ["The", "cat", "sat", "on"]  # the same words as before
    assert len(passes.texts) == 3  # three passes: exactly what normal decoding needs (one per word)
    assert decoder.drafts_accepted == 0


def test_normal_decoding_needs_one_pass_per_word():
    model = _fresh_copy(_trained(depth=1, pattern=SENTENCE))
    passes = _CountPasses(model)
    plain = sample_autoregressive(model, NO_EOS, torch.tensor([[0]]), 3, temperature=0.0)
    assert [WORDS[i] for i in plain[0].tolist()] == ["The", "cat", "sat", "on"]
    assert len(passes.texts) == 3


def test_with_four_guesses_one_pass_can_produce_several_words():
    """The first wrong guess ends the accepted run: the words before it are kept, the wrong one is replaced."""
    model = _fresh_copy(_trained(depth=4, pattern=SENTENCE))
    passes = _CountPasses(model)
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    output = decoder.generate(torch.tensor([[0]]), 5)  # "The" -> cat sat on the mat
    assert [WORDS[i] for i in output[0].tolist()] == ["The", "cat", "sat", "on", "the", "mat"]
    assert len(passes.texts) == 2  # 5 new words from 2 main-model passes (normal decoding: 5 passes)
