"""Index audit: every token index used by MTP training and inference, written out and checked on sample data.

The trick: feed *identity tokens* (token value == its position in the sequence), so any id seen by a spy
is directly an index. The expected tables below are written by hand from the DeepSeek-V3 definition:

    depth k, position i : reads  Emb(t_{i+k})  and  h^{k-1}_i ,  predicts  t_{i+k+1}
"""

from __future__ import annotations

import pytest
import torch

import hale_llm.models  # noqa: F401
from hale_llm.config.schema import load_config
from hale_llm.models.models_mtp import loss as mtp_loss_module
from hale_llm.models.models_mtp.speculative import SpeculativeDecoder
from hale_llm.registry import get_loss, get_variant
from types import SimpleNamespace

NO_EOS = SimpleNamespace(eos_token_id=None, pad_token_id=None)


def _model(depth, max_length, vocab=40):
    cfg = load_config("configs/autoregressive_mtp/dense_nano.yaml")
    cfg.set_context_window(max_length)
    cfg.model.d_model, cfg.model.n_heads, cfg.model.n_layers, cfg.model.d_ff = 32, 4, 2, 64
    cfg.model.dropout, cfg.model.mtp_depth = 0.0, depth
    torch.manual_seed(0)
    return get_variant(cfg.variant)(vocab_size=vocab, cfg=cfg)


# ----- training: T = 9 tokens 0..8, D = 3 depths -----
def test_training_reads_and_predicts_exactly_these_indices():
    T, D = 9, 3
    model = _model(D, T).train()
    ids = torch.arange(T).view(1, T)  # identity tokens
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels, "pad_token_id": None}

    embedded_ids: list[list[int]] = []
    model.backbone.token_emb.register_forward_pre_hook(lambda m, args: embedded_ids.append(args[0].flatten().tolist()))
    trunk_out: list[torch.Tensor] = []
    model.backbone.norm.register_forward_hook(lambda m, args, out: trunk_out.append(out))  # the last step of the trunk
    seen_inputs, depth_out = [], []

    def record_depth(module, args, output):  # returns None: a forward hook that returns a value would replace the output
        seen_inputs.append(args[:2])
        depth_out.append(output)

    for module in model.mtp.depths:
        module.register_forward_hook(record_depth)
    target_calls: list[tuple[int, list[int]]] = []
    original = mtp_loss_module.model_token_nll

    def spy(model_, logits, targets, **kwargs):
        target_calls.append((logits.size(1), targets.flatten().tolist()))
        return original(model_, logits, targets, **kwargs)

    mtp_loss_module.model_token_nll = spy
    try:
        get_loss(model.cfg.variant)(model, batch)
    finally:
        mtp_loss_module.model_token_nll = original

    # 1. Which tokens are embedded: the trunk reads all of them, depth k reads t_k .. t_8.
    assert embedded_ids == [
        [0, 1, 2, 3, 4, 5, 6, 7, 8],  # trunk
        [1, 2, 3, 4, 5, 6, 7, 8],  # depth 1: Emb(t_{i+1}),  i = 0..7
        [2, 3, 4, 5, 6, 7, 8],  # depth 2: Emb(t_{i+2}),  i = 0..6
        [3, 4, 5, 6, 7, 8],  # depth 3: Emb(t_{i+3}),  i = 0..5
    ]
    # 2. Which hidden state each depth reads: positions 0 .. T-k-1 of the previous depth (the trunk for depth 1).
    assert torch.equal(seen_inputs[0][0], trunk_out[0][:, :8])  # h^0_i, i = 0..7
    assert torch.equal(seen_inputs[1][0], depth_out[0][:, :7])  # h^1_i, i = 0..6
    assert torch.equal(seen_inputs[2][0], depth_out[1][:, :6])  # h^2_i, i = 0..5
    # 3. And the embedding the module received is the table row of those tokens.
    table = model.backbone.token_emb.weight
    for k, (_, embedded) in enumerate(seen_inputs, start=1):
        assert torch.equal(embedded, table[torch.arange(k, T)].unsqueeze(0))
    # 4. What every logit row is scored against: row j of depth k is the prediction of t_{j+k+1}.
    assert target_calls == [
        (8, [1, 2, 3, 4, 5, 6, 7, 8]),  # trunk: position i predicts t_{i+1}
        (7, [2, 3, 4, 5, 6, 7, 8]),  # depth 1: position i predicts t_{i+2}   (last position has no target: dropped)
        (6, [3, 4, 5, 6, 7, 8]),  # depth 2: position i predicts t_{i+3}
        (5, [4, 5, 6, 7, 8]),  # depth 3: position i predicts t_{i+4}
    ]


@pytest.mark.parametrize("depth,T", [(1, 4), (2, 5), (4, 6), (4, 12)])  # including the shortest legal window, D + 2
def test_every_depth_has_at_least_one_target_in_the_shortest_legal_window(depth, T):
    model = _model(depth, T).train()
    ids = torch.arange(T).view(1, T)
    labels = ids.clone()
    labels[:, :-1], labels[:, -1] = ids[:, 1:], -100
    loss = get_loss(model.cfg.variant)(model, {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    assert torch.isfinite(loss) and len(model.loss_parts) == 3 + depth  # ntp, mtp mean, the weight, one per depth


# ----- inference: one speculative round, L = 6 committed tokens 0..5, D = 3 -----
def test_one_speculative_round_reads_and_verifies_exactly_these_indices():
    L, D = 6, 3
    model = _model(D, max_length=32).double().eval()
    decoder = SpeculativeDecoder(model, NO_EOS, 0.0)
    prompt = torch.arange(L).view(1, L)  # identity tokens 0..5

    encoded: list[list[int]] = []  # every trunk pass: the token ids it ran over
    original_encode = model.backbone.encode
    model.backbone.encode = lambda ids, *a, **k: (encoded.append(ids.flatten().tolist()), original_encode(ids, *a, **k))[1]
    embedded_ids: list[list[int]] = []
    model.backbone.token_emb.register_forward_pre_hook(lambda m, args: embedded_ids.append(args[0].flatten().tolist()))
    steps = []  # (depth k, rows of previous hidden, token ids handed to the depth, its hidden output)
    original_step = model.mtp.step

    def spy(k, previous_hidden, tokens, token_emb, lm_head):
        hidden, logits = original_step(k, previous_hidden, tokens, token_emb, lm_head)
        steps.append((k, previous_hidden, tokens.flatten().tolist(), hidden))
        return hidden, logits

    model.mtp.step = spy
    head_inputs: list[torch.Tensor] = []
    model.backbone.lm_head.register_forward_hook(lambda m, args, out: head_inputs.append(args[0]))

    output = decoder.generate(prompt, max_new_tokens=2)  # prompt pass + exactly one round
    pending = int(output[0, L])  # t_6, the trunk's own next token

    # 1. The prompt pass, then the verify pass over  t_0..t_5, t_6 (pending), d_1, d_2, d_3.
    d1, d2 = steps[1][2][-1], steps[2][2][-1]
    assert encoded[0] == [0, 1, 2, 3, 4, 5]  # trunk over the committed tokens
    assert encoded[1][: L + 1] == [0, 1, 2, 3, 4, 5, pending] and len(encoded[1]) == L + 1 + D  # then pending + 3 drafts
    assert encoded[1][L + 1 : L + 3] == [d1, d2]
    # 2. The draft chain: depth k is handed t_0 .. t_{L+k-1}  and the hidden states of L positions.
    assert [(k, prev.size(1), tokens) for k, prev, tokens, _ in steps] == [
        (1, L, [0, 1, 2, 3, 4, 5, pending]),
        (2, L, [0, 1, 2, 3, 4, 5, pending, d1]),
        (3, L, [0, 1, 2, 3, 4, 5, pending, d1, d2]),
    ]
    # 3. Depth 1 reads the trunk's hidden states; depth k reads depth k-1's output.
    assert torch.equal(steps[0][1], original_encode(prompt))
    assert torch.equal(steps[1][1], steps[0][3]) and torch.equal(steps[2][1], steps[1][3])
    # 4. The embeddings read: depth k embeds t_{i+k} for i = 0 .. L-1  (a window of L tokens that slides right).
    chain_embeddings = [ids for ids in embedded_ids if len(ids) == L and ids != [0, 1, 2, 3, 4, 5]]
    assert chain_embeddings == [
        [1, 2, 3, 4, 5, pending],  # depth 1
        [2, 3, 4, 5, pending, d1],  # depth 2
        [3, 4, 5, pending, d1, d2],  # depth 3
    ]
    # 5. Verification reads the trunk's hidden states at positions L .. L+D of the candidate text
    #    t_0..t_5, pending, d_1..d_3: row j (0-based) judges d_{j+1}, the last row gives the bonus token.
    candidate = torch.tensor([encoded[1]])
    verify_input = next(x for x in head_inputs if x.dim() == 3 and x.size(1) == D + 1)
    assert torch.equal(verify_input, original_encode(candidate)[:, L : L + D + 1])


# ----- the same convention at training and at inference, on the same numbers -----
@pytest.mark.parametrize("depth", [1, 2, 4])
def test_training_and_inference_compute_identical_logits_for_the_same_tokens(depth):
    """Run depth k both ways on a sequence of known tokens: the training pass and the inference step must agree."""
    L = 7
    model = _model(depth, max_length=32).double().eval()
    ids = torch.arange(L + depth).view(1, -1) % 40  # t_0 .. t_{L+D-1}
    trained_way = model.mtp(model.backbone.encode(ids), ids, None, model.backbone.token_emb, model.backbone.lm_head)
    previous = model.backbone.encode(ids[:, :L])
    for k in range(1, depth + 1):
        previous, logits = model.mtp.step(k, previous, ids[:, : L + k], model.backbone.token_emb, model.backbone.lm_head)
        assert torch.allclose(logits[0], trained_way[k - 1][0, L - 1], atol=1e-9), f"depth {k}"
