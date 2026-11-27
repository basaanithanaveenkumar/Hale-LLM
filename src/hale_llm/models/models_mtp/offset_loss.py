"""The MTP loss of the offset-conditioned predictor."""

from __future__ import annotations

import torch

from hale_llm.losses import model_token_nll


def offset_conditioned_mtp_loss(model, hidden: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The mean cross-entropy of the sampled-offset MTP module, and the offsets that were used.

    ``hidden`` is the trunk's final hidden states ``[B, T, D]``. Algorithm:

        1. draw one offset n_b in 1..mtp_depth for each of the B sequences (equal frequency over the run)
        2. run the MTP module: position t reads h(t), x(t + n_b) and the embedding of n_b -> scores for x(t + n_b + 1)
        3. build the targets x(t + n_b + 1), ignoring those past the end, padding, or in another document
        4. average the cross-entropy over all the targets that count (every sequence, every valid position)

    The result replaces ``mean_k(loss_k)`` of the DeepSeek-V3 depths, so ``lambda`` weights it the same way.
    """
    backbone, predictor = model.backbone, model.mtp
    ids, mask, labels = batch["input_ids"], batch.get("attention_mask"), batch["labels"]
    document_ids = batch.get("document_ids")  # only when model.document_masking is on

    # 1. One offset per sequence, from the balanced stream.
    offsets = predictor.sample_offsets(ids.size(0), ids.device)  # [B]
    # 2. Scores of every position at its sequence's offset.
    logits = predictor(hidden, ids, mask, backbone.token_emb, backbone.lm_head, offsets, document_ids)  # [B, T, V]
    # 3. What each position should have predicted (-100 where it does not count).
    targets = predictor.targets(labels, offsets, document_ids)  # [B, T]
    # 4. Mean over the targets that count.
    loss = model_token_nll(model, logits, targets, ignore_index=-100, reduction="mean")
    return loss, offsets
