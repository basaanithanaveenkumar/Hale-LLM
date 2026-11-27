"""The MTP loss of the offset-conditioned predictor."""

from __future__ import annotations

import torch

from hale_llm.losses import model_token_nll
from hale_llm.nn.mtp import OffsetShare


def offset_conditioned_mtp_loss(model, hidden: torch.Tensor, batch: dict) -> tuple[torch.Tensor, OffsetShare]:
    """The stratified MTP loss of one micro-batch, and the offsets it used.

    ``hidden`` is the trunk's final hidden states ``[B, T, D]``. Algorithm:

        1. take this micro-batch's slice of the step's offset plan (stratified over the whole global batch)
        2. run the MTP module: position t reads h(t), x(t + n_b) and the embedding of n_b -> scores for x(t + n_b + 1)
        3. build the targets x(t + n_b + 1), ignoring those past the end, padding, or in another document
        4. per sequence s: l_s = mean cross-entropy over its valid positions
        5. per offset n: L_n = mean of l_s over the sequences of the WHOLE step that got offset n
        6. the MTP term is the SUM  L_1 + L_2 + ... + L_N  (the offsets add up, they are not averaged;
           ``model.mtp_loss_reduction: mean`` divides by N)

    Step 5 is done across micro-batches and ranks without seeing them: sequence s contributes
    ``l_s / S_n`` where ``S_n`` is how many sequences of the whole step have its offset. The trainer averages the
    micro-batch losses over ``micro_batches`` (= accumulation steps x ranks), so each term is multiplied by that number
    here to undo the average. Summed over the step this is exactly ``L_1 + ... + L_N``, with a micro-batch of 1 or of 64.
    """
    backbone, predictor = model.backbone, model.mtp
    ids, mask, labels = batch["input_ids"], batch.get("attention_mask"), batch["labels"]
    document_ids = batch.get("document_ids")  # only when model.document_masking is on

    # 1. This micro-batch's share of the step's offsets.
    share = predictor.next_offsets(ids.size(0), ids.device)  # offsets [B], global_counts [N], micro_batches
    # 2. Scores of every position at its sequence's offset.
    logits = predictor(hidden, ids, mask, backbone.token_emb, backbone.lm_head, share.offsets, document_ids)  # [B, T, V]
    # 3. What each position should have predicted (-100 where it does not count).
    targets = predictor.targets(labels, share.offsets, document_ids)  # [B, T]
    # 4. Mean cross-entropy of each sequence over the targets that count.
    nll = model_token_nll(model, logits, targets, ignore_index=-100, reduction="none")  # [B, T]
    valid = (targets != -100).to(nll.dtype)  # [B, T]
    per_sequence = (nll * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)  # [B]
    # 5. Weight of each sequence: micro_batches / S_n, so that the sum over the step is L_1 + ... + L_N.
    sequences_with_my_offset = share.global_counts[share.offsets - 1].to(per_sequence.dtype)  # [B]
    weight = share.micro_batches / sequences_with_my_offset  # [B]
    loss = (weight * per_sequence).sum()
    # 6. "mean" instead of "sum": divide by the number of offsets.
    if model.cfg.model.mtp_reduction() == "mean":
        loss = loss / predictor.depth
    return loss, share
