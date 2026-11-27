"""Which document of a packed window each token belongs to."""

from __future__ import annotations

import torch


def document_ids_from_eos(input_ids: torch.Tensor, eos_id: int) -> torch.Tensor:
    """``[B, S]`` ids -> ``[B, S]`` document numbers, counting from 0 inside each window.

    The end-of-text token belongs to the document it ends; the next token starts a new one. Example with
    ``eos_id = 9``::

        input_ids     [ 4  5  9  7  8  9  3 ]
        document_ids  [ 0  0  0  1  1  1  2 ]

    A window that starts in the middle of a document simply has that partial document as number 0.
    """
    is_eos = (input_ids == eos_id).long()
    return torch.cumsum(is_eos, dim=1) - is_eos  # EOS tokens strictly before each position
