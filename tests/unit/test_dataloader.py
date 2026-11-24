"""Packing documents into non-overlapping context windows, with EOS as the document separator."""

import pytest
import torch

from hale_llm.dataloader.context_windows import encode_documents, pack_context_windows
from hale_llm.dataloader.padding import attention_mask_for, pad_positions
from hale_llm.inference.stop import cut_at_stop

EOS = 9


class _WordTok:
    """One token per word: the id is the word's length (so 'aa bbb' -> [2, 3]); EOS is 9."""

    eos_token_id = EOS

    def __call__(self, text, add_special_tokens=False, truncation=False, padding=False, return_tensors=None):
        return {"input_ids": [len(w) for w in text.split()]}


def test_eos_follows_every_document():
    ids = encode_documents(_WordTok(), ["aa bbb", "c"])
    assert ids == [2, 3, EOS, 1, EOS]


def test_empty_documents_are_skipped():
    assert encode_documents(_WordTok(), ["", "   ", "a"]) == [1, EOS]


def test_windows_do_not_overlap_and_cover_the_stream_in_order():
    docs = ["a " * 5, "bb " * 5]  # 5 + EOS, 5 + EOS = 12 tokens
    windows = pack_context_windows(_WordTok(), docs, context_window=4)
    assert windows.shape == (3, 4)
    assert windows.flatten().tolist() == [1, 1, 1, 1, 1, EOS, 2, 2, 2, 2, 2, EOS]


def test_leftover_tokens_are_dropped():
    windows = pack_context_windows(_WordTok(), ["a a a a a a a"], context_window=4)  # 7 + EOS = 8
    assert windows.shape == (2, 4)
    windows = pack_context_windows(_WordTok(), ["a a a a a a"], context_window=4)  # 6 + EOS = 7
    assert windows.shape == (1, 4)


def test_too_little_text_gives_zero_windows():
    assert pack_context_windows(_WordTok(), ["a"], context_window=8).shape == (0, 8)


def test_tokenizer_without_eos_is_rejected():
    class NoEos(_WordTok):
        eos_token_id = None

    with pytest.raises(ValueError, match="EOS"):
        encode_documents(NoEos(), ["a"])


def test_context_window_must_be_at_least_two():
    with pytest.raises(ValueError):
        pack_context_windows(_WordTok(), ["a b"], context_window=1)


def test_no_pad_id_means_nothing_is_padding():
    ids = torch.tensor([[5, 0, 7]])
    assert not pad_positions(ids, None).any()
    assert attention_mask_for(ids, None).tolist() == [[1, 1, 1]]
    assert attention_mask_for(ids, 0).tolist() == [[1, 0, 1]]


def test_cut_at_stop_keeps_prompt_and_drops_everything_after_eos():
    ids = torch.tensor([4, EOS, 5, 6, EOS, 7])  # prompt = first 2 tokens (contains an EOS)
    assert cut_at_stop(ids, 2, EOS).tolist() == [4, EOS, 5, 6]
    assert cut_at_stop(ids, 2, None).tolist() == ids.tolist()
    assert cut_at_stop(torch.tensor([4, 5, 6]), 1, EOS).tolist() == [4, 5, 6]
