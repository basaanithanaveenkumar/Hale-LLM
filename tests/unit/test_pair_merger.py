"""PairMerger (SuperBPE stage 2) against a slow, obviously-correct reference."""

import random
from collections import Counter

from hale_llm.tokenizer.training.pair_merger import PairMerger


def naive_merges(documents, vocab, n_new, min_frequency=2):
    """Recount every pair after every merge. Slow but easy to trust."""
    id_to_token = list(vocab)
    token_to_id = {t: i for i, t in enumerate(id_to_token)}
    docs = [list(d) for d in documents]
    merges, created = [], 0
    while created < n_new:
        counts = Counter()
        for doc in docs:
            counts.update(zip(doc, doc[1:]))
        counts = {p: c for p, c in counts.items() if c >= min_frequency}
        if not counts:
            break
        pair = min(counts, key=lambda p: (-counts[p], p))
        token = id_to_token[pair[0]] + id_to_token[pair[1]]
        if token not in token_to_id:
            token_to_id[token] = len(id_to_token)
            id_to_token.append(token)
            created += 1
        merges.append((id_to_token[pair[0]], id_to_token[pair[1]]))
        new_id = token_to_id[token]
        rewritten = []
        for doc in docs:
            out, i = [], 0
            while i < len(doc):
                if i + 1 < len(doc) and (doc[i], doc[i + 1]) == pair:
                    out.append(new_id)
                    i += 2
                else:
                    out.append(doc[i])
                    i += 1
            rewritten.append(out)
        docs = rewritten
    return merges, id_to_token


def test_most_frequent_pair_is_merged_first():
    merger = PairMerger([[0, 1, 0, 1, 0, 1]], ["a", "b"])
    merges = merger.learn(1)
    assert merges == [("a", "b")]
    assert merger.vocabulary == ["a", "b", "ab"]


def test_documents_are_never_joined():
    merger = PairMerger([[0], [1], [0], [1]], ["a", "b"])
    assert merger.learn(5) == []


def test_pairs_below_min_frequency_are_not_merged():
    assert PairMerger([[0, 1]], ["a", "b"], min_frequency=2).learn(5) == []
    assert PairMerger([[0, 1]], ["a", "b"], min_frequency=1).learn(5) == [("a", "b")]


def test_overlapping_run_is_merged_left_to_right():
    merger = PairMerger([[0, 0, 0, 0]], ["a"])
    assert merger.learn(5) == [("a", "a")]  # (a,a)+(a,a) occurs once afterwards: below threshold
    assert "aa" in merger.vocabulary


def test_matches_the_slow_reference_on_random_text():
    rng = random.Random(7)
    vocab = list("abcd")
    for trial in range(30):
        docs = [[rng.randrange(4) for _ in range(rng.randrange(1, 40))] for _ in range(rng.randrange(1, 8))]
        expected, expected_vocab = naive_merges(docs, vocab, 12)
        merger = PairMerger(docs, vocab)
        assert merger.learn(12) == expected, trial
        assert merger.vocabulary == expected_vocab, trial


def test_stops_after_the_requested_number_of_new_tokens():
    merger = PairMerger([[0, 1] * 50], ["a", "b"])
    merger.learn(2)
    assert len(merger.vocabulary) == 4
