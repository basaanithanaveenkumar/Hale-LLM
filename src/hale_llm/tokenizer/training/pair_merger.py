"""Learn extra BPE merges on top of an existing vocabulary (stage 2 of SuperBPE)."""

from __future__ import annotations

import heapq
from collections import Counter, defaultdict


class PairMerger:
    """Greedy byte-pair merging over token-id sequences, with incremental updates.

    The naive algorithm recounts every adjacent pair after each merge (too slow). This
    one keeps three structures so a merge only touches the places where it happens:

    * a **doubly linked list** over all tokens (``prev`` / ``nxt`` arrays), so two
      neighbours can be fused in O(1) without shifting a list;
    * ``pair_count``: how often each adjacent pair occurs, updated only around a merge;
    * ``pair_positions``: where each pair occurs (a list of left positions; entries can
      go stale and are re-checked before use) and a **max-heap** of ``(-count, pair)`` with
      lazy deletion, so the best pair comes out in O(log n) without a full scan.

    Documents are separated by a gap in the linked list (``prev``/``nxt`` of -1), so no
    merge ever joins two documents.
    """

    def __init__(
        self,
        token_documents: list[list[int]],
        id_to_token: list[str],
        min_frequency: int = 2,
    ) -> None:
        # Copy: the table grows as new tokens are created.
        self._id_to_token = list(id_to_token)
        self._token_to_id = {token: i for i, token in enumerate(self._id_to_token)}
        self._min_frequency = max(1, min_frequency)

        # Flatten all documents into arrays; the first/last token of each document links to nothing.
        self._tokens: list[int] = []
        self._prev: list[int] = []
        self._next: list[int] = []
        for document in token_documents:
            start = len(self._tokens)
            for offset, token in enumerate(document):
                self._tokens.append(token)
                self._prev.append(start + offset - 1 if offset > 0 else -1)
                self._next.append(start + offset + 1 if offset < len(document) - 1 else -1)

        # Count every adjacent pair once and remember where it occurs.
        self._pair_count: Counter[tuple[int, int]] = Counter()
        self._pair_positions: dict[tuple[int, int], list[int]] = defaultdict(list)
        for position, token in enumerate(self._tokens):
            following = self._next[position]
            if following != -1:
                pair = (token, self._tokens[following])
                self._pair_count[pair] += 1
                self._pair_positions[pair].append(position)
        self._heap = [(-count, pair) for pair, count in self._pair_count.items()]
        heapq.heapify(self._heap)

    @property
    def vocabulary(self) -> list[str]:
        """Every token string so far, indexed by id (stage-1 tokens first, new ones after)."""
        return self._id_to_token

    def learn(self, max_new_tokens: int) -> list[tuple[str, str]]:
        """Run merges until ``max_new_tokens`` new tokens exist or no pair is frequent enough.

        Returns the merges in the order learned, each as ``(left_token, right_token)``.
        """
        merges: list[tuple[str, str]] = []
        created = 0
        while created < max_new_tokens:
            pair = self._pop_best_pair()
            if pair is None:
                break
            left, right = pair
            merged_token = self._id_to_token[left] + self._id_to_token[right]
            is_new = merged_token not in self._token_to_id
            if is_new:
                self._token_to_id[merged_token] = len(self._id_to_token)
                self._id_to_token.append(merged_token)
                created += 1
            merges.append((self._id_to_token[left], self._id_to_token[right]))
            self._apply_merge(pair, self._token_to_id[merged_token])
        return merges

    def _pop_best_pair(self) -> tuple[int, int] | None:
        """Most frequent pair still above the threshold, skipping out-of-date heap entries."""
        while self._heap:
            neg_count, pair = heapq.heappop(self._heap)
            current = self._pair_count.get(pair, 0)
            if current < self._min_frequency:
                continue  # fell below the threshold since it was pushed
            if current != -neg_count:
                heapq.heappush(self._heap, (-current, pair))  # count changed: re-queue with the true value
                continue
            return pair
        return None

    def _apply_merge(self, pair: tuple[int, int], new_id: int) -> None:
        """Fuse every occurrence of ``pair`` into ``new_id`` and fix the neighbouring pair counts."""
        left_id, right_id = pair
        tokens, prev, nxt = self._tokens, self._prev, self._next
        touched: set[tuple[int, int]] = set()

        for position in self._pair_positions.pop(pair, []):
            # The list may hold stale entries: re-check the pair is still really here.
            if tokens[position] != left_id:
                continue
            right_pos = nxt[position]
            if right_pos == -1 or tokens[right_pos] != right_id:
                continue

            before = prev[position]
            after = nxt[right_pos]

            # The pairs that touched the old tokens disappear...
            if before != -1:
                self._pair_count[(tokens[before], left_id)] -= 1
            if after != -1:
                self._pair_count[(right_id, tokens[after])] -= 1

            # ...the two tokens become one, and the linked list skips the removed slot.
            tokens[position] = new_id
            nxt[position] = after
            if after != -1:
                prev[after] = position
            tokens[right_pos] = -1  # dead slot (never matches a real id)

            # ...and the pairs around the new token appear.
            if before != -1:
                new_pair = (tokens[before], new_id)
                self._pair_count[new_pair] += 1
                self._pair_positions[new_pair].append(before)
                touched.add(new_pair)
            if after != -1:
                new_pair = (new_id, tokens[after])
                self._pair_count[new_pair] += 1
                self._pair_positions[new_pair].append(position)
                touched.add(new_pair)

        self._pair_count.pop(pair, None)
        for new_pair in touched:
            count = self._pair_count[new_pair]
            if count > 0:
                heapq.heappush(self._heap, (-count, new_pair))
