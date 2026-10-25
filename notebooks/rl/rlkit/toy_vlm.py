"""A tiny LLaVA-style vision-language model + a synthetic "ShapeWorld" for VLM RL notebooks.

Architecture (identical in spirit to LLaVA / Qwen-VL, just 1000x smaller)::

    image (3, 16, 16) --patchify 4x4--> 16 patches --linear "vision encoder"--> 16 image tokens
                                                                  |
                                       projector (MLP)  ----------+
                                                                  v
    [img_1 ... img_16] [<bos> q u e s t i o n] -> TinyGPT -> answer tokens

The language model is the same :class:`rlkit.toy_lm.TinyGPT`; the image tokens enter through its
``prefix_embeds`` argument. Consequently **every** LLM RL algorithm works unchanged on the VLM: we
just pass ``prefix_embeds=vlm.encode_images(images)`` to ``generate`` / ``sequence_logprobs``.

ShapeWorld scenes contain 1-4 coloured squares on a black 16x16 canvas. Questions:

* ``count``  - ``"how many red"``  -> ``"2"``   (verifiable, exact-match reward)
* ``color``  - ``"color at a"`` (quadrant a/b/c/d) -> ``"red"``
* ``where``  - ``"where blu"`` -> quadrant letter ``"a".."d"`` (a grounding task: reward can be
  partial, e.g. IoU-style, which is how Visual-RFT / VLM-R1 shape rewards)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .toy_lm import CharTokenizer, GPTConfig, TinyGPT

PALETTE = {"red": (1.0, 0.15, 0.15), "green": (0.15, 0.9, 0.2), "blue": (0.2, 0.35, 1.0),
           "yellow": (1.0, 0.9, 0.1)}
QUADRANTS = "abcd"  # a = top-left, b = top-right, c = bottom-left, d = bottom-right


@dataclass
class Scene:
    image: torch.Tensor                 # (3, 16, 16) float in [0, 1]
    objects: List[Tuple[str, int]]      # (color, quadrant index) for each square
    # Pixel box (row0, col0, row1, col1), exclusive ends, for each object (same order) - used by
    # grounding rewards such as IoU (Visual-RFT style).
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)


class ShapeWorld:
    """Random scenes with at most one square per quadrant (so "where" has a unique answer)."""

    SIZE = 16

    def __init__(self, seed: int = 0, max_objects: int = 4):
        self.rng = random.Random(seed)
        self.max_objects = max_objects

    def scene(self) -> Scene:
        n = self.rng.randint(1, self.max_objects)
        quads = self.rng.sample(range(4), n)
        img = torch.zeros(3, self.SIZE, self.SIZE)
        objects, boxes = [], []
        for q in quads:
            color = self.rng.choice(list(PALETTE))
            r0, c0 = (q // 2) * 8, (q % 2) * 8
            # a 4..6 px square at a random offset inside its 8x8 quadrant
            s = self.rng.randint(4, 6)
            dr, dc = self.rng.randint(0, 8 - s), self.rng.randint(0, 8 - s)
            img[:, r0 + dr:r0 + dr + s, c0 + dc:c0 + dc + s] = torch.tensor(PALETTE[color]).view(3, 1, 1)
            objects.append((color, q))
            boxes.append((r0 + dr, c0 + dc, r0 + dr + s, c0 + dc + s))
        return Scene(img, objects, boxes)

    def question(self, scene: Scene, kind: str = "count") -> Tuple[str, str]:
        """A (question, gold answer) pair about ``scene``. Questions per kind have a fixed length.

        Questions end on the word that matters (the colour / quadrant) instead of a ``?``: the 3-letter
        colour codes end in distinct characters (re*d*, gr*e*, bl*u*, ye*l*), so the last prompt token -
        the one that predicts the answer - can attend straight to the matching image token. With a
        trailing ``?`` a 2-layer model must first copy the colour onto the ``?`` and only then look at
        the image; SFT then plateaus at "find any object" (~50% with 2 objects, 25% with 4).
        """
        if kind == "count":
            color = self.rng.choice(list(PALETTE))
            n = sum(c == color for c, _ in scene.objects)
            return f"how many {color[:3]}", str(n)
        if kind == "color":
            color, q = self.rng.choice(scene.objects)
            return f"color at {QUADRANTS[q]}", color[:3]
        if kind == "where":
            # Only ask about a colour that appears exactly once, otherwise the answer is ambiguous.
            # (A scene with no unique colour falls back to its first such object; VQATask resamples
            # those scenes so its labels are always unambiguous.)
            unique = [(c, q) for c, q in scene.objects if sum(c == o for o, _ in scene.objects) == 1]
            color, q = self.rng.choice(unique or scene.objects)
            return f"where {color[:3]}", QUADRANTS[q]
        raise ValueError(f"unknown question kind {kind!r}")


class TinyVLM(nn.Module):
    """Patch encoder + MLP projector + :class:`TinyGPT` decoder."""

    def __init__(self, tokenizer: CharTokenizer, patch: int = 4, d_model: int = 64, n_layer: int = 2,
                 n_head: int = 4, max_len: int = 32):
        super().__init__()
        self.patch = patch
        n_patches = (ShapeWorld.SIZE // patch) ** 2
        self.vision = nn.Linear(3 * patch * patch, d_model)           # "ViT" with zero layers
        self.vision_pos = nn.Parameter(torch.zeros(1, n_patches, d_model))
        self.projector = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.lm = TinyGPT(GPTConfig(vocab_size=tokenizer.vocab_size, max_len=max_len, d_model=d_model,
                                    n_layer=n_layer, n_head=n_head))
        # Larger than the usual ViT 0.02: the zero-layer patch encoder emits O(1) features, and a
        # 0.02 position code is drowned out - "where" questions then need ~3x more SFT steps.
        nn.init.trunc_normal_(self.vision_pos, std=0.5)

    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """``(B, 3, 16, 16)`` -> ``(B, n_patches, d_model)`` image tokens ready for ``prefix_embeds``."""
        B, C, H, W = images.shape
        p = self.patch
        patches = images.unfold(2, p, p).unfold(3, p, p)              # B, C, H/p, W/p, p, p
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(B, -1, C * p * p)
        return self.projector(self.vision(patches) + self.vision_pos)

    def forward(self, idx: torch.Tensor, images: torch.Tensor):
        return self.lm(idx, prefix_embeds=self.encode_images(images))


class VQATask:
    """Batches of (image, question) prompts with exact-match rewards, mirroring ``AdditionTask``."""

    def __init__(self, tokenizer: CharTokenizer, kind: str = "count", seed: int = 0):
        self.tok = tokenizer
        self.kind = kind
        self.world = ShapeWorld(seed)
        self.answer_len = {"count": 1, "color": 3, "where": 1}[kind]

    def batch(self, n: int, fixed_question: bool = True):
        """Return ``(images (n,3,16,16), prompt ids (n,P), answers, scenes)``.

        Count questions all have the same length; we keep ``kind`` fixed per batch so prompts align.
        """
        scenes = [self._scene() for _ in range(n)]
        qa = [self.world.question(s, self.kind) for s in scenes]
        ids = torch.tensor([self.tok.encode(q) for q, _ in qa])
        images = torch.stack([s.image for s in scenes])
        return images, ids, [a for _, a in qa], scenes

    def _scene(self) -> Scene:
        """A scene that admits an unambiguous question of this task's kind."""
        while True:
            scene = self.world.scene()
            colors = [c for c, _ in scene.objects]
            if self.kind != "where" or any(colors.count(c) == 1 for c in colors):
                return scene

    def reward(self, seq: torch.Tensor, prompt_len: int, answers: List[str]) -> torch.Tensor:
        outs = [self.tok.decode(s[prompt_len:]) for s in seq.tolist()]
        return torch.tensor([float(o == a) for o, a in zip(outs, answers)])

    def sft_batch(self, n: int):
        """``(images, ids)`` with ``prompt + answer + <eos>`` for supervised warm-up."""
        images, ids, answers, _ = self.batch(n)
        rows = [r + self.tok.encode(a, bos=False) + [self.tok.EOS] for r, a in zip(ids.tolist(), answers)]
        L = max(map(len, rows))
        return images, torch.tensor([r + [self.tok.PAD] * (L - len(r)) for r in rows])


def show_scenes(scenes: List[Scene], titles: Optional[List[str]] = None, ncols: int = 6):
    """Grid of ShapeWorld images (with quadrant guides) for the notebooks."""
    import matplotlib.pyplot as plt

    n = len(scenes)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.8 * ncols, 1.9 * nrows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i < n:
            ax.imshow(scenes[i].image.permute(1, 2, 0).numpy(), interpolation="nearest")
            ax.axhline(7.5, color="w", lw=0.5, alpha=0.4)
            ax.axvline(7.5, color="w", lw=0.5, alpha=0.4)
            if titles:
                ax.set_title(titles[i], fontsize=8)
    plt.tight_layout()
    plt.show()
