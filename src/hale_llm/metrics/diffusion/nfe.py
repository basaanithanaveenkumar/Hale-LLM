"""
NFE (Number of Function Evaluations) metric for diffusion models.

NFE counts how many times the neural network (the score/noise predictor) is
called during a single sampling run.  This is the primary measure of sampling
efficiency for diffusion models.

Why NFE matters
---------------
Each function evaluation calls the full denoising network forward pass.
Fewer NFE = faster generation at inference time.

  DDPM (Ho et al. 2020)   — typically 1000 NFE
  DDIM (Song et al. 2021) — 10–50 NFE with similar quality
  DPM-Solver (Lu 2022)    — 5–20 NFE
  Consistency Models (Song 2023) — 1–2 NFE

The quality–speed tradeoff (FID vs NFE) is a key comparison axis for
diffusion sampling algorithms.

Sampler step counting
---------------------
For most samplers each denoising step calls the network once, so NFE = steps.
Some samplers (e.g. Heun / DPM-Solver++) call the network twice per step,
so NFE = 2 × steps.  This metric tracks whatever count is provided by the
caller — the caller must increment correctly for their sampler.
"""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("nfe")
class NFEMetric(Metric):
    """Tracks mean and total number of function evaluations per sample.

    Expects ``ctx.batch["nfe"]`` to be a list of ints — one int per generated
    sample in the batch, recording how many network calls were made for that
    sample.

    Results
    -------
    nfe_mean:  Mean NFE across all evaluated samples.
    nfe_min:   Minimum NFE observed (best-case efficiency).
    nfe_max:   Maximum NFE observed (worst-case efficiency).
    nfe_total: Total function evaluations across all samples.
    """

    metric_name = "nfe"

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self._nfe_list: list[int] = []

    def update(self, ctx: MetricContext) -> None:
        """Accumulate NFE counts from one generation batch.

        Expects:
            ctx.batch["nfe"]: list[int] — per-sample NFE counts.
        """
        nfe_values = ctx.batch.get("nfe", [])
        self._nfe_list.extend(int(v) for v in nfe_values)

    def compute(self) -> dict[str, float]:
        """Return NFE statistics.

        Returns:
            nfe_mean:  Average function evaluations per sample.
            nfe_min:   Minimum across all samples.
            nfe_max:   Maximum across all samples.
            nfe_total: Sum of all function evaluations.
        """
        if not self._nfe_list:
            return {"nfe_mean": 0.0, "nfe_min": 0.0, "nfe_max": 0.0, "nfe_total": 0.0}

        total = sum(self._nfe_list)
        n = len(self._nfe_list)
        return {
            "nfe_mean": total / n,
            "nfe_min": float(min(self._nfe_list)),
            "nfe_max": float(max(self._nfe_list)),
            "nfe_total": float(total),
        }
