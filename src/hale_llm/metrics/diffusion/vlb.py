"""
VLB (Variational Lower Bound) metric for diffusion models.

Ho et al. (2020) "Denoising Diffusion Probabilistic Models"
— https://arxiv.org/abs/2006.11239

What is VLB?
------------
The Evidence Lower BOund (ELBO / VLB) is the training objective of a DDPM.
It decomposes as:

    L_VLB = L_0 + L_1 + ... + L_{T-1} + L_T

where:
  L_T = KL(q(x_T|x_0) || p(x_T))          — prior matching (constant)
  L_t = KL(q(x_{t-1}|x_t,x_0) || p_θ(x_{t-1}|x_t))   for t in 1..T-1
  L_0 = -log p_θ(x_0|x_1)                  — reconstruction term

Each L_t is a KL divergence between two Gaussians, which has a closed form:

    KL(N(μ₁,σ₁²) || N(μ₂,σ₂²)) =
        log(σ₂/σ₁) + (σ₁² + (μ₁-μ₂)²)/(2σ₂²) - 1/2

Monitoring VLB
--------------
VLB is primarily a training diagnostic:
- Decreasing L_t at intermediate timesteps → model is learning the reverse process.
- L_0 converges to the negative log-likelihood of the final reconstruction.
- Very large L_T indicates the forward process is not approaching the prior.

Usage
-----
This metric expects the batch to provide per-timestep KL divergences
(already computed in the training loop):

    ctx.batch["vlb_per_step"]: list[float] — per-step KL, one per sample
    ctx.batch["timestep"]:     list[int]   — which diffusion timestep
"""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("vlb")
class VLBMetric(Metric):
    """Tracks mean VLB (ELBO) and per-timestep KL statistics.

    Results
    -------
    vlb_mean:         Mean VLB over all timesteps and samples (lower = better).
    vlb_l0:           Mean reconstruction term L_0 (timestep 0 only).
    vlb_lt_mean:      Mean KL for intermediate timesteps (t > 0).
    vlb_lt_max:       Maximum per-step KL (reveals worst-case timestep).
    """

    metric_name = "vlb"

    def reset(self) -> None:
        self._all_kl:    list[float] = []   # all steps, all samples
        self._l0_kl:     list[float] = []   # timestep 0 only
        self._lt_kl:     list[float] = []   # timestep > 0

    def update(self, ctx: MetricContext) -> None:
        """Accumulate VLB components from one training batch.

        Expects:
            ctx.batch["vlb_per_step"]: list[float] — one KL value per sample.
            ctx.batch["timestep"]:     list[int]   — one timestep per sample.
        """
        kl_values = ctx.batch.get("vlb_per_step", [])
        timesteps  = ctx.batch.get("timestep", [])

        for kl, t in zip(kl_values, timesteps):
            kl_f = float(kl)
            self._all_kl.append(kl_f)
            if int(t) == 0:
                self._l0_kl.append(kl_f)
            else:
                self._lt_kl.append(kl_f)

    def compute(self) -> dict[str, float]:
        """Return VLB statistics.

        Returns:
            vlb_mean:    Mean KL across all steps (overall training objective).
            vlb_l0:      Mean reconstruction term (L_0, timestep 0).
            vlb_lt_mean: Mean KL for diffusion timesteps t > 0.
            vlb_lt_max:  Maximum observed per-step KL (worst timestep region).
        """
        if not self._all_kl:
            return {
                "vlb_mean": 0.0,
                "vlb_l0": 0.0,
                "vlb_lt_mean": 0.0,
                "vlb_lt_max": 0.0,
            }

        def _mean(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "vlb_mean":    _mean(self._all_kl),
            "vlb_l0":      _mean(self._l0_kl),
            "vlb_lt_mean": _mean(self._lt_kl),
            "vlb_lt_max":  max(self._lt_kl) if self._lt_kl else 0.0,
        }
