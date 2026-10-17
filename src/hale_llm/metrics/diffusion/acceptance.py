"""
Acceptance Rate metric for diffusion samplers using MCMC correction steps.

Relevant to samplers that use Markov Chain Monte Carlo (MCMC) correction:

  Denoising Diffusion Langevin MCMC (DULA)
  DDPM with Metropolis–Hastings correction (Bao et al. 2022)
  Consistency Models with MCMC refinement
  Langevin-corrected diffusion (Song et al. 2020)

What is Acceptance Rate?
------------------------
In Metropolis–Hastings (MH) MCMC, after proposing a transition x' from x, the
move is accepted with probability:

    α(x → x') = min( 1,  p(x') q(x|x') / (p(x) q(x'|x)) )

where p is the target distribution and q is the proposal distribution.

    acceptance_rate = accepted_steps / total_steps  ∈ [0, 1]

Interpretation
--------------
  acceptance_rate ≈ 1.0 → proposal and target are closely aligned, steps are
                           accepted almost always; very efficient mixing.
  acceptance_rate ≈ 0.0 → proposals are frequently rejected; sampler is stuck.

Optimal acceptance rate for MH in high dimensions is theoretically ~0.234
(Roberts et al. 1997) for random-walk proposals and ~0.574 for MALA proposals.
Values far from these suggest the step size needs tuning.

Usage
-----
    ctx.batch["accepted_steps"]: list[int] — accepted MH steps per sample.
    ctx.batch["total_steps"]:    list[int] — total proposed MH steps per sample.
"""

from __future__ import annotations

from hale_llm.core.registry import register_metric
from hale_llm.metrics.base import Metric, MetricContext


@register_metric("acceptance_rate")
class AcceptanceRateMetric(Metric):
    """Tracks Metropolis–Hastings acceptance rate for diffusion MCMC samplers.

    Results
    -------
    acceptance_rate:         Overall accepted / proposed across all samples.
    acceptance_rate_mean:    Mean per-sample acceptance rate.
    acceptance_rate_min:     Minimum per-sample acceptance rate.
    acceptance_rate_max:     Maximum per-sample acceptance rate.
    """

    metric_name = "acceptance_rate"

    def reset(self) -> None:
        """Clear the running sums (called before every evaluation pass)."""
        self._per_sample_rates: list[float] = []
        self._total_accepted: int = 0
        self._total_proposed: int = 0

    def update(self, ctx: MetricContext) -> None:
        """Accumulate acceptance counts from one batch.

        Expects:
            ctx.batch["accepted_steps"]: list[int] per-sample accepted steps.
            ctx.batch["total_steps"]:    list[int] per-sample total steps.
        """
        accepted_list = ctx.batch.get("accepted_steps", [])
        total_list = ctx.batch.get("total_steps", [])

        for acc, tot in zip(accepted_list, total_list):
            acc_i = int(acc)
            tot_i = max(int(tot), 1)
            self._total_accepted += acc_i
            self._total_proposed += tot_i
            self._per_sample_rates.append(acc_i / tot_i)

    def compute(self) -> dict[str, float]:
        """Return acceptance rate statistics.

        Returns:
            acceptance_rate:      Aggregate rate = total accepted / total proposed.
            acceptance_rate_mean: Mean per-sample rate.
            acceptance_rate_min:  Lowest per-sample rate (worst-mixing sample).
            acceptance_rate_max:  Highest per-sample rate (best-mixing sample).
        """
        if not self._per_sample_rates:
            return {
                "acceptance_rate": 0.0,
                "acceptance_rate_mean": 0.0,
                "acceptance_rate_min": 0.0,
                "acceptance_rate_max": 0.0,
            }

        rates = self._per_sample_rates
        return {
            "acceptance_rate": self._total_accepted / max(self._total_proposed, 1),
            "acceptance_rate_mean": sum(rates) / len(rates),
            "acceptance_rate_min": min(rates),
            "acceptance_rate_max": max(rates),
        }
