"""The stable input injection of Parcae: ``u = A_bar * h + B_bar e`` with a spectral radius below 1 by construction."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class StableInjection(nn.Module):
    """Mixes the recurrent state ``h`` and the prelude output ``e`` into the input of the recurrent blocks.

    Parcae (Prairie et al., "Parcae: Scaling Laws For Stable Looped Language Models", arXiv 2604.12946, Sec. 3-4.1) views
    one loop as a dynamical system over the residual stream::

        h_{t+1} = A_bar h_t + B_bar e + R_bar(h_t, e)          (R_bar = what the transformer blocks add on top)

    and shows that the linearised system ``h_{t+1} = A_bar h_t + B_bar e`` is stable only if the spectral radius
    ``rho(A_bar) < 1``. Adding ``h`` and ``e`` (A_bar = I, rho = 1) is marginally stable, concatenating them with a
    free matrix (A_bar unconstrained) is unstable. Parcae therefore parameterises ``A`` and ``B`` in continuous time and
    discretises them with a learned step ``Delta``::

        A     = Diag(-exp(log_A))            negative diagonal, so every eigenvalue is negative
        A_bar = exp(Delta * A)               zero-order hold; elementwise, so rho(A_bar) = max_i exp(-Delta_i exp(log_A_i))
        B_bar = Delta * B                    Euler; B is a free [d, d] matrix

    ``A_bar`` is diagonal with entries in (0, 1) as long as ``Delta > 0``. The paper does not say how ``Delta`` is kept
    positive; here ``Delta = softplus(delta_raw)`` (the Mamba choice), which makes ``rho(A_bar) < 1`` hold for every value
    the optimizer can reach, not only at initialisation.

    Initialisation is not given in the paper either. Here ``log_A = 0`` (``A = -1``), ``Delta`` is chosen so that
    ``A_bar`` starts at ``decay_init`` (default 0.5), and ``B = I``.
    """

    def __init__(self, d_model: int, decay_init: float = 0.5) -> None:
        super().__init__()
        if not 0.0 < decay_init < 1.0:
            raise ValueError(f"decay_init={decay_init} must be strictly between 0 and 1 (it is the initial A_bar)")
        # 1. A = -exp(log_A): the learnable vector log_A, started at 0 so A = -1.
        self.log_a = nn.Parameter(torch.zeros(d_model))
        # 2. Delta = softplus(delta_raw) > 0, started so that exp(Delta * A) = decay_init, i.e. Delta = -ln(decay_init).
        #    softplus^-1(y) = ln(exp(y) - 1).
        delta_start = -math.log(decay_init)
        self.delta_raw = nn.Parameter(torch.full((d_model,), math.log(math.expm1(delta_start))))
        # 3. B: unconstrained, started at the identity so the prelude output is passed on unchanged (then scaled by Delta).
        self.input_matrix = nn.Parameter(torch.eye(d_model))

    def delta(self) -> torch.Tensor:
        """The learned discretisation step ``Delta``, ``[d]``, strictly positive."""
        return F.softplus(self.delta_raw)

    def state_decay(self) -> torch.Tensor:
        """``A_bar = exp(Delta * A)`` as a vector ``[d]``; every entry is in (0, 1]."""
        return torch.exp(self.delta() * -torch.exp(self.log_a))

    def input_gain(self) -> torch.Tensor:
        """``B_bar = Delta * B`` ``[d, d]``: row ``i`` of ``B`` scaled by ``Delta_i``."""
        return self.delta()[:, None] * self.input_matrix

    def spectral_radius(self) -> torch.Tensor:
        """``rho(A_bar)``: the largest eigenvalue of the diagonal ``A_bar``. Below 1 by construction."""
        return self.state_decay().max()

    def forward(self, state: torch.Tensor, prelude_output: torch.Tensor) -> torch.Tensor:
        """``A_bar * state + prelude_output @ B_bar^T``; both inputs and the result are ``[B, S, d]``."""
        # 1. The old state, decayed channel by channel.
        carried = self.state_decay() * state  # [B, S, d]
        # 2. The prelude output, mixed by B_bar (out_i = sum_j B_bar[i, j] e_j).
        injected = prelude_output @ self.input_gain().t()  # [B, S, d]
        return carried + injected
