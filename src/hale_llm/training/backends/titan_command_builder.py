"""Turn ``train.titan`` settings into the ``torchrun`` command line TorchTitan expects."""

from __future__ import annotations

import sys

from hale_llm.config.sections.titan import TitanConfig


class TorchTitanCommandBuilder:
    """Builds the same command as TorchTitan's own ``run_train.sh``.

    ``run_train.sh`` boils down to::

        torchrun --nproc_per_node=N --rdzv_backend c10d --rdzv_endpoint=localhost:0 \\
                 --local-ranks-filter 0 --role rank --tee 3 \\
                 -m torchtitan.train --module MODULE --config CONFIG [options]

    Only the pure command construction lives here, so it can be unit-tested without a GPU.
    """

    def __init__(self, settings: TitanConfig, num_gpus: int) -> None:
        if num_gpus < 1:
            raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")
        self._settings = settings
        self._num_gpus = num_gpus

    def build(self) -> list[str]:
        s = self._settings
        command = [
            sys.executable, "-m", "torch.distributed.run",  # same as `torchrun`, no PATH lookup
            f"--nproc_per_node={self._num_gpus}",
            "--rdzv_backend", "c10d",
            "--rdzv_endpoint", "localhost:0",
            "--local-ranks-filter", "0",  # only rank 0 prints
            "--role", "rank",
            "--tee", "3",
            "-m", "torchtitan.train",
            "--module", s.module,
            "--config", s.config,
        ]
        if s.output_dir:
            command += ["--output-dir", s.output_dir]
        if s.resume_step is not None:
            command += ["--resume-step", str(s.resume_step)]
        for override in s.overrides:
            command += ["--override", override]
        return command + list(s.extra_args)
