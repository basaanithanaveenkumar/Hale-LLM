"""Wall-clock time of a training run, carried across interruptions."""

from __future__ import annotations

import time


class RunTimer:
    """Time spent training so far: seconds from earlier sessions plus seconds in this one.

    A run that was interrupted and resumed (a spot instance being paused) starts a new process, so the clock
    of the first one is gone. The earlier total is saved in the resume checkpoint and passed here as
    ``previous_seconds``; the total of the whole run is then still right. Time spent waiting between sessions
    is not counted, only time spent running.
    """

    def __init__(self, previous_seconds: float = 0.0) -> None:
        if previous_seconds < 0:
            raise ValueError(f"previous_seconds cannot be negative: {previous_seconds}")
        self.previous_seconds = previous_seconds
        # perf_counter is monotonic: it does not jump when the system clock is changed.
        self._started = time.perf_counter()

    def session_seconds(self) -> float:
        """Seconds since this process began training."""
        return time.perf_counter() - self._started

    def total_seconds(self) -> float:
        """Seconds of training in all sessions so far."""
        return self.previous_seconds + self.session_seconds()
