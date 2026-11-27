"""Keep one training run alive across spot interruptions."""

from __future__ import annotations

import time
from collections.abc import Callable

from loguru import logger

from hale_llm.cloud.jarvislabs.errors import CloudError
from hale_llm.cloud.jarvislabs.instance_manager import InstanceManager
from hale_llm.cloud.jarvislabs.remote_training_job import RemoteTrainingJob
from hale_llm.cloud.jarvislabs.settings import JarvisSettings

# What Jarvislabs shows after it takes a spot GPU back.
_INTERRUPTED_STATUSES = ("Paused", "Pausing")
_MAX_TRANSIENT_FAILURES = 3  # consecutive failed polls on a *running* instance before giving up


class SpotSupervisor:
    """Wait for training to finish; if the instance is paused underneath it, bring it back.

    On an interruption the supervisor resumes the instance, re-uploads the code and restarts
    ``hale-llm-train``. The trainer finds ``resume.pt`` on the disk and continues from the last saved
    step, so an interruption costs at most ``train.lean.save_every`` steps.
    ``machine_id`` always holds the current id, because resuming can change it.
    """

    def __init__(
        self,
        manager: InstanceManager,
        job: RemoteTrainingJob,
        settings: JarvisSettings,
        machine_id: int,
        train_args: list[str],
        poll_seconds: float = 60.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.manager, self.job, self.settings = manager, job, settings
        self.machine_id = machine_id
        self.train_args = train_args
        self.poll_seconds = poll_seconds
        self.sleep = sleep
        self.restarts = 0

    def run(self) -> int:
        """Block until training ends for good and return its exit code."""
        failed_polls = 0
        while True:
            try:
                return self.job.wait_until_finished(self.machine_id, self.poll_seconds)
            except CloudError as error:
                # Step 1: why did polling fail? Ask Jarvislabs for the instance's status.
                status = self.manager.get(self.machine_id).status
                if status not in _INTERRUPTED_STATUSES:
                    # Still running: probably a network blip. Retry a few times before giving up.
                    failed_polls += 1
                    if failed_polls > _MAX_TRANSIENT_FAILURES:
                        raise
                    logger.warning("could not poll {} ({}); retrying", self.machine_id, error)
                    self.sleep(self.poll_seconds)
                    continue
                failed_polls = 0
                # Step 2: the instance was paused underneath the run: recover, or explain why not.
                self._recover(status)

    def _recover(self, status: str) -> None:
        """Resume the instance and restart training, which then resumes from its last checkpoint."""
        if not self.settings.spot:
            raise CloudError(
                f"instance {self.machine_id} is {status} but it is not a spot instance, so it was paused "
                "by someone else; not restarting"
            )
        self.restarts += 1
        if self.restarts > self.settings.max_restarts:
            raise CloudError(
                f"interrupted {self.restarts} times (max_restarts={self.settings.max_restarts}); giving up"
            )
        logger.warning(
            "instance {} was interrupted ({}); recovery {}/{}",
            self.machine_id,
            status,
            self.restarts,
            self.settings.max_restarts,
        )
        self.machine_id = self.manager.resume(self.machine_id)
        # The disk is kept, but re-deploying also covers a replaced instance.
        self.job.deploy(self.machine_id)
        self.job.start(self.machine_id, self.train_args)
