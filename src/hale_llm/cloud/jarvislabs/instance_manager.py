"""Create, inspect, pause and destroy Jarvislabs instances through the Python SDK."""

from __future__ import annotations

import re
import time

from loguru import logger

from hale_llm.cloud.jarvislabs.errors import CloudError
from hale_llm.cloud.jarvislabs.settings import JarvisSettings


class InstanceManager:
    """Lifecycle of one GPU box. The API token comes from ``jl setup`` or ``JL_API_KEY``."""

    def __init__(self, settings: JarvisSettings, client=None) -> None:
        self.settings = settings
        if client is None:
            try:
                from jarvislabs import Client
            except ImportError as err:
                raise CloudError("jarvislabs is not installed. Run: uv sync --extra cloud") from err
            client = Client()
        self.client = client

    @staticmethod
    def instance_name(prefix: str, label: str) -> str:
        """Jarvislabs names allow letters, digits, spaces, '-' and '_' up to 40 characters."""
        return re.sub(r"[^A-Za-z0-9 _-]", "-", f"{prefix}-{label}")[:40]

    def ensure_ssh_key(self) -> None:
        """Fail early if the account has no SSH key: without one an instance gets no SSH command,
        and uploading code or running commands over SSH can never work."""
        if not self.client.ssh_keys.list():
            raise CloudError(
                "no SSH key on your Jarvislabs account. Create one and register it first:\n"
                '  ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N ""\n'
                "  uv run jl ssh-key add ~/.ssh/id_ed25519.pub --name my-laptop"
            )

    def create(self, label: str):
        """Rent a new instance and wait until it is running."""
        s = self.settings
        logger.info("creating {} {} x{} in {} ({} GB)", "spot" if s.spot else "on-demand", s.gpu_type, s.num_gpus, s.region or "any region", s.storage_gb)
        return self.client.instances.create(
            gpu_type=s.gpu_type,
            num_gpus=s.num_gpus,
            template=s.template,
            storage=s.storage_gb,
            name=self.instance_name(s.name_prefix, label),
            region=s.region,
            is_spot=s.spot,
        )

    def get(self, machine_id: int):
        """Current state of an instance (status, cost, ssh command)."""
        return self.client.instances.get(machine_id)

    def wait_until_ssh_ready(self, machine_id: int, timeout_seconds: float = 300.0, poll_seconds: float = 5.0):
        """Wait until the instance is Running *and* has an SSH command.

        ``create`` returns as soon as the status says Running, but the SSH details show up a few
        seconds later; uploading before that fails with "has no SSH command".
        """
        deadline = time.monotonic() + timeout_seconds
        while True:
            instance = self.get(machine_id)
            if instance.status == "Running" and instance.ssh_command:
                return instance
            if time.monotonic() >= deadline:
                raise CloudError(
                    f"instance {machine_id} not reachable by SSH after {timeout_seconds:.0f}s (status: {instance.status})"
                )
            logger.info("waiting for SSH on {} (status: {})", machine_id, instance.status)
            time.sleep(poll_seconds)

    def resume(self, machine_id: int, poll_seconds: float = 30.0) -> int:
        """Resume a paused instance (as spot when the settings say so) and return its machine id.

        A spot instance that Jarvislabs paused has no GPU until one is free again, so this retries
        until it gets one or ``resume_wait_minutes`` run out. The id can change on resume.
        """
        from jarvislabs.exceptions import APIError, ValidationError  # the errors "no free GPU" and "still pausing" raise

        deadline = time.monotonic() + self.settings.resume_wait_minutes * 60
        while True:
            try:
                instance = self.client.instances.resume(machine_id, is_spot=self.settings.spot)
                break
            except (ValidationError, APIError) as error:
                if time.monotonic() >= deadline:
                    raise CloudError(f"could not resume {machine_id} within {self.settings.resume_wait_minutes} minutes: {error}") from error
                logger.info("cannot resume {} yet ({}); retrying in {:.0f}s", machine_id, error, poll_seconds)
                time.sleep(poll_seconds)
        logger.info("resumed {} as {}", machine_id, instance.machine_id)
        self.wait_until_ssh_ready(instance.machine_id)
        return instance.machine_id

    def pause_if_running(self, machine_id: int) -> None:
        """Pause to stop billing, but never raise: this runs while another error may be propagating."""
        try:
            status = self.get(machine_id).status
            if status == "Running":
                self.pause(machine_id)
            else:
                logger.info("instance {} is already {}; nothing to pause", machine_id, status)
        except Exception as error:  # noqa: BLE001 - must not hide the error that got us here
            logger.error("could not pause {}: {}. Pause or destroy it yourself to stop billing.", machine_id, error)

    def pause(self, machine_id: int) -> None:
        """Stop compute billing; the disk is kept."""
        self.client.instances.pause(machine_id)
        logger.info("paused {} (storage is still billed)", machine_id)

    def destroy(self, machine_id: int) -> None:
        """Delete the instance and its disk for good."""
        self.client.instances.destroy(machine_id)
        logger.info("destroyed {}", machine_id)
