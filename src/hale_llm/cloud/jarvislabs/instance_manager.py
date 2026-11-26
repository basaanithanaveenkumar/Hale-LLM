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

    def pause(self, machine_id: int) -> None:
        """Stop compute billing; the disk is kept."""
        self.client.instances.pause(machine_id)
        logger.info("paused {} (storage is still billed)", machine_id)

    def destroy(self, machine_id: int) -> None:
        """Delete the instance and its disk for good."""
        self.client.instances.destroy(machine_id)
        logger.info("destroyed {}", machine_id)
