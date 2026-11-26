"""Create, inspect, pause and destroy Jarvislabs instances through the Python SDK."""

from __future__ import annotations

import re

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

    def create(self, label: str):
        """Rent a new instance and wait until it is running."""
        s = self.settings
        logger.info("creating {} x{} in {} ({} GB)", s.gpu_type, s.num_gpus, s.region or "any region", s.storage_gb)
        return self.client.instances.create(
            gpu_type=s.gpu_type,
            num_gpus=s.num_gpus,
            template=s.template,
            storage=s.storage_gb,
            name=self.instance_name(s.name_prefix, label),
            region=s.region,
        )

    def get(self, machine_id: int):
        """Current state of an instance (status, cost, ssh command)."""
        return self.client.instances.get(machine_id)

    def pause(self, machine_id: int) -> None:
        """Stop compute billing; the disk is kept."""
        self.client.instances.pause(machine_id)
        logger.info("paused {} (storage is still billed)", machine_id)

    def destroy(self, machine_id: int) -> None:
        """Delete the instance and its disk for good."""
        self.client.instances.destroy(machine_id)
        logger.info("destroyed {}", machine_id)
