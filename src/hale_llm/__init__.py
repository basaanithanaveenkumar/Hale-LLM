"""hale-llm: decoder stacks, token losses, LLM metrics, and training plugins."""

import os
import sys

from loguru import logger

__version__ = "0.1.0"

# loguru starts with a DEBUG sink, so every library that logs while it is imported
# (registry "registered ..." lines, config loading) floods the console before
# setup_logging() runs. Start quiet; set HALE_LLM_LOG_LEVEL=DEBUG to see everything.
logger.remove()
logger.add(sys.stderr, level=os.environ.get("HALE_LLM_LOG_LEVEL", "INFO"))
