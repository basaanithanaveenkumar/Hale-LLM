"""Download WikiText-2 once so training does not hit the hub on every import."""

from datasets import load_dataset
from loguru import logger


def main() -> None:
    """Cache ``Salesforce/wikitext`` (``wikitext-2-raw-v1``) in the Hugging Face datasets cache."""
    load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1")
    logger.info("wikitext-2 cached")


if __name__ == "__main__":
    main()
