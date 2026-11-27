from hale_llm.config.sections.common import StrictModel


class SampleConfig(StrictModel):
    max_new_tokens: int = 32
    temperature: float = 1.0
    prompt: str = "hello"
    # autoregressive_mtp only: draft tokens with the MTP heads and verify them with the main model
    # (speculative decoding). Output is identical in distribution (exactly identical at temperature 0).
    speculative: bool = True
