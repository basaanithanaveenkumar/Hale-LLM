from hale_llm.config.sections.common import StrictModel


class SampleConfig(StrictModel):
    max_new_tokens: int = 32
    temperature: float = 1.0
    prompt: str = "hello"
