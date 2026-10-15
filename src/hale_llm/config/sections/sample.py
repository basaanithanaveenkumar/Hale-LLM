from hale_llm.config.sections.common import StrictModel


class SampleConfig(StrictModel):
    max_new_tokens: int = 32
    temperature: float = 1.0
    sampling_steps: int = 32
    eps: float = 1e-3
    prompt: str = "hello"
    # arch: looped - passes through the shared block at inference (null = model.loop.n_loops).
    # More loops than training is test-time compute scaling.
    num_loops: int | None = None
