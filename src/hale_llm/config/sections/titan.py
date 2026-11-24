from hale_llm.config.sections.common import StrictModel


class TitanConfig(StrictModel):
    """How to launch TorchTitan (``train.trainer: torchtitan``). NVIDIA GPUs only.

    TorchTitan describes a run as a Python recipe: ``module`` is the import path of
    the file that defines it and ``config`` the function inside it. Model, data and
    parallelism come from that recipe, not from the rest of this YAML.
    """

    module: str = "torchtitan_recipes.tests.models.llama3"
    config: str = "llama3_debugmodel"
    num_gpus: int | None = None  # None = every visible GPU
    repo_dir: str = "ext_pkgs/torchtitan"  # working directory; must contain torchtitan_recipes/
    output_dir: str | None = None  # forwarded as --output-dir
    resume_step: int | None = None  # forwarded as --resume-step
    overrides: list[str] = []  # each entry becomes `--override TARGET[=JSON]`
    extra_args: list[str] = []  # appended verbatim to the torchtitan command line
