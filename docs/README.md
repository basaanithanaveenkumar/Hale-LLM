# Docs

- [Architecture](architecture.md): the transformer block, the mixture of experts and the package layout
- [Configuration reference](config.md): every YAML key, defaults and recipes
- [Multi-token prediction](mtp.md): DeepSeek-V3 style MTP training and speculative decoding
- [Training backends](training.md): lean (default) and TorchTitan
- [Jarvislabs cloud training](jarvislabs.md): rent an India GPU and train with one command
- [Data cutting and special tokens](data_and_special_tokens.md): how documents become windows, padding, and what open models do
- [Tokenizer](tokenizer.md): trained SuperBPE/BPE, vocabulary sized to the model, ablation
- [Context length](context_length.md): 4k pretrain, 32k extension, YaRN
- [Usage](usage.md): CLI recap, chat and inference

Also: the [technical report](../technical_report/main.tex) and the [project page](../project-page/index.html)
(both still describe the earlier multi-paradigm version of this repo and are being rewritten).
