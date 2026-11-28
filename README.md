<p align="center">
  <img src="assets/block_diffusion_step4900.gif" alt="Block diffusion decoding on WikiText — tokens unmask block by block" width="100%">
</p>

<p align="center">
  <em>Block diffusion at step 4900: prompt in steelblue, committed tokens in orange, the active block in black on a light overlay.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/pytorch-2.13+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/tokenizer-trained%20SuperBPE-yellow" alt="Trained SuperBPE tokenizer">
  <img src="https://img.shields.io/badge/package-hale__llm-0A7A3E" alt="hale_llm">
</p>

# HALE-LLM

A small, readable decoder-only language-model codebase in PyTorch: dense and DeepSeek-style mixture-of-experts transformers, multi-token prediction, a trained SuperBPE tokenizer, Muon/Dion optimizers and a lean trainer.

## Train and chat

```bash
uv sync

# 1. Train a small autoregressive model on TinyStories (about 2,000 steps; stop any time, it saves checkpoints).
uv run hale-llm-train --config configs/autoregressive/small.yaml --steps 2000 --no-resume

# 1b. Or the mixture-of-experts variant (1 shared + 4 routed experts per layer, top 2 per token).
uv run hale-llm-train --config configs/autoregressive_moe/nano.yaml --steps 2000 --no-resume

# 2. Talk to it. Replies are continuations of what you type, so start with a sentence, not a question.
uv run hale-llm-chat --config configs/autoregressive/small.yaml --max-new-tokens 40 --temperature 0.8
```

See [docs/](docs/README.md) for the architecture, configuration and tokenizer notes.

The model is a base language model trained on short children's stories, not an assistant: expect
simple story-style sentences that drift. Lower `--temperature` for steadier text, raise it for
variety. Inside the chat, `/help` lists the commands and `/reset` clears the conversation.

## More

- [Docs](docs/README.md): architecture, configuration, tokenizer, MTP
- [Roadmap and notes](docs/roadmap.md): TODO checklist, research notes, baselines
