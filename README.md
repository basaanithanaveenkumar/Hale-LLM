<p align="center">
  <img src="assets/block_diffusion_step4900.gif" alt="Block diffusion decoding on WikiText — tokens unmask block by block" width="100%">
</p>

<p align="center">
  <em>Block diffusion at step 4900: prompt in steelblue, committed tokens in orange, the active block in black on a light overlay.</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-3776AB?logo=python&logoColor=white" alt="Python 3.12+">
  <img src="https://img.shields.io/badge/pytorch-2.13+-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/tokenizer-GPT--2-yellow" alt="GPT-2 tokenizer">
  <img src="https://img.shields.io/badge/package-hale__llm-0A7A3E" alt="hale_llm">
</p>


## TODO
Data strategy
1. web scale dataset filtering  and mixtures
2. streaming datsaets
3. web dataset
4. Mid trainning strategies
5. Post trianing dataset curation 
8. Check the hugging face blog on training llms with data and model strategy
9. 
10. 





Model strategy
1. Use the Next latent inspired world model
2. create a world model based MTP prediction
3. Use the flash attention latest verion so we can use less memory and faster training
4. Use the RoPE as Positional encoding
5.GQA attentiontry
6. Multi head latent attention
7.DeepSeek MOE
8. Expert routing Bias
9.RMS Norm
10.SWIGLU, GLU
11. Gated delta Net layer
12. 4Bit training FP4
13. 8 bit trainining FP8
14. Jarvislab integration
15 ray integration
16 GeLU
17 what is in Gemma file local layers per global layer 
18 Latent MoE from Nvidia
19 Attention residual over 8 blocks inkling
20 Meta uses only RoPE in local layers only
21. Instella MOE -AMD  adds a output gate to MLA
22. Deepseek sparce attention
23. 128 token window with learned sinks
24. Block attention residuals
25. flash attention and its vrients
25. Training frameworks for LLM to manage the distributed training
26. Moun optimizer have it in the registry and enabled by default
27. focal loss added for token prediction what it contributes to the table
28. Multi query attention vs group query attention
29. torch titan
29. KL divergance for world models
30. please add the gradient accumulation
31. add the activation checkpointing
32. add the gradient clipping
33. Looped transformers
34. class or cluster based expert selection so , based on propmt it loads a series of experts should be easier for inference
35. need to find a better light weight tokenizer , ann register it and also study more about tokenizers, should I pick a preexisting tokenizer or should i train it from scratch



Metrics:
1. come up with prper metrics 

Infracture
0. Docker support, Off-instance copies. Sync checkpoints and W&B artifacts to HF Hub (private),
1. Jarvislabs Intrgeation (Use a GPU VM, not a template, easy for tweaking)
2. lets use a modded trainier script from nano chat
3. lets use the torch titan becuse it claims with less modifications
4. DDP with gradient accumulation, then FSDP2 above roughly 3B
5. need to check this Compile	torch.compile on the blocks	Fuses the quantize/scale ops around GEMMs. At small model widths, quantization overhead decides whether low precision is faster at all.
6. Need to check this FP8	Transformer Engine (te.Linear, current or delayed per-tensor scaling), with torchao float8 tensorwise as an A/B
7. FP4	Your own NVFP4Linear (custom autograd function): CUTLASS SM120 block-scaled GEMM or a community kernel, plus your own quantizers
8. Attention	FlashAttention / SDPA in BF16	Keep attention out of the low-precision study at first
9. Tracking	W&B, logging per-layer amax, grad norms and update/weight ratios	You'll diagnose FP4 divergence from these
10.MXFP8 A/B through torchao.
11. NVFP4 forward only.
12. 

Baseline:
1. fp32 training, adam optimizer and model components working first without any issues
2. Bf16 or fp16 training
3. fp8 training
4. fp4 training


## Training recipie for context length 
SmolLM3, at your scale, followed it: it pretrained with a global batch size of 2.36M tokens with 4096 sequence length, then trained on an additional 100B tokens to extend its context length. 
huggingface
githubusercontent

Pretrain at 4k (8k if your data has a lot of code) for ~90–95% of tokens. Use RoPE with base θ around 10k–500k.
Extend during mid-training: move to 32k and raise RoPE θ (e.g., to 1M–5M). Train on naturally long documents (books, papers, full repository files) mixed with short data so short-context quality doesn't regress. Budget about 2–5% of total tokens for this.
Optionally go to 64k–128k: a second stage, or YaRN-style RoPE scaling at inference on top of the 32k-trained model.
Train post-training data at the target length, including long multi-turn and agent traces. Otherwise the model will technically accept 32k tokens but won't follow instructions well at that length



## Train and chat

```bash
uv sync

# 1. Train a small autoregressive model on TinyStories (about 2,000 steps; stop any time, it saves checkpoints).
uv run hale-llm-train --config configs/autoregressive/small.yaml --steps 2000 --no-resume

# 2. Talk to it. Replies are continuations of what you type, so start with a sentence, not a question.
uv run hale-llm-chat --config configs/autoregressive/small.yaml --max-new-tokens 40 --temperature 0.8
```

The model is a base language model trained on short children's stories, not an assistant: expect
simple story-style sentences that drift. Lower `--temperature` for steadier text, raise it for
variety. Inside the chat, `/help` lists the commands and `/reset` clears the conversation.
