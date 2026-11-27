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


## TODO
Data strategy
0. data loading dividing the context window among the paragraphs and adding the padding , so the pretraining is not scrap

1. web scale dataset filtering  and mixtures
2. streaming datsaets
3. web dataset
4. Mid trainning strategies
5. Post trianing dataset curation 
8. Check the hugging face blog on training llms with data and model strategy
9. https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1
10. https://huggingface.co/spaces/OpenEvals/evaluation-guidebook#basics-of-tokenization-why-and-how-do-we-tokenize-text
11. https://huggingface.co/papers/2506.05209 -- Open licenced text for pretraining
12. https://huggingface.co/papers/2306.11644


during pretraining use the positional encoding on draft tokens and randomly equally start using it in the pretraining



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
36 YaRN
37 soft softmax
38. by default use focal loss and compare with normall loss
39 look for best activation function change
40 . run the cobtrolled experiments step by step
41. losses that make model to generalize
42. add more number of losses
43. how next lat works and how is it improving speculative decoding
44 router based layers, router based residual connection should be on or off , router based mhc connection
45.QK normalization, logit softcapping, squared ReLU.
46.Training uses the Muon optimizer from Microsoft's Dion library for hidden layers and AdamW for embeddings, on FineWeb10B same as modded-nanogpt.
47. Exponential Moving average weights what does it do does it improve the performane 
48. https://huggingface.co/papers/2502.16982 - Moun
49. Use the sliding window like inkling style
50 use the looped transformers only for reasoning in the mid layers
51. check different varientients and adaptive varients of looped transformers
52. looped transformers and MTP should not exceep 8 , safer  is 4



Metrics:
1. come up with prper metrics 

Infracture
https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=high-level_overview

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

# 1b. Or the mixture-of-experts variant (1 shared + 4 routed experts per layer, top 2 per token).
uv run hale-llm-train --config configs/autoregressive_moe/nano.yaml --steps 2000 --no-resume

# 2. Talk to it. Replies are continuations of what you type, so start with a sentence, not a question.
uv run hale-llm-chat --config configs/autoregressive/small.yaml --max-new-tokens 40 --temperature 0.8
```

See [docs/](docs/README.md) for the architecture, configuration and tokenizer notes.

The model is a base language model trained on short children's stories, not an assistant: expect
simple story-style sentences that drift. Lower `--temperature` for steadier text, raise it for
variety. Inside the chat, `/help` lists the commands and `/reset` clears the conversation.


1. MTP with transformer block per token
2. MTP with transformer block per token which has looped
3. FP8 training (look at deepseek paper)
4. Fp4 training
5.  Next Latent world model
6. Soft-MTP ("Multi-Token Prediction Done Right", MIT, Oct 2026). This is the most directly relevant to you. It keeps the DeepSeek sequential design and changes only the gradient. The paper finds that the MTP loss gradient has two parts:

a useful KL self-distillation term, which pulls each MTP head toward the model's own next-token distribution;
a residual that carries about 80% of the MTP gradient reaching the shared trunk but has a signal-to-noise ratio roughly 4 to 4.6 times lower than the KL term. It also pushes against next-token learning, and the problem grows with scale.
Caveat: it was tested only at 167M, 332M, 689M, and 1.2B parameters, all LLaMA-style, trained on FineWeb-Edu. The gains are small, and the paper is very new.
7. Token order prediction
8. MTP-D (Tencent, AACL 2026). It adds self-distillation and a looped extension to the MTP heads, aimed at higher acceptance rates with more heads.
AdaMTP (Aug 2026). On Llama-3.1-8B, Qwen-2.5-7B, and Gemma-3-12B, it consistently outperforms standard MTP in both task performance and inference speedup.
9. Soft-MTP table shows MTP losing to NTP at ≤332M too.
10. Add the positional information to the deep seek MTP style does having the position of number of tokens predicted addes anything

# Topics completed
1. Deepseek v3Style MTP with DraftMTP =1
2. MTP done right (its does not show very huge improvement  as per the paper results , it stiopping the residual signaland aony passing the KL divergance signal will recheck later on )
3. Next latent
4. Next latent + LeWorld model

# baseline -- use the hyper parametes from hugging fae playbook, qwen model and inkling model
1. Autoregressive MOE
2. Autoregressive MOE_MTP
3. FP8 training
4. FP4training
6. Looped transformers
7. 


Referenes:
1. https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#introduction

MTP
1.Deepseek v3 style -λ is the knob that sets how much the extra MTP signal counts compared with the main loss. A schedule makes λ a function of training progress instead of a constant.

DeepSeek-V3 used a step schedule: λ is 0.3 for the first 10T training tokens and 0.1 for the remaining 4.8T. That means a switch about two-thirds of the way through (10 / 14.8 ≈ 0.68, my arithmetic)
2.Add the position to it so it can be extended like context window (new)
3. Soft MTP
4. MTP done right (minor tweak)
5. Tensenet MTP paper

Remarks for lateron:
1. Next Lat and MTP                         
2. Deepseek v3 fim tokens    
3. World Model into LLM 



1. add the dion optimizer
2. add the transformer engine library for (lean to use the https://github.com/alint77/nanogpt-fp8/blob/master/train_2.py)




1. Looped transformers
2. Fp4 training
3. add the MHC and predecisor varient and check
4. activation function 
5. add the MTP  meta varient 
6. MTP instead of predicting a constant token randomly predict the next t+n token where nis randome between the draft window of 1-4, with positional information also added , its not next token prediction its next random prediction in s sentence window (noval)
7. in similar way try the droup out of token prediction in window and check the results , this acts as a world model
8 .for the above compare it with the just next token prediction pipleine and NTP+ t+1 prediction prediction pipeline

MoE diff needs to Be fixed 
Where it differs from the DeepSeek-V3 paper, which I recorded in docs/config.md:
- Router weights are a softmax over the chosen top-k scores (Mixtral style). V3 uses sigmoid scores.
- Balancing uses an auxiliary loss. V3 uses an adjusted bias with no loss.
- Every layer is MoE. V3 keeps its first three layers dense.
- Padding tokens are routed and counted in the balance loss.


Blog :
1. explain the fP8, fp4 training 
 -- story how models are post quantized to int 4 bit
 -- fixed point vs floating point number
 -- block scaling
 -- inner scaling 
 -- outer scalling
 -- Stocastic rounding , what is it and why is it needed (forward pass and backward pass rounding issue, )
 -- General element matrix multiplication GEMM
2. explain looped transformers

