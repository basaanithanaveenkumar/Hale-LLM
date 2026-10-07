# Architecture

Mermaid diagrams (render on GitHub). See also the [project page](../project-page/index.html)
and the [README](../README.md) for a laymen overview.
For usage details, see [usage.md](usage.md) and [core.md](core.md).

---

## 1. Shared backbone — LGT transformer

All four generation paradigms run on the same Local-Global Transformer (LGT) or standard GPT backbone. Autoregressive uses GPT; all diffusion/flow variants use LGT because they need bidirectional context.

```mermaid
flowchart TB
  subgraph LGT["LGT layer (Local-Global Transformer)"]
    H_IN["hidden states h  [B, S, D]"]

    subgraph LOCAL["Local attention heads  (nl heads)"]
      LNorm["RMSNorm"]
      LATTN["MHA within a window of size w\n(e.g. w = 32 tokens)\nRoPE with local frequency θ_local"]
      LOUT["local context  [B, S, D_l]"]
      LNorm --> LATTN --> LOUT
    end

    subgraph GLOBAL["Global attention heads  (ng heads)"]
      GNorm["RMSNorm"]
      GATTN["MHA over full sequence\nRoPE with global frequency θ_global\n(nl + ng = n_heads total)"]
      GOUT["global context  [B, S, D_g]"]
      GNorm --> GATTN --> GOUT
    end

    MERGE["concat(local, global) → Linear → [B, S, D]"]
    ADD1["residual add"]

    subgraph FFN_L["FFN sub-layer"]
      FNorm["RMSNorm"]
      FFNMOD["GeGLU: gate(x) × up(x) → down\nor MoE (configurable)"]
      FOUT["[B, S, D]"]
      FNorm --> FFNMOD --> FOUT
    end

    ADD2["residual add"]
    H_OUT["h_out  [B, S, D]"]

    H_IN --> LOCAL & GLOBAL
    LOUT & GOUT --> MERGE --> ADD1
    ADD1 --> FNorm
    FOUT --> ADD2 --> H_OUT
  end
```

---

## 2. Four generation paradigms in detail

### 2a. Autoregressive (reference baseline)

```mermaid
flowchart LR
  subgraph AR_TRAIN["Training"]
    TOKENS["token sequence x₁ x₂ … xₙ"]
    CAUSAL["causal mask\n(each position attends\nonly to previous tokens)"]
    LOGITS["logits p(xₜ | x₁…xₜ₋₁)\nfor all positions in parallel"]
    CE_LOSS["cross-entropy loss\naverage over non-padding tokens"]
    TOKENS --> CAUSAL --> LOGITS --> CE_LOSS
  end

  subgraph AR_SAMPLE["Sampling"]
    START["prefix tokens"]
    STEP["sample xₙ₊₁ ~ softmax(logits[-1] / τ)\n(temperature τ, top-k, top-p)"]
    APPEND["append xₙ₊₁ to sequence"]
    START --> STEP --> APPEND
    APPEND -->|"repeat until EOS or max_len"| STEP
  end
```

### 2b. Masked diffusion (absorbing-state)

```mermaid
flowchart LR
  subgraph MD_TRAIN["Training"]
    X1["clean tokens x₁…xₙ"]
    T_SAMP["sample noise level t ~ U(0,1)"]
    MASK_PROB["mask probability m(t)\n(e.g. cosine schedule)"]
    XT["x_t: replace each token\nwith [MASK] with prob m(t)"]
    BIDIR["bidirectional attention\n(all positions see all others)"]
    UNMASK_LOGITS["p(x_i | x_t)  for every masked i"]
    MASK_LOSS["cross-entropy on masked positions only\nwith masking weight 1/m(t)"]
    X1 & T_SAMP --> MASK_PROB --> XT --> BIDIR --> UNMASK_LOGITS --> MASK_LOSS
  end

  subgraph MD_SAMPLE["Sampling (MDLM schedule)"]
    ALLMASKED["x = [MASK]…[MASK]\n(all positions masked)"]
    SCORE["score all masked positions\nvia one forward pass"]
    SELECT["unmask fraction 1/T tokens\n(highest-confidence predictions)"]
    ALLMASKED --> SCORE --> SELECT
    SELECT -->|"repeat T steps"| SCORE
  end
```

### 2c. Block diffusion

```mermaid
flowchart LR
  subgraph BD_TRAIN["Training"]
    SEQBD["x₁…xₙ split into blocks\nof size block_size B"]
    BLOCK_K["select block k at random"]
    PREV["x₁…x_{kB-1} committed\n(prefix, visible via block-causal mask)"]
    MASKED_BLK["x_{kB}…x_{(k+1)B-1} fully masked"]
    BLKCAUSAL["block-causal attention:\nprefix tokens see each other\nmasked block sees prefix + itself"]
    PRED["p(x_i | prefix, x_t_block)\nfor i in masked block"]
    BLKLOSS["cross-entropy on masked block tokens"]
    SEQBD --> BLOCK_K
    PREV & MASKED_BLK --> BLKCAUSAL --> PRED --> BLKLOSS
  end

  subgraph BD_SAMPLE["Sampling (left to right over blocks)"]
    COMMIT["committed = []"]
    NEXT_BLK["denoise next block via T diffusion steps\n(bidirectional within block,\nblock-causal to committed prefix)"]
    APPEND_BLK["append denoised block to committed"]
    COMMIT --> NEXT_BLK --> APPEND_BLK
    APPEND_BLK -->|"repeat over blocks"| NEXT_BLK
  end
```

### 2d. Discrete flow matching

```mermaid
flowchart LR
  subgraph FM_TRAIN["Training"]
    X0_FM["x₀ ~ uniform over vocab\n(noisy distribution)"]
    X1_FM["x₁ = ground truth tokens\n(target distribution)"]
    T_FM["t ~ U(0, 1)"]
    XT_FM["x_t: straight-line interpolation\np_t(y) = (1-t)·p₀(y) + t·p₁(y)\n(per-token token mixture)"]
    TARGET_FM["target velocity\nv*(y) = p₁(y) - p₀(y)"]
    PRED_FM["model predicts v_θ(x_t, t)\nvia full-sequence forward pass"]
    FLOW_LOSS["MSE or KL(v_θ, v*)"]
    X0_FM & X1_FM & T_FM --> XT_FM --> TARGET_FM & PRED_FM --> FLOW_LOSS
  end

  subgraph FM_SAMPLE["Sampling (Euler integration)"]
    P0["p₀ = uniform distribution"]
    EULER["for i in 0…T:\n  dt = 1/T\n  v = model(p_t, t_i)\n  p_{t+dt} = p_t + v · dt\n  (re-normalise to simplex)"]
    P1_OUT["p₁ ≈ data distribution\n→ sample token from p₁"]
    P0 --> EULER --> P1_OUT
  end
```

---

## 3. Token backbone — ids to logits

```mermaid
flowchart LR
  IDS["token ids  [B, S]"]
  EMB["Token embedding\n50,257 × D_model\n(GPT-2 tokenizer)"]
  STACK["LGT or GPT transformer stack\nN layers, D_model=384, 8 heads, D_ff=1536"]
  LM_HEAD["LM head\nLinear D_model → 50,257\n(often tied with embedding weights)"]
  LOGITS["logits  [B, S, 50,257]"]

  IDS --> EMB --> STACK --> LM_HEAD --> LOGITS
```

---

## 4. Loss functions

```mermaid
flowchart TB
  subgraph AR_LOSS["Autoregressive loss"]
    AR_L["CE( logits[:, :-1], tokens[:, 1:] )\nshift by 1 — predict next token\naverage over non-padding positions"]
  end
  subgraph MD_LOSS_F["Masked diffusion loss"]
    MD_L["CE( p(x_i | x_t) , x_i ) for masked i only\n× reweighting 1 / m(t)\n(upweights highly masked steps)"]
  end
  subgraph BD_LOSS_F["Block diffusion loss"]
    BD_L["CE( p(x_i | prefix, x_t_block) , x_i )\nfor tokens i in the masked block only"]
  end
  subgraph FM_LOSS_F["Flow matching loss"]
    FM_L["MSE( v_θ(x_t, t) , p₁ − p₀ )\nor token-level KL(v_θ || v*)\nover all positions"]
  end
```

---

## 5. Visualiser — block diffusion GIF

```mermaid
flowchart LR
  CKPT["checkpoint (latest or named run)"]
  PROMPT["prompt text (prefix)"]
  MODEL_VIZ["load model + config"]
  SAMPLE_VIZ["run block diffusion sampling\nrecord partial sequences at each\ndenoising step of each block"]
  FRAMES["list of partial sequences"]
  RENDER["render each frame:\nprompt = steelblue\ncommitted tokens = orange\nactive block = black on light overlay\nmasked positions = dimmed"]
  GIF["output.gif\n(one frame per denoising step)"]

  CKPT & PROMPT --> MODEL_VIZ --> SAMPLE_VIZ --> FRAMES --> RENDER --> GIF
```

---

## 6. Config inheritance and variant switching

```mermaid
flowchart LR
  subgraph BASE["configs/base.yaml"]
    B1["d_model: 384\nn_layers: 16\nn_heads: 8\nd_ff: 1536\nmax_length: 128\narch: lgt\ntokenizer: gpt2\ndataset: wikitext2\nstride: 10"]
  end

  subgraph VARIANTS["Variant configs (one field change each)"]
    V1["autoregressive/small.yaml\n  inherits: base.yaml\n  variant: autoregressive\n  arch: transformer\n  attention_mask: causal"]
    V2["diffusion/small.yaml\n  inherits: base.yaml\n  variant: diffusion\n  attention_mask: bidirectional"]
    V3["block_diffusion/small.yaml\n  inherits: base.yaml\n  variant: block_diffusion\n  attention_mask: block_causal\n  block_size: 16"]
    V4["flow_matching/small.yaml\n  inherits: base.yaml\n  variant: flow_matching\n  attention_mask: bidirectional"]
  end

  BASE --> V1 & V2 & V3 & V4
```
