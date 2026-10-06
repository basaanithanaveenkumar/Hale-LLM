# Architecture

Mermaid diagrams (they render on GitHub). They match the [paper](../paper/main.tex) and the
[project page](../project-page/index.html).

## 1. Paradigms as plugins

```mermaid
flowchart TB
  Y["configs/&lt;variant&gt;/small.yaml<br/>inherits: ../base.yaml"] --> R{"registries<br/>variant · collate · loss · sampler · metric"}
  R --> AR["models_autoregressive"]
  R --> MD["models_diffusion"]
  R --> BD["models_block_diffusion"]
  R --> FM["models_flow_matching"]
  R --> MTP["models_mtp (stub)"]
  AR & MD & BD & FM & MTP --> CORE["shared: core backbones · WikiText-2 + GPT-2 windows · trainer · evaluator · inference · GIF visualiser"]
```

## 2. Training objectives

```mermaid
flowchart LR
  X["x (128 GPT-2 tokens)"] --> ARc["AR: causal mask<br/>labels shifted by 1"] --> ARl["CE(next token)"]
  X --> MDc["Masked diffusion: t ~ U(ε,1)<br/>mask each token w.p. t"] --> MDl["CE on masked × 1/t"]
  X --> BDc["Block diffusion: t_k per 16-token block<br/>input = [clean ; noisy], block-causal mask"] --> BDl["CE on masked noisy × 1/t_k"]
  X --> FMc["Flow matching: t ~ U(0,1−ε)<br/>mask w.p. 1−t"] --> FMl["CE on masked × 1/(1−t)<br/>(= MD with s = 1−t)"]
```

## 3. Block diffusion: attention and decoding

```mermaid
flowchart LR
  subgraph Attention["who attends to whom"]
    direction TB
    c["clean block k"] -->|"blocks ≤ k"| cc["clean"]
    n["noisy block k"] -->|"blocks &lt; k"| cc
    n -->|"block k only"| nn["noisy"]
  end
  subgraph Decode["sampling"]
    direction TB
    p["prompt blocks fixed"] --> b1["block j: all [MASK]"]
    b1 --> s1["S steps, t: 1 → 0<br/>reveal w.p. (t − s)/t"]
    s1 --> f["freeze block j"] --> b2["block j + 1"]
  end
```

## 4. Shared backbone

```mermaid
flowchart TB
  IDS["token ids"] --> EMB["token embedding (tied)<br/>+ learned positions (GPT / DiT)"]
  T["diffusion time t"] --> TE["sinusoidal → MLP → t_emb"]
  EMB --> ST{"model.arch"}
  ST -->|transformer| G["GPTStack: pre-LN, MHA, MLP"]
  ST -->|lgt| L["LGTStack: RMSNorm, GQA, 5 local : 1 global,<br/>RoPE θ 10⁴ / 10⁶, GeGLU"]
  ST -->|dit| D["DiTStack: AdaLN-Zero"]
  TE -. "scale/shift norms" .-> L
  TE -. "AdaLN" .-> D
  G & L & D --> H["LM head → logits (B, S, 50,258)"]
```

## Sizes (measured, `configs/base.yaml`)

| Paradigm | Backbone | Params |
|---|---|---|
| AR | GPT, MLP, MHA | 47.7M |
| Diffusion / block diffusion / flow matching | LGT, GeGLU, GQA n_kv = 2, time-conditioned | 63.2M |
