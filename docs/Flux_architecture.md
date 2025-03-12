# **FLUX.1 Schnell Architecture Overview**

## Computational Graph
Dimensions considered:
- `H` = height of latents
- `W` = width of latents
- `ch` = channels of latents
- the hidden dimension is 3072 (as reported in the graph)

```mermaid
graph TD

    time["timesteps <br> (T,)"] --> |"select timestep"| timestep["t <br> (B,)"];
    prompt1["Text prompt"] --> |"CLIP Encoder (pooled_output)"| prompt1_emb["pooled_prompt_embeds (batch, 768)"];
    prompt2["Text prompt 2"] --> |T5 Encoder| prompt2_emb["prompt_embeds <br> (batch, seq_len, 4096)"];
    latents["Latents <br> (batch, ch, H, W)"] --> | 2 x 2 patch | latents_patch["Latents patched <br> (batch, H/2 * W/2, 4 * ch)"];
    prompt2_emb --> txt_img_ids["concat(txt_ids, img_ids) (shape?)"];
    latents_patch --> txt_img_ids;


    %% transformer
    plus["\+"]
    time_text_emb["temb <br> (B, 3072)"];
    image_rotary_emb["image_rotary_emb (shape?)"];
    transformer_blocks["Transformer blocks (x19)"]
    latents_patch --> |"x_embedder (Linear)"| hidden_states["hidden_states <br> (B, H/2 * W/2, 3072)"]
    timestep --> |"sinusoidal emb <br> Linear -> SiLU -> Linear"| time_emb["(B, 3072)"];
    prompt1_emb --> |Linear -> SiLU -> Linear| pooled_text_emb["(B, 3072)"];
    pooled_text_emb --> plus;
    time_emb --> plus; 
    plus --> time_text_emb;
    prompt2_emb --> |"context_emb (Linear)"| encoder_hidden_states["encoder_hidden_states <br> (batch, seq_len, 3072)"];
    txt_img_ids --> |FluxPositionalEmb | image_rotary_emb
    hidden_states --> transformer_blocks;
    encoder_hidden_states --> transformer_blocks;
    time_text_emb --> transformer_blocks;
    image_rotary_emb --> transformer_blocks;
    transformer_blocks --> encoder_hidden_states1["encoder_hidden_states (batch, seq_len, 3072)"]
    transformer_blocks --> hidden_states1["hidden_states (batch, H/2 * W/2, 3072)"]
    encoder_hidden_states1 --> concat_states["concat(enc_hid, hid) <br> (batch, seq_len + H/2 * W/2, 3072)"]
    hidden_states1 --> concat_states

    concat_states --> single_transformer["Single transformer blocks (x38)"]
    time_text_emb --> single_transformer
    image_rotary_emb --> single_transformer
    single_transformer --> |"slice output [:, seq_len:, :]"| hidden_states2["noise_pred <br> (batch, H/2 * W/2, 3072)"]
    hidden_states2 --> |"AdaLayerNormContinuous (depends on temb)"| noise_pred_norm["noise_pred Normalized <br> (batch, H/2 * W/2, 3072)"];
    time_text_emb --> noise_pred_norm
    noise_pred_norm --> |Linear Projection| output["noise_pred_out <br> (batch, H/2 * W/2, 4 * ch)"]

    %% Apply colors to specific nodes
    style transformer_blocks fill:#ffcc00,stroke:#000,stroke-width:2px,color:#000;
    style single_transformer fill:#ffcc00,stroke:#000,stroke-width:2px,color:#000;

    classDef input_prep fill:#e0f7fa,stroke:#000,stroke-width:2px,color:#000;
    class time,timestep,prompt1,prompt1_emb,prompt2,prompt2_emb,latents,latents_patch,txt_img_ids input_prep;

    classDef transformer_nodes fill:#fff9c4,stroke:#000,stroke-width:2px,color:#000;
    class plus,time_text_emb,pooled_text_emb,time_emb,image_rotary_emb,hidden_states,encoder_hidden_states,encoder_hidden_states1,hidden_states1,concat_states,hidden_states2,noise_pred_norm,output transformer_nodes

```
Pay attention to output **AdaLayerNormContinuous(temb, x)**:
1. Applies SiLU(temb) -> Linear (3072 → 6144) -> split in `scale`|`shift` (both with shape `(batch, 3072)`)
2. outputs `standardize(x) * (1 + scale)[:, None, :] + shift[:, None, :]`


## Transformer Blocks (19 layers)
These blocks are introduced in the [SD-3 paper](https://arxiv.org/abs/2403.03206). TODO
```mermaid
graph TD


```

In particular, pay attention to:
- **AdaLayerNormZero**
  - **Adaptive Layer Normalization**
  - Uses SiLU activation.
  - Projects features from 3072 → 18432 dimensions.
- **Attention Mechanism**
  - Uses **RMSNorm** for normalizing query (Q), key (K), and value (V).
  - Multiple **projection layers** (to_q, to_k, to_v).
  - **Output Layers**
    - `to_out`: Two linear layers with dropout.
    - `to_add_out`: Additional projection for residuals.
- **Feed-Forward Networks (FFN)**
  - Uses **GELU activation**.
  - Expands features from **3072 → 12288 → 3072**.




## Single Transformer Blocks
These blocks are introduced in the [DiT](https://arxiv.org/abs/2212.09748) paper.

In particular, pay attention to:
- **AdaLayerNormZeroSingle**: Similar to AdaLayerNormZero but adapted for single-path attention.
- **Projection MLP**
  - Expands from **3072 → 12288** dimensions using GELU activation.
  - Outputs **15360 → 3072** features.
- **Self-Attention Mechanism**
  - Uses RMSNorm normalization.
  - Query (Q), Key (K), Value (V) projection layers.
