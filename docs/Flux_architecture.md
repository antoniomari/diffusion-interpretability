# **FLUX.1 Schnell Architecture Overview**

The **FluxTransformer2DModel** is the core backbone of the **FLUX.1 Schnell** model.

---

## **📌 Model Components**
### **1. Positional & Context Embeddings**
- **FluxPosEmbed**: Provides 2D positional embeddings.
- **CombinedTimestepTextProjEmbeddings**: Merges **timestep embeddings** (used for diffusion) and **text embeddings** (guiding the image generation).
  - **Timestep Embeddings (TimestepEmbedding)**
    - Two linear layers with SiLU activation.
  - **Text Embeddings (PixArtAlphaTextProjection)**
    - Similar structure to timestep embeddings but for textual input.
- **Context & Input Embeddings**
  - **context_embedder**: Maps input text features (4096) to model dimensions (3072).
  - **x_embedder**: Projects input tokens (64) to model dimensions (3072).

---

### **2. Transformer Blocks**
#### **FluxTransformerBlock (19 Layers)**
Each transformer block consists of:
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

#### **FluxSingleTransformerBlock (38 Layers)**
- **AdaLayerNormZeroSingle**: Similar to AdaLayerNormZero but adapted for single-path attention.
- **Projection MLP**
  - Expands from **3072 → 12288** dimensions using GELU activation.
  - Outputs **15360 → 3072** features.
- **Self-Attention Mechanism**
  - Uses RMSNorm normalization.
  - Query (Q), Key (K), Value (V) projection layers.

---

### **3. Output Processing**
- **AdaLayerNormContinuous**
  - Applies SiLU activation.
  - Expands 3072 → 6144.
- **Projection Layer**
  - Final output layer projects from **3072 → 64**.

---

## **🔎 Summary**
| Component | Function |
|-----------|----------|
| **FluxPosEmbed** | Encodes positional information. |
| **Timestep & Text Embeddings** | Projects time steps and text prompts to a shared latent space. |
| **Transformer Blocks (19x)** | Multi-headed self-attention and feed-forward networks for feature processing. |
| **Single Transformer Blocks (38x)** | Lightweight transformer layers focusing on refinement. |
| **AdaLayerNorm & Attention** | Normalization and query-key-value attention mechanisms. |
| **Final Projection (proj_out)** | Outputs the final image feature representation. |

---

## **🚀 Key Takeaways**
✅ **Hybrid Transformer-Diffusion Model** optimized for fast inference.  
✅ **Lightweight and Efficient** with only **64 output features**.  
✅ **Advanced Attention Mechanisms** (RMSNorm + Multi-Head Projections).  
✅ **Adaptive Normalization** (AdaLayerNorm) for stability in training.  

---

This structured breakdown should help you understand the architecture at a glance. Let me know if you need any deeper insights! 🚀


## Computational Graph
H = height of latents

W = width of latents

ch = channels of latents

```mermaid
graph TD
    %% nodes
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
    hidden_states2 --> |AdaLayerNormContinuous| noise_pred_norm["noise_pred Normalized <br> (batch, H/2 * W/2, 3072)"];
    noise_pred_norm --> |Linear Projection| output["noise_pred_out <br> (batch, H/2 * W/2, 4 * ch)"]

    %% Apply colors to specific nodes
    style transformer_blocks fill:#ffcc00,stroke:#000,stroke-width:2px,color:#000;
    style single_transformer fill:#ffcc00,stroke:#000,stroke-width:2px,color:#000;

```
