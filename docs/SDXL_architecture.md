
Input: 4 channels (3 RGB + 1 Noise)
Convolution: 3x3 -> maps the 4 channels into 320 channels
time_proj: encodes the diffusion timestep  (e.g. sinusoidal or learned embedding -> high dimensional representation 320) 
time_embedding(time_proj): 2layers MLP (SiLU) -> goes to 1280 features
add_time_proj: MAYBE residual connection on time_proj (time_proj + time_embedding(time_proj)) OR Timestep embedding from text-conditioning part
add_embedding: same 2layers MLP (SiLU) -> from 2816 to 1280 features

## Down blocks
1. DownBlock2D
2. CrossAttnDownBlock2D
3. ..


### DownBlock2D
2 x ResnetBlock2D (GroupNorm)
Downsample2D: strided convolution  (3x3, stride is 2x2) keeps 320 channels

### CrossAttnDownBlock2D
2 x Transformer2DModel -> enhances feature extraction across different image regions (U-net convolutional layer focus on local features instead)
2 x ResNetBlock (first conv_shortcut, second no) -> output 640 channels
downsample2D: Conv2d(640, 640) 3x3 with stride 2x2
    (downsamplers): ModuleList(
    (0): Downsample2D(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    )

### Transformer2DModel
GroupNorm
proj_in: 640 -> 640
2xTransformer (LayerNorm -> Self-Attention (640) -> LayerNorm -> Cross-Attention (self (Q) is 640, external (K, V) is 2048) -> LayerNorm -> MLP* )

This MLP: 
1. GEGLU(x (dim=640)): proj to 5120 -> gelu(x[:2560]) * x[2560:] (dim=2560) (gelu(x) = x * phi(x), where phi(x) is standard Gaussian CDF)
2. proj 2560 -> 640


                    Transformer2DModel(
                        num_attention_heads,
                        out_channels // num_attention_heads,
                        in_channels=out_channels,
                        num_layers=transformer_layers_per_block[i],
                        cross_attention_dim=cross_attention_dim,
                        norm_num_groups=resnet_groups,
                        use_linear_projection=use_linear_projection,
                        only_cross_attention=only_cross_attention,
                        upcast_attention=upcast_attention,
                        attention_type=attention_type,
                    )


### ResNetBlock
GroupNormalization -> normalize feature maps across groupof channels  (normalizes 320 chanells into 32 groups)
conv1: 320 to 640 channels (3x3)
time_emb_proj(time_emb): proj 1280 to 640 # injects temporal info
conv1 + time_emb_proj
groupnorm2
conv2d
silu (x * sigmoid(x))
conv_shortcut: Conv2d(320, 640, kernel_size=(1, 1)) -> skip connection with conv1x1 to ensure output-shape consistency (input channels were 320)

Second-resnet is the same but since input channels are 640 then there is no need to conv_shortcut









(down_blocks): ModuleList(
(0): DownBlock2D(
    (resnets): ModuleList(
    (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv1): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
    )
    )
    (downsamplers): ModuleList(
    (0): Downsample2D(
        (conv): Conv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    )
)
(1): CrossAttnDownBlock2D(
    (attentions): ModuleList(
    (0-1): 2 x Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
        (0-1): 2 x BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=2048, out_features=640, bias=False)
            (to_v): Linear(in_features=2048, out_features=640, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
            (net): ModuleList(
                (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
            )
            )
        )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
    )
    )
    (resnets): ModuleList(
    (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 320, eps=1e-05, affine=True)
        (conv1): Conv2d(320, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(320, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
    )
    )
    (downsamplers): ModuleList(
    (0): Downsample2D(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    )
    )
)
(2): CrossAttnDownBlock2D(
    (attentions): ModuleList(
    (0-1): 2 x Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
        (0-9): 10 x BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=2048, out_features=1280, bias=False)
            (to_v): Linear(in_features=2048, out_features=1280, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
            (net): ModuleList(
                (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
            )
        )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
    )
    )
    (resnets): ModuleList(
    (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(640, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
    )
    )
)
)
(up_blocks): ModuleList(
(0): CrossAttnUpBlock2D(
    (attentions): ModuleList(
    (0-2): 3 x Transformer2DModel(
        (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
        (transformer_blocks): ModuleList(
        (0-9): 10 x BasicTransformerBlock(
            (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=2048, out_features=1280, bias=False)
            (to_v): Linear(in_features=2048, out_features=1280, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=1280, out_features=1280, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
            (net): ModuleList(
                (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
            )
        )
        )
        (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
    )
    )
    (resnets): ModuleList(
    (0-1): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 2560, eps=1e-05, affine=True)
        (conv1): Conv2d(2560, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(2560, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv1): Conv2d(1920, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
        (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1920, 1280, kernel_size=(1, 1), stride=(1, 1))
    )
    )
    (upsamplers): ModuleList(
    (0): Upsample2D(
        (conv): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    )
)
(1): CrossAttnUpBlock2D(
    (attentions): ModuleList(
    (0-2): 3 x Transformer2DModel(
        (norm): GroupNorm(32, 640, eps=1e-06, affine=True)
        (proj_in): Linear(in_features=640, out_features=640, bias=True)
        (transformer_blocks): ModuleList(
        (0-1): 2 x BasicTransformerBlock(
            (norm1): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn1): Attention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=640, out_features=640, bias=False)
            (to_v): Linear(in_features=640, out_features=640, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm2): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (attn2): Attention(
            (to_q): Linear(in_features=640, out_features=640, bias=False)
            (to_k): Linear(in_features=2048, out_features=640, bias=False)
            (to_v): Linear(in_features=2048, out_features=640, bias=False)
            (to_out): ModuleList(
                (0): Linear(in_features=640, out_features=640, bias=True)
                (1): Dropout(p=0.0, inplace=False)
            )
            )
            (norm3): LayerNorm((640,), eps=1e-05, elementwise_affine=True)
            (ff): FeedForward(
            (net): ModuleList(
                (0): GEGLU(
                (proj): Linear(in_features=640, out_features=5120, bias=True)
                )
                (1): Dropout(p=0.0, inplace=False)
                (2): Linear(in_features=2560, out_features=640, bias=True)
            )
            )
        )
        )
        (proj_out): Linear(in_features=640, out_features=640, bias=True)
    )
    )
    (resnets): ModuleList(
    (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 1920, eps=1e-05, affine=True)
        (conv1): Conv2d(1920, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1920, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (1): ResnetBlock2D(
        (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
        (conv1): Conv2d(1280, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(1280, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    (2): ResnetBlock2D(
        (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv1): Conv2d(960, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=640, bias=True)
        (norm2): GroupNorm(32, 640, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(960, 640, kernel_size=(1, 1), stride=(1, 1))
    )
    )
    (upsamplers): ModuleList(
    (0): Upsample2D(
        (conv): Conv2d(640, 640, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    )
    )
)
(2): UpBlock2D(
    (resnets): ModuleList(
    (0): ResnetBlock2D(
        (norm1): GroupNorm(32, 960, eps=1e-05, affine=True)
        (conv1): Conv2d(960, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(960, 320, kernel_size=(1, 1), stride=(1, 1))
    )
    (1-2): 2 x ResnetBlock2D(
        (norm1): GroupNorm(32, 640, eps=1e-05, affine=True)
        (conv1): Conv2d(640, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (time_emb_proj): Linear(in_features=1280, out_features=320, bias=True)
        (norm2): GroupNorm(32, 320, eps=1e-05, affine=True)
        (dropout): Dropout(p=0.0, inplace=False)
        (conv2): Conv2d(320, 320, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (nonlinearity): SiLU()
        (conv_shortcut): Conv2d(640, 320, kernel_size=(1, 1), stride=(1, 1))
    )
    )
)
)
(mid_block): UNetMidBlock2DCrossAttn(
(attentions): ModuleList(
    (0): Transformer2DModel(
    (norm): GroupNorm(32, 1280, eps=1e-06, affine=True)
    (proj_in): Linear(in_features=1280, out_features=1280, bias=True)
    (transformer_blocks): ModuleList(
        (0-9): 10 x BasicTransformerBlock(
        (norm1): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attn1): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=1280, out_features=1280, bias=False)
            (to_v): Linear(in_features=1280, out_features=1280, bias=False)
            (to_out): ModuleList(
            (0): Linear(in_features=1280, out_features=1280, bias=True)
            (1): Dropout(p=0.0, inplace=False)
            )
        )
        (norm2): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (attn2): Attention(
            (to_q): Linear(in_features=1280, out_features=1280, bias=False)
            (to_k): Linear(in_features=2048, out_features=1280, bias=False)
            (to_v): Linear(in_features=2048, out_features=1280, bias=False)
            (to_out): ModuleList(
            (0): Linear(in_features=1280, out_features=1280, bias=True)
            (1): Dropout(p=0.0, inplace=False)
            )
        )
        (norm3): LayerNorm((1280,), eps=1e-05, elementwise_affine=True)
        (ff): FeedForward(
            (net): ModuleList(
            (0): GEGLU(
                (proj): Linear(in_features=1280, out_features=10240, bias=True)
            )
            (1): Dropout(p=0.0, inplace=False)
            (2): Linear(in_features=5120, out_features=1280, bias=True)
            )
        )
        )
    )
    (proj_out): Linear(in_features=1280, out_features=1280, bias=True)
    )
)
(resnets): ModuleList(
    (0-1): 2 x ResnetBlock2D(
    (norm1): GroupNorm(32, 1280, eps=1e-05, affine=True)
    (conv1): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (time_emb_proj): Linear(in_features=1280, out_features=1280, bias=True)
    (norm2): GroupNorm(32, 1280, eps=1e-05, affine=True)
    (dropout): Dropout(p=0.0, inplace=False)
    (conv2): Conv2d(1280, 1280, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (nonlinearity): SiLU()
    )
)
)
(conv_norm_out): GroupNorm(32, 320, eps=1e-05, affine=True)
(conv_act): SiLU()
(conv_out): Conv2d(320, 4, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
