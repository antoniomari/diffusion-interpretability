import torch
from typing import Literal
from utils.hooks import ablate_block
from SDLens.hooked_sd_pipeline import HookedFluxPipeline
from utils.hooks import AttentionCacheForwardHook


PROMPT = "A cinematic shot of a professor sloth wearing a tuxedo at a BBQ party."
BLOCK =  "unet.down_blocks.2.attentions.1.transformer_blocks.0.ff" # "unet.down_blocks.2.attentions.1"

PIPE: HookedFluxPipeline = None
DTYPE: torch.dtype = None

def set_flux_context(pipe: HookedFluxPipeline, dtype: torch.dtype):
    global PIPE 
    global DTYPE 
    PIPE = pipe
    DTYPE = dtype


def clear_hooks():

    pipe = PIPE

     # Clear hooks
    for i in range(len(pipe._locate_block("transformer.transformer_blocks"))):
        pipe._locate_block(f"transformer.transformer_blocks.{i}")._forward_hooks.clear()
        

    for i in range(len(pipe._locate_block("transformer.single_transformer_blocks"))):
        pipe._locate_block(f"transformer.single_transformer_blocks.{i}")._forward_hooks.clear()


def ablate_transformer_blocks( prompt="A cinematic shot of a professor sloth wearing a tuxedo at a BBQ party.",
                              block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks",
                              width=1024, height=1024):

    pipe = PIPE
    dtype = DTYPE
    clear_hooks()

    LAYERS = [ (f"transformer.{block_type}.{i}", f"TB{i}") for i in range(len(getattr(pipe.pipe.transformer, block_type)))]

    with torch.no_grad():

        with torch.autocast(device_type="cuda", dtype=dtype):
            output = pipe.run_with_hooks(
                prompt,
                position_hook_dict={},
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=torch.Generator(device="cpu").manual_seed(42),
                width=width,
                height=height,
            )
            images = [[output.images[0]]] 
            titles = [["Output"]] 


            for block, block_label in LAYERS:
                    
                    print({ block: ablate_block })

                    output_ablated = pipe.run_with_hooks(
                        prompt,
                        position_hook_dict={ block: ablate_block },
                        with_kwargs=True,
                        num_inference_steps=1,
                        guidance_scale=0.0,
                        generator=torch.Generator(device="cpu").manual_seed(42),
                        width=width,
                        height=height,
                    )

                    images[0].append(output_ablated.images[0])
                    titles[0].append(f"Ablating {block_label}")

    return images, titles


def ablate_block_chunk(prompt="A cinematic shot of a professor sloth wearing a tuxedo at a BBQ party.",
                        block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks",
                        blocks_idx=[0], width=1024, height=1024):
    
    pipe = PIPE
    dtype = DTYPE

    assert all(idx <= len(getattr(pipe.pipe.transformer, block_type)) for idx in blocks_idx)
    clear_hooks()
    blocks_idx = list(set(blocks_idx))

    # ablation hooks
    position_hook_dict = {f"transformer.{block_type}.{i}": ablate_block for i in blocks_idx}

    print(position_hook_dict)

    with torch.autocast(device_type="cuda", dtype=dtype):
        with torch.no_grad():

            output_ablated = pipe.run_with_hooks(
                prompt,
                position_hook_dict=position_hook_dict,
                with_kwargs=True,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=torch.Generator(device="cpu").manual_seed(42),
                width=width,
                height=height,
            )

    return output_ablated.images[0], f"Ablating {blocks_idx}"


# ablation hooks


def activation_patching(prompt: str, i: int, block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks",
                        encoder_hidden_states: bool = False, empty_prompt_seed=42, prompt_seed=42):
    
    pipe = PIPE
    dtype = DTYPE
    
    if type(empty_prompt_seed) == list:
        empty_generators = [torch.Generator(device="cpu").manual_seed(j) for j in empty_prompt_seed]
        empty_prompt = [""] * len(empty_prompt_seed)
        prompt = [prompt] * len(empty_prompt_seed)
    else:
        empty_generators = torch.Generator(device="cpu").manual_seed(empty_prompt_seed)
        empty_prompt = ""

    if type(prompt_seed) == list:
        generators = [torch.Generator(device="cpu").manual_seed(j) for j in prompt_seed]
    else:
        generators = torch.Generator(device="cpu").manual_seed(prompt_seed)

    with torch.autocast(device_type="cuda", dtype=dtype):
        with torch.no_grad():
            attn_cache = AttentionCacheForwardHook()

            output_empty_prompt = pipe.run_with_hooks(
                prompt=empty_prompt,
                position_hook_dict={f"transformer.{block_type}.{i}": attn_cache.get_attention_output},
                with_kwargs=True,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=empty_generators,
                width=1024,
                height=1024,
            )


            output_ablated = pipe.run_with_hooks(
                prompt=prompt,
                position_hook_dict={f"transformer.{block_type}.{i}": lambda *args: attn_cache.set_attention_output(*args, encoder_hidden_states=encoder_hidden_states)},
                with_kwargs=True,
                num_inference_steps=1,
                guidance_scale=0.0,
                generator=generators,
                width=1024,
                height=1024,
            )
    
    return output_ablated