import torch
from typing import Callable, Literal
from utils.hooks import ablate_block
from SDLens.hooked_sd_pipeline import HookedFluxPipeline
from utils.hooks import AttentionCacheForwardHook, TransformerActivationCache, AttentionAblationCacheHook


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

    LAYERS = [ (f"transformer.{block_type}.{i}", f"block {i}") for i in range(len(getattr(pipe.pipe.transformer, block_type)))]

    images = []
    titles = []

    with torch.no_grad():


        output = pipe.run_with_hooks(
            prompt,
            position_hook_dict={},
            num_inference_steps=1,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
            width=width,
            height=height,
        )

        images.append(output.images[0])
        titles.append("Original")

        for block, block_label in LAYERS:
                
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

            images.append(output_ablated.images[0])
            titles.append(f"Ablating {block_label}")

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


class Ablation:

    def __init__(self, ablator, vanilla_pre_forward_dict: Callable[[str, int], dict],
                                vanilla_forward_dict: Callable[[str, int], dict],
                                ablated_pre_forward_dict: Callable[[str, int], dict],
                                ablated_forward_dict: Callable[[str, int], dict],):
        self.ablator=ablator
        self.vanilla_seed = 42
        self.vanilla_pre_forward_dict = vanilla_pre_forward_dict
        self.vanilla_forward_dict = vanilla_forward_dict

        self.ablated_seed = 42
        self.ablated_pre_forward_dict = ablated_pre_forward_dict
        self.ablated_forward_dict = ablated_forward_dict
    
    def get_ablation(name: str, **kwargs):

        if name == "intermediate_text_stream_to_input":

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.cache_attention_activation(*args, full_output=True)},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": ablator.replace_text_stream_input},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        if name == "set_input_text":

            tensor: torch.Tensor = kwargs["tensor"]

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.replace_text_stream_input(*args, use_tensor=tensor)},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.clamp_output(*args)})

        if name == "replace_text_stream":
            ablator = AttentionAblationCacheHook()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0


            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.set_ablated_attention(*args, weight=weight)})
        
        if name == "input=output":
            return Ablation(None,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablate_block(*args)})
        
        if name == "reweight_text_stream": 
            ablator = TransformerActivationCache()

            residual_w=kwargs["residual_w"]
            activation_w=kwargs["activation_w"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.reweight_text_stream(*args, residual_w=residual_w, activation_w=activation_w)})
        

    


def layer_ablation(ablation: Ablation, prompt: str, i: int, vanilla_prompt: str = None, block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks",
                    empty_prompt_seed=42, prompt_seed=42, num_inference_steps: int = 1):
    
    pipe = PIPE

    if vanilla_prompt is None:
        vanilla_prompt = prompt

    vanilla_generator = torch.Generator(device="cpu").manual_seed(ablation.vanilla_seed)
    ablated_generator = torch.Generator(device="cpu").manual_seed(ablation.ablated_seed) 

    # with torch.autocast(device_type="cuda", dtype=dtype):
    with torch.no_grad():
            
            vanilla_forward_dict = ablation.vanilla_forward_dict(block_type=block_type, layer_num=i)
            vanilla_pre_forward_dict = ablation.vanilla_pre_forward_dict(block_type=block_type, layer_num=i)
            ablated_pre_forward_dict = ablation.ablated_pre_forward_dict(block_type=block_type, layer_num=i)
            ablated_forward_dict = ablation.ablated_forward_dict(block_type=block_type, layer_num=i)

            if len(vanilla_pre_forward_dict) + len(vanilla_forward_dict) > 0:
                _ = pipe.run_with_hooks(
                    prompt="",
                    prompt_2=vanilla_prompt,
                    position_hook_dict=vanilla_forward_dict,
                    position_pre_hook_dict=vanilla_pre_forward_dict,
                    with_kwargs=True,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,
                    generator=vanilla_generator,
                    width=1024,
                    height=1024,
                )

            output_ablated = pipe.run_with_hooks(
                prompt="",
                prompt_2=prompt,
                position_pre_hook_dict=ablated_pre_forward_dict,
                position_hook_dict=ablated_forward_dict,
                with_kwargs=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=ablated_generator,
                width=1024,
                height=1024,
            )
    
    return output_ablated


def ablate_attention_all_layers(ablation: Ablation, prompt: str, vanilla_prompt: str = None, block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks", empty_seed=42, num_inference_steps=1, weight=1,
                                return_cache=False):

    pipe = PIPE
    clear_hooks()

    NUM_LAYERS = 19 if block_type == "transformer_blocks" else 38

    images = []
    labels = []

    with torch.no_grad():
        output_ablated = pipe.run_with_hooks(
            prompt,
            position_pre_hook_dict={},
            position_hook_dict={},
            with_kwargs=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=torch.Generator(device="cpu").manual_seed(42),
            width=1024,
            height=1024,
        )

        images.append(output_ablated.images[0])
        labels.append("Original")

    caches = []
    for layer in range(0, NUM_LAYERS):
        output = layer_ablation(ablation, prompt, layer, vanilla_prompt=vanilla_prompt, block_type=block_type, num_inference_steps=num_inference_steps)

        if return_cache:
            caches.append(ablation.ablator.cache.copy())

        images.extend(output.images)
        labels.extend([f"Layer {layer} " + f"seed {empty_seed}"])

    if return_cache:
        return images, labels, caches
    else:
        return images, labels


