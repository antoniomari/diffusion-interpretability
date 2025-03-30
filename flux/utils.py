import torch
from typing import Callable, List, Literal
from utils.hooks import ablate_block
from SDLens.hooked_sd_pipeline import HookedFluxPipeline
from utils.hooks import AttentionCacheForwardHook, TransformerActivationCache, AttentionAblationCacheHook
import matplotlib.pyplot as plt
from PIL import Image


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



def plot_images_grid(image_rows, title_rows, nrows, ncols, figsize=(10, 10), title: str = ""):
    """
    Plots a grid of images with corresponding titles from a list of lists.

    :param image_rows: List of lists containing PIL.Image.Image objects (each inner list is a row)
    :param title_rows: List of lists containing titles corresponding to the images
    :param figsize: Tuple specifying figure size
    """

    image_rows = [image_rows[ncols * j : ncols*(j+1)] for j in range(nrows)]
    title_rows = [title_rows[ncols * j : ncols*(j+1)] for j in range(nrows)]

    rows = len(image_rows)  # Number of rows
    cols = max(len(row) for row in image_rows)  # Maximum number of columns

    fig, axes = plt.subplots(rows, cols, figsize=figsize)

    # Ensure axes is always a 2D array, even if there's only one row or column
    if rows == 1:
        axes = [axes]  # Convert 1D array to 2D list
    if cols == 1:
        axes = [[ax] for ax in axes]  # Convert 1D array to 2D list

    for r, (img_row, title_row) in enumerate(zip(image_rows, title_rows)):
        for c, (img, title) in enumerate(zip(img_row, title_row)):
            axes[r][c].imshow(img)
            axes[r][c].set_title(title)
            axes[r][c].axis("off")

    # Hide unused subplots (in case of uneven rows)
    for r in range(rows):
        for c in range(len(image_rows[r]), cols):
            axes[r][c].axis("off")
    
    plt.title(title)
    plt.tight_layout()
    plt.show()



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
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, stream="text")},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        elif name == "set_input_text":

            tensor: torch.Tensor = kwargs["tensor"]

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.replace_stream_input(*args, use_tensor=tensor, stream="text")},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.clamp_output(*args)})

        elif name == "replace_text_stream_activation":
            ablator = AttentionAblationCacheHook()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0


            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.set_ablated_attention(*args, weight=weight)})
        
        elif name == "replace_text_stream":
            ablator = TransformerActivationCache()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {})
 
        
        elif name == "input=output":
            return Ablation(None,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablate_block(*args)})
        
        elif name == "reweight_text_stream": 
            ablator = TransformerActivationCache()

            residual_w=kwargs["residual_w"]
            activation_w=kwargs["activation_w"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.reweight_text_stream(*args, residual_w=residual_w, activation_w=activation_w)})
        
        elif name == "add_input_text":

            tensor: torch.Tensor = kwargs["tensor"]

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.add_text_stream_input(*args, use_tensor=tensor)},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.0": lambda *args: ablator.clamp_output(*args)})

        elif name == "nothing":
            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {})
        
        elif name == "reweight_image_stream": 
            ablator = TransformerActivationCache()
            residual_w=kwargs["residual_w"]
            activation_w=kwargs["activation_w"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.reweight_image_stream(*args, residual_w=residual_w, activation_w=activation_w)})
        
        if name == "intermediate_image_stream_to_input":

            ablator = TransformerActivationCache()
            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": lambda *args: ablator.cache_attention_activation(*args, full_output=True)},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, stream='image')},
                            ablated_forward_dict=lambda block_type, layer_num: {})


        elif name == "replace_text_stream_one_layer":
            ablator = AttentionAblationCacheHook()
            weight = kwargs["weight"] if "weight" in kwargs else 1.0


            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_text_stream},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.cache_and_inject_pre_forward},
                            ablated_forward_dict=lambda block_type, layer_num: {f"transformer.{block_type}.{layer_num}": ablator.restore_text_stream})

        elif name == "replace_intermediate_representation":
            ablator = TransformerActivationCache()
            tensor: torch.Tensor = kwargs["tensor"]

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.0": lambda *args: ablator.replace_stream_input(*args, use_tensor=tensor, stream='text_image')},
                            ablated_forward_dict=lambda block_type, layer_num: {})

        elif name == "destroy_registers":
            ablator = TransformerActivationCache()
            layers: List[int] = kwargs['layers']
            k: float = kwargs["k"]
            stream: str = kwargs['stream']
            random: bool = kwargs["random"] if "random" in kwargs else False
            lowest_norm: bool = kwargs["lowest_norm"] if "lowest_norm" in kwargs else False

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num: {},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.destroy_registers(*args,  k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            ablated_forward_dict=lambda block_type, layer_num: {})
        
        elif name == "patch_registers":
            ablator = TransformerActivationCache()
            layers: List[int] = kwargs['layers']
            k: float = kwargs["k"]
            stream: str = kwargs['stream']
            random: bool = kwargs["random"] if "random" in kwargs else False
            lowest_norm: bool = kwargs["lowest_norm"] if "lowest_norm" in kwargs else False

            return Ablation(ablator,
                            vanilla_pre_forward_dict=lambda block_type, layer_num:  {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.destroy_registers(*args, k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            vanilla_forward_dict=lambda block_type, layer_num: {},
                            ablated_pre_forward_dict=lambda block_type, layer_num: {f"transformer.single_transformer_blocks.{i}": lambda *args: ablator.set_cached_registers(*args, k=k, stream=stream, random_ablation=random, lowest_norm=lowest_norm) for i in layers},
                            ablated_forward_dict=lambda block_type, layer_num: {})



        

def layer_ablation(ablation: Ablation, prompt: str, i: int, vanilla_prompt: str = None, block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks",
                    vanilla_seed: list = [42], ablated_seed: list = [42], num_inference_steps: int = 1):
    
    pipe = PIPE

    # Create generators
    vanilla_gen = [torch.Generator(device="cpu").manual_seed(seed) for seed in vanilla_seed]
    ablated_gen = [torch.Generator(device="cpu").manual_seed(seed) for seed in ablated_seed]

    if vanilla_prompt is None:
        vanilla_prompt = prompt

    # with torch.autocast(device_type="cuda", dtype=dtype):
    with torch.no_grad():
            
            vanilla_forward_dict = ablation.vanilla_forward_dict(block_type=block_type, layer_num=i)
            vanilla_pre_forward_dict = ablation.vanilla_pre_forward_dict(block_type=block_type, layer_num=i)
            ablated_pre_forward_dict = ablation.ablated_pre_forward_dict(block_type=block_type, layer_num=i)
            ablated_forward_dict = ablation.ablated_forward_dict(block_type=block_type, layer_num=i)

            if len(vanilla_pre_forward_dict) + len(vanilla_forward_dict) > 0:
                _ = pipe.run_with_hooks(
                    prompt=[""] * len(vanilla_gen),
                    prompt_2=[vanilla_prompt] * len(vanilla_gen),
                    position_hook_dict=vanilla_forward_dict,
                    position_pre_hook_dict=vanilla_pre_forward_dict,
                    with_kwargs=True,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,
                    generator=vanilla_gen,
                    width=1024,
                    height=1024,
                )

            output_ablated = pipe.run_with_hooks(
                prompt=[""] * len(ablated_gen),
                prompt_2=[prompt] * len(ablated_gen),
                position_pre_hook_dict=ablated_pre_forward_dict,
                position_hook_dict=ablated_forward_dict,
                with_kwargs=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=ablated_gen,
                width=1024,
                height=1024,
            )
    
    return output_ablated


def ablate_attention_all_layers(ablation: Ablation, 
                                prompt: str, 
                                vanilla_prompt: str = None, 
                                block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks", 
                                vanilla_seed=42, 
                                ablated_seed=42,
                                num_inference_steps=1, 
                                return_cache=False):

    if isinstance(vanilla_seed, int):
        vanilla_seed = [vanilla_seed]
    
    if isinstance(ablated_seed, int):
        ablated_seed = [ablated_seed]

    # In case one has multiple values, the other will be repeated
    # e.g. [0, 1, 2], [4] becomes [0, 1, 2], [4, 4, 4]
    if len(vanilla_seed) > 1 and len(ablated_seed) == 1:
        ablated_seed *= len(vanilla_seed)
    elif len(ablated_seed) > 1 and len(vanilla_seed) == 1:
        vanilla_seed *= len(ablated_seed)
    else: 
        assert len(vanilla_seed) == len(ablated_seed)


    pipe = PIPE
    clear_hooks()

    NUM_LAYERS = 19 if block_type == "transformer_blocks" else 38

    images = []
    labels = []

    with torch.no_grad():
        output_ablated = pipe.run_with_hooks(
            [prompt] * len(ablated_seed),
            position_pre_hook_dict={},
            position_hook_dict={},
            with_kwargs=True,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            generator=[torch.Generator(device="cpu").manual_seed(seed) for seed in ablated_seed],
            width=1024,
            height=1024,
        )

        images.extend(output_ablated.images)
        labels.extend([f"Original (seed {seed})" for seed in ablated_seed])

    caches = []
    for layer in range(0, NUM_LAYERS):
        output = layer_ablation(ablation, prompt, layer, vanilla_prompt=vanilla_prompt, block_type=block_type, num_inference_steps=num_inference_steps,
                                vanilla_seed=vanilla_seed, ablated_seed=ablated_seed)

        if return_cache:
            caches.append(ablation.ablator.cache.copy())

        images.extend(output.images)
        labels.extend([f"Layer {layer} " + f"seed {seed}" for seed in ablated_seed])

    if return_cache:
        return images, labels, caches
    else:
        return images, labels



def single_layer_ablation_with_cache(ablation: Ablation, 
                                     prompt: str, 
                                     layer: int,
                                     vanilla_prompt: str = None, 
                                     block_type: Literal["transformer_blocks", "single_transformer_blocks"] = "transformer_blocks", 
                                     vanilla_seed=42, 
                                     ablated_seed=42,
                                     num_inference_steps=1):

    assert isinstance(vanilla_seed, int)
    assert isinstance(ablated_seed, int)
    vanilla_seed = [vanilla_seed]
    ablated_seed = [ablated_seed]

    pipe = PIPE
    clear_hooks()

    vanilla_forward_dict = ablation.vanilla_forward_dict(block_type=block_type, layer_num=layer)
    vanilla_pre_forward_dict = ablation.vanilla_pre_forward_dict(block_type=block_type, layer_num=layer)
    ablated_pre_forward_dict = ablation.ablated_pre_forward_dict(block_type=block_type, layer_num=layer)
    ablated_forward_dict = ablation.ablated_forward_dict(block_type=block_type, layer_num=layer)
    assert isinstance(ablation.ablator, TransformerActivationCache)
    ablator: TransformerActivationCache = ablation.ablator

    # insert cache storing in dict
    for block_type in 'transformer_blocks', "single_transformer_blocks":
        NUM_LAYERS = 19 if block_type == "transformer_blocks" else 38
        for i in range(NUM_LAYERS):
            
            if f"transformer.{block_type}.{i}" in ablated_forward_dict:
                existing_hooks = ablated_forward_dict[f"transformer.{block_type}.{i}"]
                if isinstance(existing_hooks, list):
                    ablated_forward_dict[f"transformer.{block_type}.{i}"] = existing_hooks.append(ablator.cache_residual_and_activation)
                else:
                    ablated_forward_dict[f"transformer.{block_type}.{i}"] = [existing_hooks, ablator.cache_residual_and_activation]
            else:
                ablated_forward_dict[f"transformer.{block_type}.{i}"] = [ablator.cache_residual_and_activation]

        
    # Create generators
    vanilla_gen = [torch.Generator(device="cpu").manual_seed(seed) for seed in vanilla_seed]
    ablated_gen = [torch.Generator(device="cpu").manual_seed(seed) for seed in ablated_seed]

    if vanilla_prompt is None:
        vanilla_prompt = prompt

    # with torch.autocast(device_type="cuda", dtype=dtype):
    with torch.no_grad():
            
            if len(vanilla_pre_forward_dict) + len(vanilla_forward_dict) > 0:
                _ = pipe.run_with_hooks(
                    prompt=[""] * len(vanilla_gen),
                    prompt_2=[vanilla_prompt] * len(vanilla_gen),
                    position_hook_dict=vanilla_forward_dict,
                    position_pre_hook_dict=vanilla_pre_forward_dict,
                    with_kwargs=True,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=0.0,
                    generator=vanilla_gen,
                    width=1024,
                    height=1024,
                )

            output_ablated = pipe.run_with_hooks(
                prompt=[""] * len(ablated_gen),
                prompt_2=[prompt] * len(ablated_gen),
                position_pre_hook_dict=ablated_pre_forward_dict,
                position_hook_dict=ablated_forward_dict,
                with_kwargs=True,
                num_inference_steps=num_inference_steps,
                guidance_scale=0.0,
                generator=ablated_gen,
                width=1024,
                height=1024,
            )
    
    return output_ablated, ablator.cache_lists



