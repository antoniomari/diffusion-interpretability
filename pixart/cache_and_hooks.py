from SDLens import HookedPixArtPipeline
import torch


def clear_hooks(pipe: HookedPixArtPipeline):

    # Clear hooks
    for i in range(len(pipe._locate_block("transformer.transformer_blocks"))):
        module._forward_pre_hooks.clear()
        pipe._locate_block(f"transformer.transformer_blocks.{i}")._forward_hooks.clear()


def run_with_cache(pipe: HookedPixArtPipeline, prompt: str, seed=42, num_inference_steps=1):

    assert isinstance(seed, int)
    seed = [seed]

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