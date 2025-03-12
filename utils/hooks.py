import torch
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention import BasicTransformerBlock, FeedForward 

@torch.no_grad()
def add_feature(sae, feature_idx, value, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    mask[..., feature_idx] = value
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def add_feature_on_area(sae, feature_idx, activation_map, module, input, output):

    # adds feature to ouptut of the layer
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    if len(activation_map) == 2:
        activation_map = activation_map.unsqueeze(0)
    mask[..., feature_idx] = activation_map.to(mask.device)
    to_add = mask @ sae.decoder.weight.T
    return (output[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def replace_with_feature(sae, feature_idx, value, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    mask = torch.zeros_like(activated, device=diff.device)
    mask[..., feature_idx] = value
    to_add = mask @ sae.decoder.weight.T
    return (input[0] + to_add.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def reconstruct_sae_hook(sae, module, input, output):
    diff = (output[0] - input[0]).permute((0, 2, 3, 1)).to(sae.device)
    activated = sae.encode(diff)
    reconstructed = sae.decoder(activated) + sae.pre_bias
    return (input[0] + reconstructed.permute(0, 3, 1, 2).to(output[0].device),)


@torch.no_grad()
def ablate_block(*args):

    # Case 1: no kwards are passed to the module
    if len(args) == 3:
        module, input, output = args
    # Case 2: when kwargs are passed to the model as input
    elif len(args) == 4:
        module, input, kwinput, output = args

    # case the input is a tuple
    if isinstance(module, Transformer2DModel):
        return input
    elif isinstance(module, (BasicTransformerBlock, FeedForward)):
        return input[0]
    elif isinstance(module, FluxTransformerBlock):
        # Note; here kwinput is used to call forward
        return kwinput["encoder_hidden_states"], kwinput["hidden_states"]
    elif isinstance(module, FluxSingleTransformerBlock):
        # Note; here kwinput is used to call forward
        return kwinput["hidden_states"]

    else:
        print(type(input))
        print(len(input))
        print(input[0].shape, input[1].shape, output.shape)
        print(output)
        raise AssertionError(f"Block {module.__class__.__name__} not supported.")
