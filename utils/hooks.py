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


class AttentionCacheForwardHook:

    def __init__(self):
        self.cache = {}

    # Define a hook function
    @torch.no_grad()
    def get_attention_output(self, *args):

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            self.cache["encoder_hidden_states"] = output[0] - kwinput["encoder_hidden_states"]
            self.cache["hidden_states"] = output[1] - kwinput["hidden_states"]
        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["hidden_states"] = output - kwinput["hidden_states"]

        
    @torch.no_grad()
    def set_attention_output(self, *args, encoder_hidden_states=False):

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):

            ablated_hidden_states = kwinput["hidden_states"] + self.cache["hidden_states"]

            if encoder_hidden_states:
                ablated_encoder_hidden_states = kwinput["encoder_hidden_states"] + self.cache["encoder_hidden_states"]
            else: 
                ablated_encoder_hidden_states = output[0]

            return ablated_encoder_hidden_states, ablated_hidden_states

        elif isinstance(module, FluxSingleTransformerBlock):
            return kwinput["hidden_states"] + self.cache["hidden_states"]

            
class PromptCachePreForwardHook:

    def __init__(self):
        self.cache = {}

    # Define a hook function
    @torch.no_grad()
    def get_hidden_states(self, *args):

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):
            self.cache["encoder_hidden_states"] = kwinput["encoder_hidden_states"]
        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["hidden_states"] = kwinput["hidden_states"]

        
    @torch.no_grad()
    def set_hidden_states(self, *args):

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):

            kwinput["encoder_hidden_states"] =  self.cache["encoder_hidden_states"]

        elif isinstance(module, FluxSingleTransformerBlock):
            kwinput["hidden_states"][:, :4608 - 4096, :] = self.cache["hidden_states"][:, :4608 - 4096, :]


class AttentionAblationCacheHook:

    def __init__(self):
        self.cache = {}

    # Define a hook function
    @torch.no_grad()
    def cache_text_stream(self, *args):
        """ 
            To be used as a pre forward hook on prompt used for ablation.
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):
            self.cache["injected_encoder_hidden_states"] = kwinput["encoder_hidden_states"]
        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["injected_hidden_states"] = kwinput["hidden_states"]

        
    @torch.no_grad()
    def cache_and_inject_pre_forward(self, *args):
        """ 
            To be used as a pre forward hook on the main prompt.
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):
            # Cache the original text stream to restore after the forward pass
            self.cache["encoder_hidden_states"] = kwinput["encoder_hidden_states"]
            # inject the external text stream 
            kwinput["encoder_hidden_states"] =  self.cache["injected_encoder_hidden_states"]

        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["hidden_states"] = kwinput["hidden_states"].clone()
            kwinput["hidden_states"][:, :4608 - 4096, :] = self.cache["injected_hidden_states"][:, :4608 - 4096, :]
        
        
    @torch.no_grad()
    def set_ablated_attention(self, *args, weight=1.0):
        """ 
            To be used as a forward hook on main prompt.
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            # Compute og_text_stream + g(image_stream, injected_text_stream, c)
            hidden_states = output[1]

            og_text_stream = self.cache["encoder_hidden_states"]
            injected_text_stream = self.cache["injected_encoder_hidden_states"]

            encoder_hidden_states = weight * (output[0] - injected_text_stream) + og_text_stream

            if encoder_hidden_states.dtype == torch.float16:
                encoder_hidden_states[torch.isposinf(encoder_hidden_states)] = 65504
                encoder_hidden_states[torch.isneginf(encoder_hidden_states)] = -65504

            assert torch.all(torch.isfinite(encoder_hidden_states)), "Output has NaN/Inf values"

            return encoder_hidden_states, hidden_states

        elif isinstance(module, FluxSingleTransformerBlock):

            weight_tensor = torch.full_like(output, weight)
            weight_tensor[:, 512:, :] = 1.0

            new_output = weight_tensor * (output - kwinput["hidden_states"]) + self.cache["hidden_states"]

            # empirically found inf values with float16 (in the code clipping is done on -65504, 65504)
            # but the clipping does not treat inf values
            if new_output.dtype == torch.float16:
                new_output[torch.isposinf(new_output)] = 65504
                new_output[torch.isneginf(new_output)] = -65504

            assert torch.all(torch.isfinite(new_output)), "Output has NaN/Inf values"

            return new_output
