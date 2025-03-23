from typing import Literal
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
            kwinput["hidden_states"][:, :512, :] = self.cache["injected_hidden_states"][:, :512, :]
        
        
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
            
            self.cache["output_text"] = encoder_hidden_states

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

            self.cache["output_text_image"] = new_output

            assert torch.all(torch.isfinite(new_output)), "Output has NaN/Inf values"

            return new_output

    @torch.no_grad()
    def restore_text_stream(self, *args):
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
            # previous input for text stream
            return self.cache["encoder_hidden_states"], hidden_states

        elif isinstance(module, FluxSingleTransformerBlock):
            output[:, :512, :] = self.cache["hidden_states"][:, :512, :]
            return output

class TransformerActivationCache:

    def __init__(self):
        self.cache = {}
        self.cache_lists = {
            "image_residual": [],
            "text_residual": [],
            "image_activation": [],
            "text_activation": [],
            "image_text_residual": [],
            "image_text_activation": []
        }

    @staticmethod
    def _safe_clip(x: torch.Tensor):
        if x.dtype == torch.float16:
            x[torch.isposinf(x)] = 65504
            x[torch.isneginf(x)] = -65504
        return x

    @torch.no_grad()
    def cache_attention_activation(self, *args, full_output=False):
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
            # Cache f(z, p, c) 
            encoder_hidden_states = output[0]            
            hidden_states = output[1]

            if full_output:
                self.cache["image_stream"] = hidden_states 
                self.cache["text_stream"] = encoder_hidden_states 
            else:
                self.cache["image_stream"] = hidden_states - kwinput["hidden_states"]
                self.cache["text_stream"] = encoder_hidden_states - kwinput["encoder_hidden_states"]

            self.cache["is_full_output"] = full_output

        elif isinstance(module, FluxSingleTransformerBlock):
            if full_output: 
                self.cache["text_image_stream"] = output
            else:
                self.cache["text_image_stream"] = output - kwinput["hidden_states"]
            self.cache["is_full_output"] = full_output


    @torch.no_grad()
    def cache_residual_and_activation(self, *args):
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
            # Cache f(z, p, c) 
            encoder_hidden_states = output[0]            
            hidden_states = output[1]

            self.cache_lists["image_activation"].append(hidden_states - kwinput["hidden_states"])
            self.cache_lists["text_activation"].append(encoder_hidden_states - kwinput["encoder_hidden_states"])
            self.cache_lists["image_residual"].append(kwinput["hidden_states"])
            self.cache_lists["text_residual"].append(kwinput["encoder_hidden_states"])


        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache_lists["text_image_activation"].append(output - kwinput["hidden_states"])
            self.cache_lists["text_image_residual"].append(kwinput["hidden_states"])

    

    @torch.no_grad()
    def replace_attention_activation(self, *args):
        """ 
            x + f(X) replaced with x + cache[f] if cache["full_output"] is False
            x + f(x) replaed with cache[f] if cache["full_output"] is True
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):

            if self.cache["is_full_output"]:
                image_stream_output = self.cache["image_stream"]
                text_stream_output = self.cache["text_stream"]
            else:
                image_stream_output = kwinput["hidden_states"] + self.cache["image_stream"]       
                text_stream_output = kwinput["encoder_hidden_states"] + self.cache["text_stream"]

            return text_stream_output, image_stream_output

        elif isinstance(module, FluxSingleTransformerBlock):

            if self.cache["is_full_output"]:
                output = self.cache["text_image_stream"]
            else:
                output = kwinput["hidden_states"] + self.cache["text_image_stream"]

            return output
        
    @torch.no_grad()
    def replace_stream_input(self, *args, use_random=False, use_tensor: torch.Tensor = None, stream: Literal["text", "image"] = "text"):
        """ 
            x replaced with cache[x] 
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):

            # TODO: extend tis also for FluxSingleTransformerBlock
            if stream == "text":
                cache_key = "text_stream"
                input_key = "encoder_hidden_states"
            elif stream == "image":
                cache_key = "image_stream"
                input_key = "hidden_states"
            else:
                raise AssertionError("invalid stream")

            if use_random:
                self.cache[cache_key] = torch.randn_like(kwinput[input_key],
                                                             dtype=kwinput[input_key].dtype,
                                                             device=kwinput[input_key].device)
            elif use_tensor is not None:
                self.cache[cache_key] = use_tensor
            else:
                assert self.cache["is_full_output"], "A layer input must be replaced with the full output."

                # case of single_transformer_block cached
                if cache_key not in self.cache and "text_image_stream" in self.cache: 
                    if cache_key == "text_stream":
                        self.cache[cache_key] = self.cache["text_image_stream"][:, :512, :] # text stream position
                    else:
                        self.cache[cache_key] = self.cache["text_image_stream"][:, 512:, :] # image stream position

            kwinput[input_key] = self.cache[cache_key]

        elif isinstance(module, FluxSingleTransformerBlock):

            if use_random:
                self.cache["text_image_stream"] = torch.randn_like(kwinput["hidden_states"],
                                                             dtype=kwinput["hidden_states"].dtype,
                                                             device=kwinput["hidden_states"].device)
            elif use_tensor:
                raise NotImplementedError
            else:
                assert self.cache["is_full_output"], "A layer input must be replaced with the full output."


            kwinput["hidden_states"] = self.cache["text_image_stream"]


    @torch.no_grad()
    def disable_text_residual(self, *args):
        """ 
            x + f(X) replaced with f(X)
        """

        TransformerActivationCache.reweight_text_stream(*args, residual_w=0, activation_w=1)
    
    @torch.no_grad()
    def disable_text_attention(self, *args):
        """ 
            x + f(X) replaced with x
        """
        TransformerActivationCache.reweight_text_stream(*args, residual_w=1, activation_w=0)
    
        

    @torch.no_grad()
    def reweight_text_stream(self, *args, residual_w: float = 1.0, activation_w: float = 1.0):
        """ 
            x + f(X) replaced with w_res * x + w_act * f(x)
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            residual = kwinput["encoder_hidden_states"]
            activation = output[0] - residual

            self.cache["text_residual"] = residual
            self.cache["text_activation"] = activation

            return TransformerActivationCache._safe_clip(residual * residual_w + activation * activation_w), TransformerActivationCache._safe_clip(output[1])

        elif isinstance(module, FluxSingleTransformerBlock):
            activation_w = torch.full_like(output, activation_w)
            activation_w[:, 512:, :] = 1.0

            residual_w = torch.full_like(output, residual_w)
            residual_w[:, 512:, :] = 1.0

            residual = kwinput["hidden_states"]
            activation = output - kwinput["hidden_states"]

            self.cache["residual"] = residual
            self.cache["activation"] = activation

            return TransformerActivationCache._safe_clip(residual * residual_w + activation * activation_w)
        
    @torch.no_grad()
    def reweight_image_stream(self, *args, residual_w: float = 1.0, activation_w: float = 1.0):
        """
            x + f(X) replaced with w_res * x + w_act * f(x)
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            residual = kwinput["hidden_states"]
            activation = output[1] - residual

            self.cache["image_residual"] = residual
            self.cache["image_activation"] = activation

            return TransformerActivationCache._safe_clip(output[0]), TransformerActivationCache._safe_clip(residual * residual_w + activation * activation_w)

        elif isinstance(module, FluxSingleTransformerBlock):
            activation_w = torch.full_like(output, activation_w)
            activation_w[:, :512, :] = 1.0

            residual_w = torch.full_like(output, residual_w)
            residual_w[:, :512, :] = 1.0

            residual = kwinput["hidden_states"]
            activation = output - kwinput["hidden_states"]

            self.cache["residual"] = residual
            self.cache["activation"] = activation

            return TransformerActivationCache._safe_clip(residual * residual_w + activation * activation_w)
        
    
    @torch.no_grad()
    def clamp_output(self, *args):

        # Case 1: no kwards are passed to the module
        if len(args) == 3:
            module, input, output = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 4:
            module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):

            return TransformerActivationCache._safe_clip(output[0]), TransformerActivationCache._safe_clip(output[1])

        elif isinstance(module, FluxSingleTransformerBlock):

            return TransformerActivationCache._safe_clip(output)
        

    @torch.no_grad()
    def add_text_stream_input(self, *args, use_tensor: torch.Tensor = None):
        """ 
            x replaced with cache[x] 
        """

        # Case 1: no kwards are passed to the module
        if len(args) == 2:
            module, input = args
        # Case 2: when kwargs are passed to the model as input
        elif len(args) == 3:
            module, input, kwinput = args

        if isinstance(module, FluxTransformerBlock):

            if use_tensor is not None:
                self.cache["text_stream"] = use_tensor
            else:
                assert self.cache["is_full_output"], "A layer input must be replaced with the full output."

                if "text_stream" not in self.cache and "text_image_stream" in self.cache: # case of single_transformer_block cached
                    self.cache["text_stream"] = self.cache["text_image_stream"][:, :512, :]

            kwinput["encoder_hidden_states"] = self.cache["text_stream"] + kwinput["encoder_hidden_states"]

        elif isinstance(module, FluxSingleTransformerBlock):


            raise NotImplementedError

