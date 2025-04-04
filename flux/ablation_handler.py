from typing import Literal
import torch
from diffusers.models.transformers.transformer_2d import Transformer2DModel
from diffusers.models.transformers.transformer_flux import FluxTransformerBlock, FluxSingleTransformerBlock
from diffusers.models.attention import BasicTransformerBlock, FeedForward 


class TransformerActivationCache:

    def __init__(self):
        self.cache = {

            "transformer_blocks": {
                "image_residual": [],
                "text_residual": [],
                "image_activation": [],
                "text_activation": [],
            },

            "single_transformer_blocks": {
                "image_text_residual": [],
                "image_text_activation": []
            }
        }
  

    @staticmethod
    def _safe_clip(x: torch.Tensor):
        if x.dtype == torch.float16:
            x[torch.isposinf(x)] = 65504
            x[torch.isneginf(x)] = -65504
        return x

    @torch.no_grad()
    def cache_residual_and_activation(self, *args):
        """ 
            To be used as a forward hook on main prompt, it caches both residual-stream and 
            activation 
        """

        assert len(args) == 4
        module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):
            # Cache f(z, p, c) 
            encoder_hidden_states = output[0]            
            hidden_states = output[1]

            self.cache["transformer_blocks"]["image_activation"].append(hidden_states - kwinput["hidden_states"])
            self.cache["transformer_blocks"]["text_activation"].append(encoder_hidden_states - kwinput["encoder_hidden_states"])
            self.cache["transformer_blocks"]["image_residual"].append(kwinput["hidden_states"])
            self.cache["transformer_blocks"]["text_residual"].append(kwinput["encoder_hidden_states"])

        elif isinstance(module, FluxSingleTransformerBlock):
            self.cache["single_transformer_blocks"]["text_image_activation"].append(output - kwinput["hidden_states"])
            self.cache["single_transformer_blocks"]["text_image_residual"].append(kwinput["hidden_states"])

    
    @torch.no_grad()
    def replace_attention_activation(self, *args, full_output=False):
        """ 
            x + f(X) replaced with x + cache[f] if cache["full_output"] is False
            x + f(x) replaed with cache[f] if cache["full_output"] is True
        """

        assert len(args) == 4
        module, input, kwinput, output = args

        if isinstance(module, FluxTransformerBlock):

            if full_output:
                image_stream_output = self.cache["transformer_blocks"]["image_residual"] + self.cache["transformer_blocks"]["image_activation"]
                text_stream_output = self.cache["transformer_blocks"]["text_stream"] +  self.cache["transformer_blocks"]["text_activation"]
            else:
                image_stream_output = kwinput["hidden_states"] + self.cache["transformer_blocks"]["image_activation"]       
                text_stream_output = kwinput["encoder_hidden_states"] + self.cache["transformer_blocks"]["text_activation"]

            return text_stream_output, image_stream_output

        elif isinstance(module, FluxSingleTransformerBlock):

            if self.cache["is_full_output"]:
                output = self.cache["single_transformer_blocks"]["text_image_stream"]
            else:
                output = kwinput["hidden_states"] + self.cache["single_transformer_blocks"]["text_image_stream"]

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

