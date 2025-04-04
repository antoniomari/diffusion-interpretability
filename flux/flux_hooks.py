import torch
from diffusers.models.transformers import FluxTransformer2DModel
from diffusers.models.modeling_outputs import Transformer2DModelOutput


@torch.no_grad()
def insert_extra_registers(*args, num_registers=64):

    # Case 1: no kwards are passed to the module
    if len(args) == 2:
        module, input = args
    # Case 2: when kwargs are passed to the model as input
    elif len(args) == 3:
        module, input, kwinput = args

    assert isinstance(module, FluxTransformer2DModel)

    shape_noise = list(kwinput["hidden_states"].shape)
    shape_noise[1] = num_registers
    registers = torch.randn(shape_noise, 
                            device=kwinput["hidden_states"].device, 
                            dtype=kwinput["hidden_states"].dtype)

    kwinput["hidden_states"] = torch.concat([kwinput["hidden_states"], registers], dim=1)
    
@torch.no_grad()
def discard_extra_registers(*args, num_registers=64):

    # Case 1: no kwards are passed to the module
    if len(args) == 3:
        module, input, output = args
    # Case 2: when kwargs are passed to the model as input
    elif len(args) == 4:
        module, input, kwinput, output = args

    assert isinstance(module, FluxTransformer2DModel)

    if isinstance(output, tuple):
        
        return (output[0][:, : -num_registers],)
    else:
        assert isinstance(output, Transformer2DModelOutput)
        
        output.sample = output.sample[:, : -num_registers]

        return output
