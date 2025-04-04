from typing import Callable, Dict, List, Literal
import torch
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider

from typing import Dict, List
import torch
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import interact, IntSlider, FloatText, VBox, HBox
import ipywidgets as widgets

@torch.no_grad
def _safe_clip(x: torch.Tensor):
    if x.dtype == torch.float16:
        x[torch.isposinf(x)] = 65504
        x[torch.isneginf(x)] = -65504
    return x


def mean_tokens(x):
    result = torch.mean(_safe_clip(x).to(torch.float32).squeeze(0), dim=1).cpu()
    assert torch.all(torch.isfinite(result))
    return result

def norm_tokens(x):
    result = torch.norm(_safe_clip(x).to(torch.float32).squeeze(0), dim=1).cpu()

    assert torch.all(torch.isfinite(result))
    return result


def plot_text_activation(cache):
    # Generate example data (replace this with your actual data)
    norms_tokens = np.array([norm_tokens(cache[i]["output_text"] - cache[i]["encoder_hidden_states"]).cpu() for i in range(len(cache))]).T

    # Plotting
    plt.figure(figsize=(8, 8))  # Adjust figure size to keep the plot square
    plt.imshow(norms_tokens, aspect='auto', cmap='viridis', interpolation='none', vmin=0, vmax=200)
    plt.colorbar(label='Vector Norm')
    plt.xlabel('Layer')
    plt.ylabel('Tokens')
    plt.title('Empty prompt norm (512 x 19)')
    plt.show()



def interactive_image_activation(cache: Dict[str, List[torch.Tensor]], use_mean=False):

    # process image
    image_residuals = cache["image_residual"] + list(map(lambda x: x[:, 512:, :], cache["text_image_residual"]))
    image_activations = cache["image_activation"] +  list(map(lambda x: x[:, 512:, :], cache["text_image_activation"]))
    image_residuals.append(image_residuals[-1] + image_activations[-1])
    image_activations.append(image_activations[-1] - image_activations[-1])

    if use_mean:
        fn = mean_tokens
    else:
        fn = norm_tokens

    image_residuals = list(map(lambda x: fn(x.cpu()), image_residuals))
    image_activations = list(map(lambda x: fn(x.cpu()), image_activations))
    labels = [f"Stream {idx}" for idx in range(len(image_residuals))]

    # process text
    text_residuals = cache["text_residual"] + list(map(lambda x: x[:, :512, :], cache["text_image_residual"]))
    text_activations = cache["text_activation"] +  list(map(lambda x: x[:, :512, :], cache["text_image_activation"]))
    text_residuals.append(text_residuals[-1] + text_activations[-1])
    text_activations.append(text_activations[-1] - text_activations[-1])

    text_residuals = list(map(lambda x: fn(x.cpu()), text_residuals))
    text_activations = list(map(lambda x: fn(x.cpu()), text_activations))

    # Widgets
    layer_slider = IntSlider(min=0, max=len(image_residuals) - 1, step=1, value=0, description="Layer")
    percentile_input = FloatText(value=0.999, step=0.001, description='Percentile')

    # Output plot function
    def update_plot(column_idx, percentile):
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        # Image data (top row)
        acc_image = image_residuals[column_idx].numpy().reshape(64, 64)
        raw_image = image_activations[column_idx].numpy().reshape(64, 64)

        vmax_acc = np.quantile(acc_image, min(percentile, 1.0))
        im0 = axs[0, 0].imshow(acc_image, cmap='viridis', interpolation='none', vmax=vmax_acc)
        axs[0, 0].set_title(f'Image Accum. Difference - {labels[column_idx]}')
        axs[0, 0].axis('off')
        fig.colorbar(im0, ax=axs[0, 0], fraction=0.046, pad=0.04)

        vmax_raw = np.quantile(raw_image, 0.999)
        im1 = axs[0, 1].imshow(raw_image, cmap='plasma', interpolation='none', vmax=vmax_raw)
        axs[0, 1].set_title(f'Image Activation - {labels[column_idx]}')
        axs[0, 1].axis('off')
        fig.colorbar(im1, ax=axs[0, 1], fraction=0.046, pad=0.04)

        # Text data (bottom row)
        acc_text = text_residuals[column_idx].numpy().reshape(8, 64)
        raw_text = text_activations[column_idx].numpy().reshape(8, 64)

        vmax_text_acc = np.quantile(acc_text, min(percentile, 1.0))
        im2 = axs[1, 0].imshow(acc_text, cmap='viridis', interpolation='none', vmax=vmax_text_acc)
        axs[1, 0].set_title(f'Text Accum. Difference - {labels[column_idx]}')
        axs[1, 0].axis('off')
        fig.colorbar(im2, ax=axs[1, 0], fraction=0.046, pad=0.04)

        vmax_text_raw = np.quantile(raw_text, 0.999)
        im3 = axs[1, 1].imshow(raw_text, cmap='plasma', interpolation='none', vmax=vmax_text_raw)
        axs[1, 1].set_title(f'Text Activation - {labels[column_idx]}')
        axs[1, 1].axis('off')
        fig.colorbar(im3, ax=axs[1, 1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    # Link widgets to output
    out = widgets.interactive_output(update_plot, {
        'column_idx': layer_slider,
        'percentile': percentile_input
    })

    # Display everything together
    controls = HBox([layer_slider, percentile_input])
    return VBox([controls, out])



def plot_activation_by_layer(cache: Dict[str, List[torch.Tensor]], stream: Literal["image", "text", "both"] = "image"):
    # Recap: replace text-stream at some layer 

    if stream == "both":
        stream = ["image", "text"]
    else: 
        stream = [stream]


    fig, axes = plt.subplots(nrows=1, ncols=len(stream), figsize=(8 * len(stream), 6))

    for idx, mode in enumerate(stream):
        # Compute norms
        norm_residual = [norm_tokens(cache[f"{mode}_residual"][i]).mean().item() for i in range(19)]
        norm_activation = [norm_tokens(cache[f"{mode}_activation"][i]).mean().item() for i in range(19)]
        norm_output = [norm_tokens(cache[f"{mode}_residual"][i] + cache[f"{mode}_activation"][i]).mean().item() for i in range(19)]
        time_steps = list(range(19))

        # Plotting
        ax = axes[idx] if len(stream) == 2 else axes

        ax.plot(time_steps, norm_residual, label='||res||', marker='o')
        ax.plot(time_steps, norm_activation, label='||act||', marker='s')
        ax.plot(time_steps, norm_output, label='||out||', marker='^')

        ax.set_xlabel('Layer')
        ax.set_ylabel('Tensor Norm')
        ax.set_title(f'Norms of {mode} residual-stream, activaton, output over layers ')
        ax.grid(True)
    
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_activation_by_layer_og_ablated(cache1: Dict[str, List[torch.Tensor]], 
                             cache2: Dict[str, List[torch.Tensor]]):
    
    # normalize
    def fix_outliers(x: torch.Tensor):
        q999 = torch.quantile(x, 0.999)
        x[x > q999] = q999
        return x

    # Recap: replace text-stream at some layer 

    stream = ["image", "text", "text_image"]

    fig, axes = plt.subplots(nrows=1, ncols=len(stream), figsize=(8 * len(stream), 6))

    for idx, mode in enumerate(stream):
        # Compute norms
        for cache, line, label in (cache1, "-", "OG"), (cache2, "--", "Abl."):

            num_layers = len(cache[f"{mode}_residual"])
            norm_residual = [fix_outliers(norm_tokens(cache[f"{mode}_residual"][i])).mean().item() for i in range(num_layers)]
            norm_activation = [fix_outliers(norm_tokens(cache[f"{mode}_activation"][i])).mean().item() for i in range(num_layers)]
            norm_output = [fix_outliers(norm_tokens(cache[f"{mode}_residual"][i] + cache[f"{mode}_activation"][i])).mean().item() for i in range(num_layers)]
            time_steps = list(range(num_layers))

            # Plotting
            ax = axes[idx] if len(stream) > 1 else axes

            ax.plot(time_steps, norm_residual, label=f'{label} ||res||', marker='o', linestyle=line, color="blue")
            ax.plot(time_steps, norm_activation, label=f'{label} ||act||', marker='s', linestyle=line, color="orange")
            ax.plot(time_steps, norm_output, label=f'{label} ||out||', marker='^', linestyle=line, color='green')

        ax.set_xlabel('Time Step')
        ax.set_ylabel('Tensor Norm')
        ax.set_title(f'Norms of {mode} residual-stream, activaton, output over layers ')
        ax.grid(True)
    
    plt.legend()
    plt.tight_layout()
    plt.show()



def interactive_image_activation_2(op1: Callable, op2: Callable):
    
    cache1 = []
    cache2 = []

    for layer in range(6):
        cache1.append(op1(layer))
        cache2.append(op2(layer))

    plot_cache = lambda i: plot_activation_by_layer_og_ablated(cache1[i], cache2[i])

    # Interactive slider to select columns
    interact(plot_cache, i=IntSlider(min=0, max=6, step=1, value=0))
