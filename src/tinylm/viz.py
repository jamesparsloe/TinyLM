import matplotlib.pyplot as plt
from torch import Tensor


# TODO make this stuff a bit more like https://github.com/openai/transformer-debugger and others etc
def plot_attn(attn_weights, path: str):
    n_layers, n_heads, T, T = attn_weights.size()

    fig, axes = plt.subplots(
        nrows=n_layers,
        ncols=n_heads,
        sharex=True,
        sharey=True,
        figsize=(3 * n_heads, 3 * n_layers),
    )

    layer_idx = 0

    for layer_idx in range(n_layers):
        for h in range(n_heads):
            axes[layer_idx, h].matshow(attn_weights[layer_idx, h].cpu().numpy())
            axes[layer_idx, h].set_xticks([])
            axes[layer_idx, h].set_yticks([])

    fig.tight_layout()
    plt.savefig(path)
