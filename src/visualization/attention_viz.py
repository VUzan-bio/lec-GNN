import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_attention_weights(weights: np.ndarray, save_path: Optional[str] = None) -> None:
    """
    Plot attention weights as a heatmap.

    Args:
        weights: Attention matrix
        save_path: Optional path to save the figure
    """
    plt.imshow(weights, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention")
    plt.title("GAT Attention Weights")
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved attention plot to %s", save_path)
    else:
        plt.show()
