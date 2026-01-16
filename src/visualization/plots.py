import logging
from typing import Optional

import matplotlib.pyplot as plt
import scanpy as sc

logger = logging.getLogger(__name__)


def plot_umap(adata: sc.AnnData, color: Optional[str] = None, save_path: Optional[str] = None) -> None:
    """
    Plot UMAP embedding for an AnnData object.

    Args:
        adata: AnnData with UMAP computed
        color: Optional obs column to color by
        save_path: Optional path to save the figure
    """
    sc.pl.umap(adata, color=color, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        logger.info("Saved UMAP to %s", save_path)
    else:
        plt.show()
