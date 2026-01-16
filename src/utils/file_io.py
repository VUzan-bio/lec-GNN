import logging
from pathlib import Path
from typing import Any, Dict

import anndata
import pandas as pd

logger = logging.getLogger(__name__)


def save_anndata(adata: anndata.AnnData, path: Path) -> None:
    """
    Save AnnData to disk.

    Args:
        adata: AnnData object
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    adata.write_h5ad(path)
    logger.info("Saved AnnData to %s", path)


def load_anndata(path: Path) -> anndata.AnnData:
    """
    Load AnnData from disk.

    Args:
        path: Input path

    Returns:
        AnnData object
    """
    if not path.exists():
        raise FileNotFoundError(f"AnnData file not found: {path}")
    return anndata.read_h5ad(path)


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    """
    Save DataFrame to CSV.

    Args:
        df: DataFrame to save
        path: Output path
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved DataFrame to %s", path)


def load_yaml(path: Path) -> Dict[str, Any]:
    """
    Load YAML file into a dictionary.

    Args:
        path: Path to YAML file

    Returns:
        Parsed dictionary
    """
    import yaml

    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
