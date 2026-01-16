import logging
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


def overlap_with_reference(
    network: pd.DataFrame,
    reference_genes: Iterable[str],
    regulator_col: str = "regulator",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Filter inferred edges to those overlapping a reference gene set.

    Args:
        network: GRN edges DataFrame
        reference_genes: Iterable of genes for validation
        regulator_col: Column name for regulators
        target_col: Column name for targets

    Returns:
        Filtered DataFrame with overlapping edges
    """
    ref = set(reference_genes)
    filtered = network[network[regulator_col].isin(ref) | network[target_col].isin(ref)]
    logger.info("Validation overlap: %s edges", len(filtered))
    return filtered
