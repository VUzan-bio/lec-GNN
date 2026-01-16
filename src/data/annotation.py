import logging
from typing import Optional

import anndata
import pandas as pd

logger = logging.getLogger(__name__)


def map_gene_symbols(
    adata: anndata.AnnData,
    mapping: pd.DataFrame,
    source_col: str = "symbol",
    target_col: str = "ensembl",
    new_var_key: Optional[str] = "gene_id",
) -> anndata.AnnData:
    """
    Map gene symbols to another identifier and store in adata.var.

    Args:
        adata: AnnData with gene symbols in adata.var_names
        mapping: DataFrame with source and target columns
        source_col: Column name containing current symbols
        target_col: Column name containing mapped IDs
        new_var_key: Column name to store mapped IDs in adata.var

    Returns:
        AnnData with mapped identifiers in .var
    """
    if new_var_key is None:
        new_var_key = target_col

    if source_col not in mapping.columns or target_col not in mapping.columns:
        raise ValueError("Mapping DataFrame must include source and target columns")

    lookup = dict(zip(mapping[source_col].astype(str), mapping[target_col].astype(str)))
    adata.var[new_var_key] = [lookup.get(gene, "") for gene in adata.var_names]

    logger.info("Mapped %s gene symbols to %s", adata.n_vars, new_var_key)
    return adata
