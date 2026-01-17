import argparse
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
from scipy.sparse import issparse

logger = logging.getLogger(__name__)


def _rank_genes_by_variance(adata: anndata.AnnData, genes: List[str]) -> List[str]:
    """Rank genes by variance in the dataset."""
    if not genes:
        return []

    subset = adata[:, genes]
    matrix = subset.X
    if issparse(matrix):
        mean = matrix.mean(axis=0).A1
        mean_sq = matrix.multiply(matrix).mean(axis=0).A1
        variances = mean_sq - mean**2
    else:
        variances = np.var(matrix, axis=0)
        if hasattr(variances, "A1"):
            variances = variances.A1

    ranked = [gene for _, gene in sorted(zip(variances, genes), reverse=True)]
    return ranked


def _load_grn(grn_path: Path) -> pd.DataFrame:
    """Load GRN CSV and standardize columns."""
    grn = pd.read_csv(grn_path)
    if "TF" not in grn.columns and "regulator" in grn.columns:
        grn = grn.rename(columns={"regulator": "TF"})
    if "TF" not in grn.columns or "target" not in grn.columns:
        raise ValueError("GRN file must contain 'TF' and 'target' columns.")
    if "importance" not in grn.columns:
        raise ValueError("GRN file must contain 'importance' column.")
    return grn


def add_coexpression_edges(
    adata_path: Path,
    grn_path: Path,
    output_path: Path,
    threshold: float = 0.5,
    max_genes: int = 500,
    bidirectional: bool = True,
) -> pd.DataFrame:
    """
    Add coexpression edges to a GRN based on gene-gene correlation.

    Args:
        adata_path: Path to processed .h5ad file
        grn_path: Path to GRN CSV file
        output_path: Path to write combined GRN CSV
        threshold: Absolute correlation threshold
        max_genes: Maximum genes to include in correlation calculation
        bidirectional: Add both directions for coexpression edges

    Returns:
        Combined GRN DataFrame
    """
    adata = anndata.read_h5ad(adata_path)
    grn = _load_grn(grn_path)

    genes = sorted(set(grn["TF"]).union(set(grn["target"])))
    genes = [gene for gene in genes if gene in adata.var_names]
    if len(genes) < 2:
        raise ValueError("Need at least 2 genes in dataset to compute coexpression.")

    if len(genes) > max_genes:
        ranked = _rank_genes_by_variance(adata, genes)
        genes = ranked[:max_genes]
        logger.info("Subsetted to top %s genes by variance for coexpression.", len(genes))

    subset = adata[:, genes]
    matrix = subset.X
    if issparse(matrix):
        matrix = matrix.toarray()
    corr = np.corrcoef(matrix.T)

    upper = np.triu(np.ones_like(corr, dtype=bool), k=1)
    mask = upper & (np.abs(corr) >= threshold)
    idx_i, idx_j = np.where(mask)

    edges = []
    for i, j in zip(idx_i, idx_j):
        gene_i = genes[i]
        gene_j = genes[j]
        weight = float(abs(corr[i, j]))
        edges.append({"TF": gene_i, "target": gene_j, "importance": weight, "edge_type": "coexpression"})
        if bidirectional:
            edges.append({"TF": gene_j, "target": gene_i, "importance": weight, "edge_type": "coexpression"})

    grn = grn.copy()
    if "edge_type" not in grn.columns:
        grn["edge_type"] = "regulatory"
    combined = pd.concat([grn, pd.DataFrame(edges)], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)
    logger.info("Wrote combined GRN with %s coexpression edges to %s", len(edges), output_path)

    return combined


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Augment GRN with coexpression edges")
    parser.add_argument("--adata", required=True, help="Path to processed .h5ad file")
    parser.add_argument("--grn", required=True, help="Path to GRN CSV file")
    parser.add_argument("--output", required=True, help="Output path for combined GRN CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="Absolute correlation threshold")
    parser.add_argument("--max_genes", type=int, default=500, help="Max genes for correlation calculation")
    parser.add_argument(
        "--no_bidirectional",
        action="store_false",
        dest="bidirectional",
        help="Only add one direction for coexpression edges",
    )
    parser.set_defaults(bidirectional=True)

    args = parser.parse_args()

    add_coexpression_edges(
        adata_path=Path(args.adata),
        grn_path=Path(args.grn),
        output_path=Path(args.output),
        threshold=args.threshold,
        max_genes=args.max_genes,
        bidirectional=args.bidirectional,
    )
