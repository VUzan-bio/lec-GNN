import argparse
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc

from src.grn.inference import GRNInferenceEngine

logger = logging.getLogger(__name__)


def _load_gene_list(path: Optional[str]) -> List[str]:
    if not path:
        return []
    df = pd.read_csv(path)
    for col in ("symbol", "gene", "gene_symbol"):
        if col in df.columns:
            return df[col].dropna().astype(str).tolist()
    return df.iloc[:, 0].dropna().astype(str).tolist()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description="Run GRN inference on a smaller sample with stats.")
    parser.add_argument("--adata", required=True, help="Path to input .h5ad file")
    parser.add_argument("--output", required=True, help="Output path for GRN (without extension)")
    parser.add_argument("--tf_list", default=None, help="Path to TF list CSV (symbol/gene column)")
    parser.add_argument(
        "--gene_universe",
        default=None,
        help="Path to gene universe CSV (symbol/gene column) to expand candidate genes.",
    )
    parser.add_argument("--pathway_genes", default=None, help="Optional pathway genes CSV")
    parser.add_argument("--sample_cells", type=int, default=10000, help="Number of cells to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_hvg", action="store_true", help="Include HVGs to reach min genes")
    parser.add_argument("--min_genes", type=int, default=600, help="Minimum genes for GRN")
    parser.add_argument("--max_genes", type=int, default=800, help="Maximum genes for GRN")
    parser.add_argument("--filter_method", default="percentile", choices=["percentile", "pvalue"])
    parser.add_argument("--filter_threshold", type=float, default=0.85)
    parser.add_argument("--n_workers", type=int, default=4, help="Workers for GRNBoost2")
    parser.add_argument(
        "--backend",
        choices=["auto", "grnboost2", "sklearn"],
        default="sklearn",
        help="GRN inference backend",
    )
    parser.add_argument(
        "--no_fallback",
        action="store_true",
        help="Disable fallback to sklearn GBM when GRNBoost2 fails",
    )
    parser.add_argument(
        "--subset_out",
        default=None,
        help="Optional path to save the sampled AnnData subset",
    )
    args = parser.parse_args()

    adata_full = sc.read_h5ad(args.adata)
    total_cells = adata_full.n_obs
    total_genes = adata_full.n_vars

    n_cells = args.sample_cells
    if n_cells and n_cells < total_cells:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(total_cells, size=n_cells, replace=False)
        adata = adata_full[idx].copy()
    else:
        adata = adata_full.copy()
        n_cells = total_cells

    if args.subset_out:
        subset_path = Path(args.subset_out)
        subset_path.parent.mkdir(parents=True, exist_ok=True)
        adata.write_h5ad(subset_path)
        logger.info("Saved sampled AnnData to %s", subset_path)

    logger.info(
        "Cells used: %s/%s (%.1f%%)",
        adata.n_obs,
        total_cells,
        100.0 * adata.n_obs / total_cells,
    )
    logger.info("Genes in dataset: %s", total_genes)

    engine = GRNInferenceEngine(
        adata,
        tf_list_path=args.tf_list,
        pathway_genes_path=args.pathway_genes,
        gene_universe_path=args.gene_universe,
    )

    tf_total = len(_load_gene_list(args.tf_list)) if args.tf_list else len(engine.tf_list)
    tf_in_data = len(engine.tf_list)

    gene_universe = _load_gene_list(args.gene_universe)
    if gene_universe:
        universe_in_data = len(set(gene_universe) & set(adata.var_names))
        logger.info(
            "Gene universe overlap: %s/%s (%.1f%%) present in data",
            universe_in_data,
            len(gene_universe),
            100.0 * universe_in_data / len(gene_universe),
        )

    logger.info(
        "TF overlap: %s/%s (%.1f%%) present in data",
        tf_in_data,
        tf_total,
        100.0 * tf_in_data / max(tf_total, 1),
    )

    expr_matrix = engine.prepare_expression_matrix(
        use_highly_variable=args.use_hvg,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )

    logger.info(
        "Genes selected for GRN: %s/%s (%.1f%%)",
        expr_matrix.shape[1],
        total_genes,
        100.0 * expr_matrix.shape[1] / total_genes,
    )

    grn = engine.infer_grn(
        expr_matrix,
        n_workers=args.n_workers,
        seed=args.seed,
        backend=args.backend,
        allow_fallback=not args.no_fallback,
    )

    grn_filtered = engine.filter_high_confidence_edges(
        grn,
        method=args.filter_method,
        threshold=args.filter_threshold,
    )

    engine.save_network(grn_filtered, args.output)
    logger.info("GRN inference complete: %s edges", len(grn_filtered))


if __name__ == "__main__":
    main()
