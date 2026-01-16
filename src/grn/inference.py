import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GRNInferenceEngine:
    """
    Gene regulatory network inference using GRNBoost2 with statistical validation.

    Follows methodology from:
    - Moerman et al. 2019, Bioinformatics (GRNBoost2)
    - Pratapa et al. 2020, Nat Methods (GRN benchmarking)
    """

    def __init__(
        self,
        adata: anndata.AnnData,
        tf_list_path: str = "data/external/mouse_tfs_animalTFDB.csv",
        pathway_genes_path: str = "data/external/go_pathways_trafficking.csv",
    ) -> None:
        """
        Initialize GRN inference engine.

        Args:
            adata: Preprocessed AnnData object (log-normalized)
            tf_list_path: Path to transcription factor database
            pathway_genes_path: Path to GO pathway gene sets
        """
        self.adata = adata
        self.tf_list = self._load_tf_list(tf_list_path)
        self.pathway_genes = self._load_pathway_genes(pathway_genes_path)

        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    def _load_tf_list(self, path: str) -> List[str]:
        """Load transcription factor gene symbols."""
        tf_df = pd.read_csv(path)
        if "symbol" in tf_df.columns:
            tfs = tf_df["symbol"].astype(str).unique().tolist()
        elif "gene" in tf_df.columns:
            tfs = tf_df["gene"].astype(str).unique().tolist()
        else:
            raise ValueError("TF file must contain a 'symbol' or 'gene' column")

        tfs_present = [tf for tf in tfs if tf in self.adata.var_names]

        logger.info("Loaded %s/%s TFs present in dataset", len(tfs_present), len(tfs))
        return tfs_present

    def _load_pathway_genes(self, path: str) -> List[str]:
        """Load trafficking/glycosylation pathway genes from GO."""
        pathway_df = pd.read_csv(path)
        if "gene" not in pathway_df.columns:
            raise ValueError("Pathway gene file must contain a 'gene' column")

        genes = pathway_df["gene"].astype(str).unique().tolist()
        genes_present = [gene for gene in genes if gene in self.adata.var_names]

        logger.info("Loaded %s pathway genes present in dataset", len(genes_present))
        return genes_present

    def _get_highly_variable_genes(self, adata: anndata.AnnData) -> List[str]:
        """Return highly variable genes if available."""
        if "highly_variable" in adata.var.columns:
            return adata.var_names[adata.var["highly_variable"]].tolist()
        logger.warning("highly_variable not found in adata.var; skipping HVG fallback")
        return []

    def _rank_genes_by_variance(self, adata: anndata.AnnData, genes: List[str]) -> List[str]:
        """Rank genes by variance in the subset."""
        if not genes:
            return []

        subset = adata[:, genes]
        matrix = subset.X
        if hasattr(matrix, "multiply"):
            mean = matrix.mean(axis=0).A1
            mean_sq = matrix.multiply(matrix).mean(axis=0).A1
            variances = mean_sq - mean**2
        else:
            variances = np.var(matrix, axis=0)
            if hasattr(variances, "A1"):
                variances = variances.A1

        ranked = [gene for _, gene in sorted(zip(variances, genes), reverse=True)]
        return ranked

    def prepare_expression_matrix(
        self,
        cell_subset: Optional[str] = None,
        use_highly_variable: bool = False,
        min_genes: int = 400,
        max_genes: int = 800,
    ) -> pd.DataFrame:
        """
        Extract gene expression matrix for GRN inference.

        Args:
            cell_subset: Cell type to subset (e.g., "floor_LEC"), None for all cells
            use_highly_variable: Use HVGs as fallback to reach min_genes
            min_genes: Minimum number of genes to include
            max_genes: Maximum number of genes to include

        Returns:
            DataFrame: cells x genes expression matrix
        """
        logger.info("Preparing expression matrix")

        if cell_subset:
            if "cell_type" not in self.adata.obs:
                raise ValueError("cell_type column not found in adata.obs")
            adata_subset = self.adata[self.adata.obs["cell_type"] == cell_subset].copy()
            logger.info("  Using %s %s cells", adata_subset.n_obs, cell_subset)
        else:
            adata_subset = self.adata.copy()
            logger.info("  Using all %s cells", adata_subset.n_obs)

        candidates = sorted(set(self.tf_list + self.pathway_genes))

        if use_highly_variable and len(candidates) < min_genes:
            hvgs = self._get_highly_variable_genes(adata_subset)
            for gene in hvgs:
                if gene not in candidates:
                    candidates.append(gene)
                if len(candidates) >= min_genes:
                    break

        genes_present = [gene for gene in candidates if gene in adata_subset.var_names]

        if len(genes_present) > max_genes:
            ranked = self._rank_genes_by_variance(adata_subset, genes_present)
            genes_present = ranked[:max_genes]

        logger.info("  Selected %s genes for GRN inference", len(genes_present))
        logger.info("    TFs: %s", len([g for g in genes_present if g in self.tf_list]))
        logger.info("    Pathway genes: %s", len([g for g in genes_present if g in self.pathway_genes]))

        expr_matrix = adata_subset[:, genes_present].to_df()
        return expr_matrix

    def infer_grn(self, expr_matrix: pd.DataFrame, n_workers: int = 4, seed: int = 42) -> pd.DataFrame:
        """
        Infer gene regulatory network using GRNBoost2.

        Args:
            expr_matrix: cells x genes DataFrame
            n_workers: Number of parallel workers
            seed: Random seed for reproducibility

        Returns:
            DataFrame with columns [TF, target, importance]
        """
        try:
            from arboreto.algo import grnboost2
        except ImportError as exc:
            raise ImportError("arboreto is required for GRNBoost2. Install arboreto.") from exc

        logger.info("Running GRNBoost2 on %s cells x %s genes", expr_matrix.shape[0], expr_matrix.shape[1])

        tfs_in_matrix = [tf for tf in self.tf_list if tf in expr_matrix.columns]
        logger.info("  Using %s TFs as regulators", len(tfs_in_matrix))

        network = grnboost2(
            expression_data=expr_matrix,
            tf_names=tfs_in_matrix,
            verbose=True,
            seed=seed,
            n_jobs=n_workers,
            client_or_address=None,
        )

        network = network.rename(columns={"regulator": "TF"})
        logger.info("  Inferred %s TF-target interactions", len(network))

        return network

    def permutation_test(
        self,
        expr_matrix: pd.DataFrame,
        n_permutations: int = 100,
        seed: int = 42,
        n_workers: int = 2,
    ) -> pd.DataFrame:
        """
        Compute null distribution of edge importances via permutation.

        Args:
            expr_matrix: Original expression matrix
            n_permutations: Number of random permutations
            seed: Random seed
            n_workers: Workers for GRNBoost2 during permutations

        Returns:
            DataFrame with null distribution statistics
        """
        logger.info("Running %s permutations for statistical validation", n_permutations)

        rng = np.random.default_rng(seed)
        null_importances: List[float] = []

        for i in range(n_permutations):
            if i % 20 == 0:
                logger.info("  Permutation %s/%s", i, n_permutations)

            expr_permuted = expr_matrix.apply(lambda col: rng.permutation(col.values), axis=0)
            network_perm = self.infer_grn(expr_permuted, n_workers=n_workers, seed=seed)
            null_importances.extend(network_perm["importance"].values.tolist())

        null_df = pd.DataFrame({"null_importances": null_importances})

        logger.info("  Computed null distribution")
        logger.info("    Mean: %.4f", np.mean(null_importances))
        logger.info("    95th percentile: %.4f", np.percentile(null_importances, 95))

        return null_df

    def filter_high_confidence_edges(
        self,
        network: pd.DataFrame,
        null_dist: Optional[pd.DataFrame] = None,
        method: str = "percentile",
        threshold: float = 0.85,
    ) -> pd.DataFrame:
        """
        Filter for statistically significant / high-importance edges.

        Args:
            network: GRN from infer_grn()
            null_dist: Null distribution from permutation_test()
            method: "percentile" (top X%) or "pvalue" (statistical significance)
            threshold: Percentile cutoff (0.85 = top 15%) or p-value threshold

        Returns:
            Filtered network DataFrame
        """
        logger.info("Filtering edges using method='%s', threshold=%s", method, threshold)

        if method == "percentile":
            cutoff = network["importance"].quantile(threshold)
            filtered = network[network["importance"] >= cutoff].copy()
            logger.info("  Retained %s/%s edges", len(filtered), len(network))
            return filtered

        if method == "pvalue":
            if null_dist is None:
                raise ValueError("null_dist required for p-value filtering")

            null_values = np.sort(null_dist["null_importances"].values)
            n_null = len(null_values)

            def compute_pvalue(value: float) -> float:
                idx = np.searchsorted(null_values, value, side="left")
                return (n_null - idx + 1) / (n_null + 1)

            network = network.copy()
            network["pvalue"] = network["importance"].apply(compute_pvalue)
            network["pvalue_adj"] = (network["pvalue"] * len(network)).clip(upper=1.0)

            filtered = network[network["pvalue_adj"] < threshold].copy()
            logger.info("  Retained %s/%s edges (p_adj < %s)", len(filtered), len(network), threshold)
            return filtered

        raise ValueError(f"Unknown method: {method}")

    def compare_subset_grns(self, subset1: str, subset2: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Infer separate GRNs for two cell subsets and compare.

        Args:
            subset1: First cell type label
            subset2: Second cell type label

        Returns:
            (filtered_grn1, filtered_grn2)
        """
        logger.info("Comparing GRNs: %s vs %s", subset1, subset2)

        expr1 = self.prepare_expression_matrix(cell_subset=subset1)
        grn1 = self.infer_grn(expr1)
        grn1_filtered = self.filter_high_confidence_edges(grn1, method="percentile", threshold=0.85)

        expr2 = self.prepare_expression_matrix(cell_subset=subset2)
        grn2 = self.infer_grn(expr2)
        grn2_filtered = self.filter_high_confidence_edges(grn2, method="percentile", threshold=0.85)

        edges1 = set(zip(grn1_filtered["TF"], grn1_filtered["target"]))
        edges2 = set(zip(grn2_filtered["TF"], grn2_filtered["target"]))

        logger.info("  %s-specific edges: %s", subset1, len(edges1 - edges2))
        logger.info("  %s-specific edges: %s", subset2, len(edges2 - edges1))
        logger.info("  Shared edges: %s", len(edges1 & edges2))

        return grn1_filtered, grn2_filtered

    def save_network(self, network: pd.DataFrame, output_path: str) -> None:
        """Save GRN to CSV and GraphML formats."""
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

        csv_path = output.with_suffix(".csv")
        network.to_csv(csv_path, index=False)
        logger.info("Saved GRN to %s", csv_path)

        import networkx as nx

        graph = nx.DiGraph()
        for _, row in network.iterrows():
            graph.add_edge(row["TF"], row["target"], weight=row["importance"])

        graphml_path = output.with_suffix(".graphml")
        nx.write_graphml(graph, graphml_path)
        logger.info("Saved GRN graph to %s", graphml_path)


if __name__ == "__main__":
    import argparse

    import scanpy as sc

    parser = argparse.ArgumentParser(description="Infer gene regulatory network")
    parser.add_argument("--adata", required=True, help="Path to preprocessed .h5ad file")
    parser.add_argument("--output", required=True, help="Output path for GRN (without extension)")
    parser.add_argument("--cell_type", default=None, help="Cell type subset (optional)")
    parser.add_argument("--n_permutations", type=int, default=0, help="Number of permutations (0=skip)")
    parser.add_argument("--n_workers", type=int, default=4, help="Workers for GRNBoost2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use_hvg", action="store_true", help="Include HVGs to reach min genes")
    parser.add_argument("--min_genes", type=int, default=400, help="Minimum genes for GRN")
    parser.add_argument("--max_genes", type=int, default=800, help="Maximum genes for GRN")

    args = parser.parse_args()

    adata = sc.read_h5ad(args.adata)

    engine = GRNInferenceEngine(adata)
    expr_matrix = engine.prepare_expression_matrix(
        cell_subset=args.cell_type,
        use_highly_variable=args.use_hvg,
        min_genes=args.min_genes,
        max_genes=args.max_genes,
    )

    grn = engine.infer_grn(expr_matrix, n_workers=args.n_workers, seed=args.seed)

    if args.n_permutations > 0:
        null_dist = engine.permutation_test(
            expr_matrix,
            n_permutations=args.n_permutations,
            seed=args.seed,
            n_workers=max(1, args.n_workers // 2),
        )
        grn_filtered = engine.filter_high_confidence_edges(
            grn,
            null_dist=null_dist,
            method="pvalue",
            threshold=0.01,
        )
    else:
        grn_filtered = engine.filter_high_confidence_edges(grn, method="percentile", threshold=0.85)

    engine.save_network(grn_filtered, args.output)

    print(f"\nGRN inference complete: {len(grn_filtered)} high-confidence edges")
