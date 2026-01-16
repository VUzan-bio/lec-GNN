import logging
import random
from pathlib import Path
from typing import Dict, List, Optional

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import yaml

logger = logging.getLogger(__name__)


class LECDataPreprocessor:
    """
    Quality control and preprocessing pipeline for LEC scRNA-seq data.

    Follows best practices from:
    - Luecken and Theis 2019 (Mol Syst Biol)
    - scanpy tutorials (sc-best-practices.org)
    """

    def __init__(self, config_path: str = "config/preprocessing_params.yaml"):
        """
        Initialize preprocessor with configuration.

        Args:
            config_path: Path to preprocessing parameters YAML
        """
        with open(config_path, "r", encoding="utf-8") as handle:
            self.config = yaml.safe_load(handle)

        self.qc_params = self.config["quality_control"]
        self.norm_params = self.config["normalization"]
        self.batch_params = self.config.get("batch_correction", {})
        self.cluster_params = self.config.get("clustering", {})

        self.random_seed = int(self.config.get("random_seed", 42))
        self._set_seed(self.random_seed)

        logging.basicConfig(level=logging.INFO)

    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    def load_arrayexpress_data(self, data_dir: Path, accession: str) -> anndata.AnnData:
        """
        Load processed counts from ArrayExpress download.

        Args:
            data_dir: Path to E-MTAB-XXXX directory
            accession: Dataset accession (e.g., "E-MTAB-8414")

        Returns:
            AnnData object with raw counts
        """
        logger.info("Loading ArrayExpress data: %s", accession)

        counts_files = []
        for pattern in ["*.txt", "*.txt.gz", "*.tsv", "*.tsv.gz"]:
            counts_files.extend(
                f
                for f in data_dir.glob(pattern)
                if not f.name.endswith("sdrf.txt") and "sdrf" not in f.name.lower()
            )
        if not counts_files:
            raise FileNotFoundError(f"No counts matrix found in {data_dir}")

        counts_file = counts_files[0]
        metadata_file = data_dir / f"{accession}.sdrf.txt"

        counts = pd.read_csv(counts_file, sep="\t", index_col=0)

        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file, sep="\t")
            if "Source Name" in metadata.columns:
                metadata.index = metadata["Source Name"].astype(str)
        else:
            metadata = pd.DataFrame(index=counts.columns)

        adata = anndata.AnnData(
            X=counts.T.values,
            obs=metadata,
            var=pd.DataFrame(index=counts.index),
        )

        logger.info("  Loaded %s cells x %s genes", adata.n_obs, adata.n_vars)
        return adata

    def load_geo_data(self, data_dir: Path, accession: str) -> anndata.AnnData:
        """
        Load GEO dataset (handle multiple formats).

        Args:
            data_dir: Path to GSE directory
            accession: GEO accession

        Returns:
            AnnData object
        """
        logger.info("Loading GEO data: %s", accession)

        candidate_dirs = [data_dir, data_dir / "supplementary"]
        for directory in candidate_dirs:
            if not directory.exists():
                continue

            if (directory / "matrix.mtx").exists() or list(directory.glob("*matrix.mtx*")):
                try:
                    adata = sc.read_10x_mtx(directory)
                    logger.info("  Loaded 10x matrix from %s", directory)
                    return adata
                except Exception:
                    pass

        patterns = ["*.h5ad", "*.h5", "*.mtx", "*.mtx.gz", "*.txt", "*.txt.gz", "*.csv", "*.tsv", "*.tsv.gz"]
        filepath: Optional[Path] = None
        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for pattern in patterns:
                files = list(directory.glob(pattern))
                if files:
                    filepath = files[0]
                    break
            if filepath:
                break

        if filepath is None:
            raise FileNotFoundError(f"No count matrix found in {data_dir}")

        if filepath.suffix == ".h5ad":
            adata = sc.read_h5ad(filepath)
        elif filepath.suffix == ".h5":
            adata = sc.read_10x_h5(filepath)
        elif filepath.suffix == ".mtx":
            adata = sc.read_mtx(filepath).T
        else:
            adata = sc.read_text(filepath).T

        logger.info("  Loaded %s cells x %s genes", adata.n_obs, adata.n_vars)
        return adata

    def quality_control(self, adata: anndata.AnnData, species: str = "mouse") -> anndata.AnnData:
        """
        Apply quality control filters.

        Args:
            adata: Raw AnnData object
            species: "mouse" or "human" (for mitochondrial gene detection)

        Returns:
            Filtered AnnData
        """
        logger.info("Running quality control")

        n_cells_before = adata.n_obs
        n_genes_before = adata.n_vars

        mt_prefix = "MT-" if species == "human" else "mt-"
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

        min_genes = int(self.qc_params["min_genes_per_cell"])
        max_genes = int(self.qc_params["max_genes_per_cell"])
        min_counts = int(self.qc_params["min_counts_per_cell"])
        max_mito = float(self.qc_params["max_mitochondrial_pct"])

        adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
        adata = adata[adata.obs["n_genes_by_counts"] <= max_genes].copy()
        adata = adata[adata.obs["total_counts"] >= min_counts].copy()
        adata = adata[adata.obs["pct_counts_mt"] < max_mito].copy()

        sc.pp.filter_genes(adata, min_cells=int(self.qc_params["min_cells_per_gene"]))

        n_cells_after = adata.n_obs
        n_genes_after = adata.n_vars

        logger.info(
            "  Cells: %s -> %s (%.1f%% retained)",
            n_cells_before,
            n_cells_after,
            100 * n_cells_after / max(n_cells_before, 1),
        )
        logger.info(
            "  Genes: %s -> %s (%.1f%% retained)",
            n_genes_before,
            n_genes_after,
            100 * n_genes_after / max(n_genes_before, 1),
        )

        return adata

    def normalize(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Normalize and log-transform counts.

        Args:
            adata: Filtered AnnData

        Returns:
            Normalized AnnData
        """
        logger.info("Normalizing counts")

        adata.layers["counts"] = adata.X.copy()

        sc.pp.normalize_total(adata, target_sum=self.norm_params["target_sum"])
        sc.pp.log1p(adata)

        adata.layers["log1p_norm"] = adata.X.copy()

        sc.pp.highly_variable_genes(
            adata,
            n_top_genes=self.norm_params["highly_variable_genes"]["n_top_genes"],
            flavor=self.norm_params["highly_variable_genes"]["flavor"],
            batch_key=None,
            subset=False,
        )

        n_hvg = int(adata.var["highly_variable"].sum())
        logger.info("  Identified %s highly variable genes", n_hvg)

        return adata

    def integrate_batches(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Integrate batches using Harmony if requested.

        Args:
            adata: AnnData with PCA computed

        Returns:
            AnnData with Harmony-corrected embeddings in .obsm
        """
        method = self.batch_params.get("method")
        batch_key = self.batch_params.get("batch_key")

        if method != "harmony" or not batch_key:
            return adata

        if batch_key not in adata.obs:
            logger.warning("Batch key %s not found in adata.obs", batch_key)
            return adata

        if adata.obs[batch_key].nunique() < 2:
            logger.info("Only one batch present, skipping Harmony")
            return adata

        try:
            from scanpy.external.pp import harmony_integrate
        except ImportError as exc:
            raise ImportError("Harmony integration requires scanpy[external] and harmonypy.") from exc

        logger.info("Running Harmony batch correction on %s", batch_key)
        harmony_integrate(adata, key=batch_key)
        return adata

    def annotate_cell_types(self, adata: anndata.AnnData) -> anndata.AnnData:
        """
        Annotate LEC subsets using marker genes.

        Args:
            adata: Normalized AnnData with clustering

        Returns:
            AnnData with "cell_type" annotation in .obs
        """
        logger.info("Annotating cell types with marker genes")

        markers: Dict[str, List[str]] = self.config["cell_type_annotation"]["marker_genes"]

        for cell_type, genes in markers.items():
            genes_present = [g for g in genes if g in adata.var_names]
            if genes_present:
                sc.tl.score_genes(adata, genes_present, score_name=f"{cell_type}_score")

        score_cols = [col for col in adata.obs.columns if col.endswith("_score")]
        if not score_cols:
            adata.obs["cell_type"] = "unknown"
            logger.warning("No marker genes found in dataset for annotation")
            return adata

        adata.obs["cell_type"] = adata.obs[score_cols].idxmax(axis=1).str.replace("_score", "")

        logger.info("  Cell type distribution:")
        for cell_type, count in adata.obs["cell_type"].value_counts().items():
            logger.info("    %s: %s cells", cell_type, count)

        return adata

    def process_dataset(
        self,
        data_dir: Path,
        accession: str,
        species: str,
        source: str,
        output_dir: Path = Path("data/processed"),
    ) -> anndata.AnnData:
        """
        Complete preprocessing pipeline for one dataset.

        Args:
            data_dir: Path to raw data directory
            accession: Dataset ID
            species: "mouse" or "human"
            source: "arrayexpress" or "geo"
            output_dir: Directory for processed .h5ad files

        Returns:
            Preprocessed AnnData ready for analysis
        """
        logger.info("=" * 60)
        logger.info("Processing %s", accession)
        logger.info("=" * 60)

        if source == "arrayexpress":
            adata = self.load_arrayexpress_data(data_dir, accession)
        else:
            adata = self.load_geo_data(data_dir, accession)

        adata.obs["dataset"] = accession
        adata.obs["species"] = species

        adata = self.quality_control(adata, species=species)
        adata = self.normalize(adata)

        n_pcs = int(self.cluster_params.get("n_pcs", 50))
        sc.tl.pca(adata, n_comps=n_pcs, use_highly_variable=True, random_state=self.random_seed)

        adata = self.integrate_batches(adata)

        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else None
        sc.pp.neighbors(
            adata,
            n_neighbors=int(self.cluster_params.get("n_neighbors", 15)),
            use_rep=use_rep,
        )
        sc.tl.leiden(adata, resolution=float(self.cluster_params.get("resolution", 0.5)), random_state=self.random_seed)
        sc.tl.umap(adata, random_state=self.random_seed)

        adata = self.annotate_cell_types(adata)

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{accession}_processed.h5ad"
        adata.write_h5ad(output_path)

        logger.info("Saved to %s", output_path)

        return adata


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess LEC scRNA-seq data")
    parser.add_argument("--data_dir", required=True, help="Path to raw data directory")
    parser.add_argument("--accession", required=True, help="Dataset accession ID")
    parser.add_argument("--species", choices=["mouse", "human"], required=True)
    parser.add_argument("--source", choices=["arrayexpress", "geo"], required=True)
    parser.add_argument("--output", default="data/processed", help="Output directory")

    args = parser.parse_args()

    preprocessor = LECDataPreprocessor()
    adata = preprocessor.process_dataset(
        Path(args.data_dir), args.accession, args.species, args.source, Path(args.output)
    )

    print(f"\nProcessed dataset: {adata.n_obs} cells x {adata.n_vars} genes")
