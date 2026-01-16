import logging
import random
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import anndata
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
from scipy.sparse import issparse

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

    def _is_bulk_like(self, adata: anndata.AnnData) -> bool:
        """Heuristic to detect bulk-like datasets by sample count."""
        return adata.n_obs < int(self.config.get("bulk_max_samples", 50))

    def _estimate_non_integer_ratio(self, adata: anndata.AnnData, sample_size: int = 10000) -> float:
        """Estimate fraction of non-integer expression values."""
        data = adata.X.data if issparse(adata.X) else np.asarray(adata.X).ravel()
        if data.size == 0:
            return 0.0
        if data.size > sample_size:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(data.size, size=sample_size, replace=False)
            data = data[idx]
        return float(np.mean(np.abs(data - np.round(data)) > 1e-6))

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

        counts_files: List[Path] = []
        for pattern in [
            "*.txt",
            "*.txt.gz",
            "*.tsv",
            "*.tsv.gz",
            "*.csv",
            "*.csv.gz",
            "*.mtx",
            "*.mtx.gz",
            "*.xlsx",
            "*.xls",
        ]:
            counts_files.extend(
                f
                for f in data_dir.rglob(pattern)
                if not f.name.endswith("sdrf.txt") and "sdrf" not in f.name.lower()
            )
        if not counts_files:
            raise FileNotFoundError(f"No counts matrix found in {data_dir}")

        counts_files = [path for path in counts_files if self._is_counts_candidate(path)]
        if not counts_files:
            raise FileNotFoundError(
                f"No usable counts matrix found in {data_dir} (files present, but none are valid counts)"
            )

        def _score_counts_file(path: Path) -> Tuple[int, int]:
            name = path.name.lower()
            score = 0
            if any(token in name for token in ["data", "count", "counts", "matrix", "expression", "processed"]):
                score += 3
            if "table" in name or "supplement" in name or "signature" in name:
                score -= 2
            if name.endswith((".mtx", ".mtx.gz", ".h5", ".h5ad")):
                score += 2
            if name.endswith((".csv", ".tsv", ".txt", ".xlsx", ".xls", ".csv.gz", ".tsv.gz", ".txt.gz")):
                score += 1
            size = path.stat().st_size
            return score, size

        counts_files = sorted(counts_files, key=_score_counts_file, reverse=True)
        counts_file = counts_files[0]
        logger.info("  Selected counts file: %s", counts_file.name)
        metadata_file = data_dir / f"{accession}.sdrf.txt"

        if counts_file.suffix in {".xlsx", ".xls"}:
            adata = self._load_excel_counts(counts_file)
        elif counts_file.suffix == ".mtx" or counts_file.name.endswith(".mtx.gz"):
            adata = sc.read_mtx(counts_file).T

            def _find_aux_file(directory: Path, names: List[str]) -> Optional[Path]:
                for name in names:
                    candidate = directory / name
                    if candidate.exists():
                        return candidate
                return None

            barcodes_file = _find_aux_file(
                counts_file.parent,
                ["barcodes.tsv", "barcodes.tsv.gz", "cells.tsv", "cells.tsv.gz"],
            )
            genes_file = _find_aux_file(
                counts_file.parent,
                ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"],
            )

            if barcodes_file is not None:
                barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")[0].astype(str).tolist()
                if len(barcodes) == adata.n_obs:
                    adata.obs_names = barcodes
            if genes_file is not None:
                genes = pd.read_csv(genes_file, header=None, sep="\t")[0].astype(str).tolist()
                if len(genes) == adata.n_vars:
                    adata.var_names = genes
        else:
            sep = "," if counts_file.suffix == ".csv" or counts_file.name.endswith(".csv.gz") else "\t"
            counts = pd.read_csv(counts_file, sep=sep, index_col=0)
            adata = anndata.AnnData(
                X=counts.T.values,
                obs=pd.DataFrame(index=counts.columns),
                var=pd.DataFrame(index=counts.index),
            )

        metadata = pd.DataFrame(index=adata.obs_names)
        if metadata_file.exists():
            metadata = pd.read_csv(metadata_file, sep="\t")
            for col in ["Source Name", "Sample Name", "Assay Name", "Scan Name"]:
                if col in metadata.columns:
                    metadata.index = metadata[col].astype(str)
                    break

        if not metadata.empty:
            common = metadata.index.intersection(adata.obs_names)
            if len(common) > 0:
                metadata = metadata.reindex(adata.obs_names)
                adata.obs = metadata
            else:
                logger.warning("SDRF metadata index does not match counts matrix; keeping default obs.")

        logger.info("  Loaded %s cells x %s genes", adata.n_obs, adata.n_vars)
        return adata

    def _is_counts_candidate(self, path: Path) -> bool:
        """Filter out non-data files masquerading as spreadsheets."""
        suffix = path.suffix.lower()
        signature = self._read_file_signature(path)
        if signature.startswith(b"%PDF"):
            logger.warning("  Skipping %s: file is PDF despite data extension", path.name)
            return False
        if suffix in {".xlsx", ".xls"} and not (
            signature.startswith(b"PK") or signature.startswith(b"\xD0\xCF\x11\xE0")
        ):
            logger.warning("  Skipping %s: unsupported Excel signature", path.name)
            return False
        return True

    def _read_file_signature(self, path: Path) -> bytes:
        """Read the first few bytes of a file for signature detection."""
        try:
            with path.open("rb") as handle:
                return handle.read(8)
        except OSError:
            return b""

    def _load_excel_counts(self, counts_file: Path) -> anndata.AnnData:
        """
        Load counts from an Excel file by selecting the most matrix-like sheet.

        Args:
            counts_file: Path to Excel file

        Returns:
            AnnData object
        """
        logger.info("  Loading Excel counts from %s", counts_file.name)

        with counts_file.open("rb") as handle:
            signature = handle.read(8)

        if signature.startswith(b"PK"):
            engine = "openpyxl"
        elif signature.startswith(b"\xD0\xCF\x11\xE0"):
            engine = "xlrd"
        else:
            raise ValueError(
                f"Unsupported Excel format for {counts_file}. "
                "File does not appear to be a valid .xlsx or .xls."
            )

        excel = pd.ExcelFile(counts_file, engine=engine)
        best_sheet = None
        best_score = -1

        for sheet in excel.sheet_names:
            df = pd.read_excel(excel, sheet_name=sheet, engine=engine)
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")
            if df.shape[0] < 2 or df.shape[1] < 2:
                continue

            first_col = df.iloc[:, 0].astype(str)
            numeric_block = df.iloc[:, 1:]
            numeric_ratio = numeric_block.apply(pd.to_numeric, errors="coerce").notna().mean().mean()
            gene_like = first_col.str.isalpha().mean()
            score = numeric_ratio + gene_like

            if score > best_score:
                best_score = score
                best_sheet = sheet

        if best_sheet is None:
            raise ValueError(f"No suitable count matrix sheet found in {counts_file}")

        df = pd.read_excel(excel, sheet_name=best_sheet, engine=engine)
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        df = df.set_index(df.columns[0])
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.fillna(0)

        logger.info("  Using sheet '%s' with shape %s", best_sheet, df.shape)

        return anndata.AnnData(X=df.T.values, obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))

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

        candidate_dirs = [data_dir / "supplementary", data_dir]
        expression_files: List[Path] = []
        patterns = [
            "*.h5ad",
            "*.h5",
            "*.mtx",
            "*.mtx.gz",
            "*.txt",
            "*.txt.gz",
            "*.csv",
            "*.tsv",
            "*.tsv.gz",
            "*.xlsx",
            "*.xls",
        ]

        for directory in candidate_dirs:
            if not directory.exists():
                continue
            for pattern in patterns:
                expression_files.extend(directory.rglob(pattern))

        # Deduplicate paths when scanning both data_dir and its supplementary subdir.
        expression_files = list({path.resolve(): path for path in expression_files}.values())
        expression_files = [path for path in expression_files if self._is_geo_expression_file(path)]

        if not expression_files:
            raise FileNotFoundError(f"No count matrix found in {data_dir}")

        h5_files = [path for path in expression_files if path.suffix == ".h5ad"]
        if h5_files:
            adata_list = []
            keys = self._make_unique_keys(h5_files)
            for path in h5_files:
                adata = sc.read_h5ad(path)
                adata.var_names_make_unique()
                adata_list.append(adata)
            adata = anndata.concat(adata_list, label="batch", keys=keys)
            adata.obs_names_make_unique()
            logger.info("  Loaded %s cells x %s genes from %s h5ad files", adata.n_obs, adata.n_vars, len(h5_files))
            return adata

        h5_files = [path for path in expression_files if path.suffix == ".h5"]
        if h5_files:
            filtered = [path for path in h5_files if "cmo" not in path.name.lower() and "hto" not in path.name.lower()]
            h5_files = filtered if filtered else h5_files
            adata_list = []
            keys = self._make_unique_keys(h5_files)
            for path in h5_files:
                adata = sc.read_10x_h5(path)
                adata.var_names_make_unique()
                adata_list.append(adata)
            adata = anndata.concat(adata_list, label="batch", keys=keys)
            adata.obs_names_make_unique()
            logger.info("  Loaded %s cells x %s genes from %s h5 files", adata.n_obs, adata.n_vars, len(h5_files))
            return adata

        mtx_dirs = []
        for path in expression_files:
            if path.suffix in {".mtx", ".gz"} and "mtx" in path.name:
                mtx_dirs.append(path.parent)
        for directory in sorted(set(mtx_dirs)):
            try:
                adata = sc.read_10x_mtx(directory)
                logger.info("  Loaded 10x matrix from %s", directory)
                return adata
            except Exception:
                continue

        excel_files = [path for path in expression_files if path.suffix in {".xlsx", ".xls"}]
        if excel_files:
            excel_files = sorted(excel_files, key=self._score_geo_excel_file, reverse=True)
            normalized_excels = [
                path
                for path in excel_files
                if "normalized" in path.name.lower() and "gene" in path.name.lower()
            ]
            selected_excels = normalized_excels if normalized_excels else [excel_files[0]]

            adata_list = []
            keys = self._make_unique_keys(selected_excels)
            for path in selected_excels:
                logger.info("  Loading Excel counts from %s", path.name)
                adata = self._load_excel_counts(path)
                adata.var_names_make_unique()
                adata.obs_names_make_unique()
                adata_list.append(adata)

            if len(adata_list) == 1:
                adata = adata_list[0]
            else:
                adata = anndata.concat(adata_list, join="outer", label="batch", keys=keys)
                adata.obs_names_make_unique()
                adata.var_names_make_unique()
            logger.info("  Loaded %s cells x %s genes from %s Excel files", adata.n_obs, adata.n_vars, len(adata_list))
            return adata

        text_files = [
            path
            for path in expression_files
            if path.suffix in {".txt", ".csv", ".tsv"} or path.name.endswith((".txt.gz", ".csv.gz", ".tsv.gz"))
        ]
        matrix_files = [path for path in text_files if "series_matrix" in path.name.lower()]
        if matrix_files:
            filepath = matrix_files[0]
            adata = self._load_geo_series_matrix(filepath)
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
            logger.info("  Loaded %s cells x %s genes from %s", adata.n_obs, adata.n_vars, filepath.name)
            return adata

        if len(text_files) > 1:
            adata = self._load_geo_per_sample_tables(text_files)
            adata.var_names_make_unique()
            adata.obs_names_make_unique()
            logger.info("  Loaded %s samples x %s genes from %s per-sample files", adata.n_obs, adata.n_vars, len(text_files))
            return adata

        filepath = text_files[0] if text_files else expression_files[0]
        adata = self._load_geo_matrix_file(filepath)
        adata.var_names_make_unique()
        adata.obs_names_make_unique()
        logger.info("  Loaded %s cells x %s genes from %s", adata.n_obs, adata.n_vars, filepath.name)
        return adata

    def _make_unique_keys(self, paths: List[Path]) -> List[str]:
        """Generate unique keys for concatenation based on file stems."""
        seen: Dict[str, int] = {}
        keys: List[str] = []
        for path in paths:
            base = path.stem
            count = seen.get(base, 0)
            key = base if count == 0 else f"{base}_{count + 1}"
            seen[base] = count + 1
            keys.append(key)
        return keys

    def _is_geo_expression_file(self, path: Path) -> bool:
        """Filter out GEO metadata files."""
        name = path.name.lower()
        if name.endswith(("family.soft", "family.soft.gz", ".soft", ".soft.gz")):
            return False
        if name == "sample_metadata.csv":
            return False
        if "metadata" in name and name.endswith((".csv", ".tsv", ".txt")):
            return False
        if "comparison" in name and name.endswith((".xlsx", ".xls")):
            return False
        return True

    def _score_geo_excel_file(self, path: Path) -> Tuple[int, int]:
        """Score GEO Excel files to prefer normalized gene expression tables."""
        name = path.name.lower()
        score = 0
        if "normalized" in name:
            score += 3
        if "gene" in name:
            score += 2
        if "expression" in name:
            score += 2
        if "comparison" in name:
            score -= 3
        if "table" in name:
            score -= 1
        size = path.stat().st_size
        return score, size

    def _load_geo_series_matrix(self, path: Path) -> anndata.AnnData:
        """Load GEO series matrix files with table delimiters."""
        import gzip

        rows = []
        in_table = False
        open_func = Path.open if path.suffix != ".gz" else lambda p, **kwargs: gzip.open(p, **kwargs)  # type: ignore[assignment]

        with open_func(path, "rt", encoding="utf-8", errors="ignore") as handle:  # type: ignore[arg-type]
            for line in handle:
                if line.startswith("!series_matrix_table_begin"):
                    in_table = True
                    continue
                if line.startswith("!series_matrix_table_end"):
                    break
                if in_table:
                    rows.append(line.strip().split("\t"))

        if not rows:
            raise ValueError(f"No expression table found in series matrix: {path}")

        header = rows[0]
        data = rows[1:]
        df = pd.DataFrame(data, columns=header)
        df = df.set_index(df.columns[0])
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.fillna(0)
        return anndata.AnnData(X=df.T.values, obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))

    def _load_geo_matrix_file(self, path: Path) -> anndata.AnnData:
        """Load a matrix-like table into AnnData."""
        sep = "," if path.suffix == ".csv" or path.name.endswith(".csv.gz") else "\t"
        df = pd.read_csv(path, sep=sep, index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df = df.dropna(axis=1, how="all")
        df = df.fillna(0)
        return anndata.AnnData(X=df.T.values, obs=pd.DataFrame(index=df.columns), var=pd.DataFrame(index=df.index))

    def _select_numeric_column(self, df: pd.DataFrame) -> str:
        """Select the most numeric column in a table."""
        best_col = df.columns[1]
        best_score = -1.0
        for col in df.columns[1:]:
            numeric_ratio = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if numeric_ratio > best_score:
                best_score = numeric_ratio
                best_col = col
        return best_col

    def _extract_sample_id(self, filename: str) -> str:
        """Extract a sample ID from a filename."""
        match = re.search(r"GSM\\d+", filename)
        if match:
            return match.group(0)
        return Path(filename).stem

    def _load_geo_per_sample_tables(self, paths: List[Path]) -> anndata.AnnData:
        """Merge per-sample gene count tables into a matrix."""
        merged: Optional[pd.DataFrame] = None
        sample_names: List[str] = []

        for path in sorted(paths):
            sep = "," if path.suffix == ".csv" or path.name.endswith(".csv.gz") else "\t"
            df = pd.read_csv(path, sep=sep)
            df = df.dropna(axis=1, how="all")
            df = df.dropna(axis=0, how="all")

            if df.shape[1] < 2:
                continue

            sample_id = self._extract_sample_id(path.name)
            value_col = self._select_numeric_column(df)
            counts = df[[df.columns[0], value_col]].copy()
            counts.columns = ["gene", sample_id]
            counts[sample_id] = pd.to_numeric(counts[sample_id], errors="coerce").fillna(0)

            if merged is None:
                merged = counts.set_index("gene")
            else:
                merged = merged.join(counts.set_index("gene"), how="outer")
            sample_names.append(sample_id)

        if merged is None:
            raise FileNotFoundError("No usable per-sample tables found for GEO dataset")

        merged = merged.fillna(0)
        return anndata.AnnData(X=merged.T.values, obs=pd.DataFrame(index=merged.columns), var=pd.DataFrame(index=merged.index))

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

        is_bulk = self._is_bulk_like(adata)
        mt_prefix = "MT-" if species == "human" else "mt-"
        adata.var["mt"] = adata.var_names.str.startswith(mt_prefix)

        sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)

        min_genes = int(self.qc_params["min_genes_per_cell"])
        max_genes = int(self.qc_params["max_genes_per_cell"])
        min_counts = int(self.qc_params["min_counts_per_cell"])
        max_mito = float(self.qc_params["max_mitochondrial_pct"])

        if is_bulk:
            logger.info("Detected bulk-like dataset (n_obs=%s); skipping cell-level QC filters.", adata.n_obs)
        else:
            adata = adata[adata.obs["n_genes_by_counts"] >= min_genes].copy()
            adata = adata[adata.obs["n_genes_by_counts"] <= max_genes].copy()
            adata = adata[adata.obs["total_counts"] >= min_counts].copy()
            adata = adata[adata.obs["pct_counts_mt"] < max_mito].copy()

        min_cells = int(self.qc_params["min_cells_per_gene"])
        min_cells = min(min_cells, adata.n_obs)
        sc.pp.filter_genes(adata, min_cells=min_cells)

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

        is_bulk = self._is_bulk_like(adata)
        non_integer_ratio = self._estimate_non_integer_ratio(adata)
        use_counts_layer = "counts" if "counts" in adata.layers else None

        if non_integer_ratio > 0.05:
            logger.info("Detected non-integer expression values; skipping total-count normalization.")
            sample = adata.X.data if issparse(adata.X) else np.asarray(adata.X).ravel()
            max_value = float(np.nanmax(sample)) if sample.size else 0.0
            if max_value > 50:
                sc.pp.log1p(adata)
            adata.layers["log1p_norm"] = adata.X.copy()
        else:
            sc.pp.normalize_total(adata, target_sum=self.norm_params["target_sum"])
            sc.pp.log1p(adata)
            adata.layers["log1p_norm"] = adata.X.copy()

        if is_bulk:
            logger.info("Bulk-like dataset detected; skipping HVG selection.")
            adata.var["highly_variable"] = True
        else:
            sc.pp.highly_variable_genes(
                adata,
                n_top_genes=self.norm_params["highly_variable_genes"]["n_top_genes"],
                flavor=self.norm_params["highly_variable_genes"]["flavor"],
                batch_key=None,
                subset=False,
                layer=use_counts_layer,
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
        max_pcs = max(1, min(n_pcs, adata.n_obs - 1, adata.n_vars - 1))
        use_hvg = not self._is_bulk_like(adata)
        sc.tl.pca(adata, n_comps=max_pcs, use_highly_variable=use_hvg, random_state=self.random_seed)

        adata = self.integrate_batches(adata)

        use_rep = "X_pca_harmony" if "X_pca_harmony" in adata.obsm else None
        sc.pp.neighbors(
            adata,
            n_neighbors=int(self.cluster_params.get("n_neighbors", 15)),
            use_rep=use_rep,
        )
        try:
            sc.tl.leiden(
                adata,
                resolution=float(self.cluster_params.get("resolution", 0.5)),
                random_state=self.random_seed,
            )
        except ImportError as exc:
            logger.warning("Leiden skipped (missing dependency): %s", exc)

        try:
            sc.tl.umap(adata, random_state=self.random_seed)
        except ImportError as exc:
            logger.warning("UMAP skipped (missing dependency): %s", exc)

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
