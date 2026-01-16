from pathlib import Path

import anndata
import numpy as np
import pandas as pd

from src.data.preprocessing import LECDataPreprocessor


def test_normalization_layers(tmp_path: Path) -> None:
    config_path = tmp_path / "preprocessing.yaml"
    config_path.write_text(
        """
random_seed: 1
quality_control:
  min_genes_per_cell: 0
  max_genes_per_cell: 9999
  min_counts_per_cell: 0
  max_mitochondrial_pct: 100
  min_cells_per_gene: 0
normalization:
  method: "log1p"
  target_sum: 1000
  highly_variable_genes:
    n_top_genes: 5
    flavor: "seurat_v3"
batch_correction:
  method: "harmony"
  batch_key: "dataset"
clustering:
  resolution: 0.5
  n_neighbors: 5
  n_pcs: 5
cell_type_annotation:
  marker_genes:
    test_type: ["Gene0", "Gene1"]
""",
        encoding="utf-8",
    )

    rng = np.random.default_rng(0)
    counts = rng.integers(0, 5, size=(10, 8))
    var = pd.DataFrame(index=[f"Gene{i}" for i in range(8)])
    adata = anndata.AnnData(X=counts, var=var)

    preprocessor = LECDataPreprocessor(config_path=str(config_path))
    adata = preprocessor.quality_control(adata, species="mouse")
    adata = preprocessor.normalize(adata)

    assert "counts" in adata.layers
    assert "log1p_norm" in adata.layers
    assert "highly_variable" in adata.var
