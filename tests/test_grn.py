import anndata
import numpy as np
import pandas as pd
import pytest

from src.grn.inference import GRNInferenceEngine


def test_infer_grn_requires_arboreto() -> None:
    counts = np.array([[1, 2], [3, 4]])
    var = pd.DataFrame(index=["GeneA", "GeneB"])
    adata = anndata.AnnData(X=counts, var=var)
    adata.var_names = ["GeneA", "GeneB"]

    engine = GRNInferenceEngine(adata, tf_list_path="tests/data/tfs.csv", pathway_genes_path="tests/data/pathways.csv")
    expr = pd.DataFrame(counts, columns=["GeneA", "GeneB"])

    with pytest.raises(ImportError):
        engine.infer_grn(expr)
