from pathlib import Path

import scanpy as sc


def main() -> None:
    adata_path = Path("data/processed/GSE282417_processed.h5ad")
    output_path = Path("data/processed/GSE282417_processed_symbols.h5ad")

    adata = sc.read_h5ad(adata_path)
    adata.var["original_name"] = adata.var_names
    adata.var_names = [str(gene).split("_")[0] for gene in adata.var_names]
    adata.var_names_make_unique()

    adata.write_h5ad(output_path)
    print("Cleaned gene names (removed suffixes)")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
