from pathlib import Path

import scanpy as sc


def main() -> None:
    adata_path = Path("data/processed/GSE282417_processed.h5ad")
    output_path = Path("data/processed/GSE282417_processed_symbols.h5ad")

    adata = sc.read_h5ad(adata_path)

    if "gene_name" in adata.var.columns:
        print("Using existing 'gene_name' column...")
        adata.var["ensembl_id"] = adata.var_names
        adata.var_names = adata.var["gene_name"].astype(str)
    else:
        try:
            import pybiomart as pbm
        except ImportError as exc:
            raise ImportError(
                "pybiomart is required to query Ensembl. Install pybiomart or "
                "provide a gene_name column."
            ) from exc

        print("Querying Ensembl BioMart for gene symbols...")
        dataset = pbm.Dataset(name="hsapiens_gene_ensembl", host="http://www.ensembl.org")
        ensembl_ids = [str(x).split(".")[0] for x in adata.var_names]

        results = dataset.query(
            attributes=["ensembl_gene_id", "external_gene_name"],
            filters={"ensembl_gene_id": ensembl_ids},
        )

        id_to_symbol = dict(zip(results["Gene stable ID"], results["Gene name"]))
        adata.var["gene_symbol"] = [id_to_symbol.get(x, x) for x in ensembl_ids]
        adata.var["ensembl_id"] = adata.var_names
        adata.var_names = adata.var["gene_symbol"].astype(str)

    adata.var_names_make_unique()
    adata.write_h5ad(output_path)
    print(f"Saved fixed AnnData with {len(adata.var_names)} genes to {output_path}")


if __name__ == "__main__":
    main()
