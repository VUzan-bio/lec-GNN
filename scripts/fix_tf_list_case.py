from pathlib import Path

import pandas as pd
import scanpy as sc


def main() -> None:
    adata_path = Path("data/processed/GSE282417_processed.h5ad")
    tf_path = Path("data/external/lec_comprehensive_genes.csv")
    output_path = Path("data/external/lec_comprehensive_genes_fixed.csv")

    adata = sc.read_h5ad(adata_path)
    tf_df = pd.read_csv(tf_path)

    example = str(adata.var_names[0]) if len(adata.var_names) > 0 else ""
    if example.isupper():
        tf_df["symbol"] = tf_df["symbol"].astype(str).str.upper()
        print("Converted TF list to uppercase format")
    elif example.islower():
        tf_df["symbol"] = tf_df["symbol"].astype(str).str.lower()
        print("Converted TF list to lowercase format")
    elif len(example) > 1 and example[0].isupper() and example[1:].islower():
        tf_df["symbol"] = tf_df["symbol"].astype(str).str.capitalize()
        print("Converted TF list to capitalized format")
    else:
        print("TF list already matches gene name format")

    tf_df.to_csv(output_path, index=False)

    overlap = set(tf_df["symbol"]) & set(adata.var_names)
    print(f"Overlap: {len(overlap)} / {len(tf_df)} genes")
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
