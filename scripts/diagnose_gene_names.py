import sys

import pandas as pd
import scanpy as sc


def diagnose_gene_names(adata_path: str, tf_list_path: str) -> str:
    """Diagnose gene name format mismatch between AnnData and TF list."""
    print("Loading data...")
    adata = sc.read_h5ad(adata_path)
    tf_df = pd.read_csv(tf_list_path)
    if "symbol" in tf_df.columns:
        tf_list = tf_df["symbol"].astype(str).tolist()
    elif "gene" in tf_df.columns:
        tf_list = tf_df["gene"].astype(str).tolist()
    else:
        raise ValueError("TF list must contain a 'symbol' or 'gene' column")

    print("\n" + "=" * 80)
    print("GENE NAME FORMAT DIAGNOSIS")
    print("=" * 80)

    print("\nAnnData Gene Format:")
    print(f"  Total genes: {len(adata.var_names)}")
    print(f"  First 20 genes: {list(adata.var_names[:20])}")
    example = str(adata.var_names[0]) if len(adata.var_names) > 0 else ""
    print(f"  Example gene: '{example}'")

    format_type = "unknown"
    if example.startswith("ENSG"):
        format_type = "ensembl_id"
        print(f"  Format detected: Ensembl ID (e.g., {example})")
    elif "_" in example and "ENSG" in example:
        format_type = "symbol_ensembl"
        print(f"  Format detected: Symbol_EnsemblID (e.g., {example})")
    elif example.isupper():
        format_type = "uppercase"
        print(f"  Format detected: Uppercase (e.g., {example})")
    elif example.islower():
        format_type = "lowercase"
        print(f"  Format detected: Lowercase (e.g., {example})")
    elif len(example) > 1 and example[0].isupper() and example[1:].islower():
        format_type = "capitalized"
        print(f"  Format detected: Capitalized (e.g., {example})")
    else:
        print("  Format detected: Unknown")

    print("\nAnnData .var columns:")
    print(f"  {adata.var.columns.tolist()}")

    has_gene_name = "gene_name" in adata.var.columns
    has_gene_symbol = "gene_symbol" in adata.var.columns

    if has_gene_name:
        print("  Has 'gene_name' column")
        print(f"    First 10: {adata.var['gene_name'].head(10).tolist()}")
    if has_gene_symbol:
        print("  Has 'gene_symbol' column")
        print(f"    First 10: {adata.var['gene_symbol'].head(10).tolist()}")

    print("\nTF List vs AnnData Overlap:")
    print(f"  TF list size: {len(tf_list)}")

    test_tfs = ["NFKB1", "PROX1", "SOX18", "GATA2", "FOXC2"]
    print(f"  Testing {len(test_tfs)} key TFs:")

    for tf in test_tfs:
        in_var_names = tf in adata.var_names
        in_gene_name = has_gene_name and tf in adata.var["gene_name"].values
        in_gene_symbol = has_gene_symbol and tf in adata.var["gene_symbol"].values

        status = []
        if in_var_names:
            status.append("var_names")
        if in_gene_name:
            status.append("gene_name")
        if in_gene_symbol:
            status.append("gene_symbol")

        if status:
            print(f"    {tf}: FOUND in {', '.join(status)}")
        else:
            print(f"    {tf}: NOT FOUND")

    overlap_var_names = set(tf_list) & set(adata.var_names)
    overlap_gene_name = set(tf_list) & set(adata.var["gene_name"].values) if has_gene_name else set()
    overlap_gene_symbol = set(tf_list) & set(adata.var["gene_symbol"].values) if has_gene_symbol else set()

    print("\nOverlap Summary:")
    print(f"  TF list ∩ var_names: {len(overlap_var_names)} genes")
    print(f"  TF list ∩ gene_name: {len(overlap_gene_name)} genes")
    print(f"  TF list ∩ gene_symbol: {len(overlap_gene_symbol)} genes")
    if overlap_var_names:
        print(f"  Example matches: {list(overlap_var_names)[:10]}")

    print("\n" + "=" * 80)
    print("RECOMMENDED FIX")
    print("=" * 80)

    if len(overlap_var_names) > 100:
        print("Gene names already match. No fix needed.")
        return "no_fix_needed"
    if format_type == "ensembl_id":
        print("Fix: Convert Ensembl IDs to gene symbols using .var['gene_name']")
        return "ensembl_to_symbol"
    if format_type == "symbol_ensembl":
        print("Fix: Strip Ensembl suffix from var_names (keep only symbol)")
        return "strip_suffix"
    if format_type in {"capitalized", "lowercase"}:
        print(f"Fix: Convert TF list to {format_type} format")
        return "convert_case"
    if len(overlap_gene_name) > 100:
        print("Fix: Use .var['gene_name'] as var_names")
        return "use_gene_name_column"

    print("Unable to determine fix automatically. Manual inspection required.")
    return "manual_inspection"


if __name__ == "__main__":
    adata_path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/GSE282417_processed.h5ad"
    tf_list_path = sys.argv[2] if len(sys.argv) > 2 else "data/external/lec_comprehensive_genes.csv"

    fix_type = diagnose_gene_names(adata_path, tf_list_path)
    print(f"\nDiagnostic complete. Fix type: {fix_type}")
