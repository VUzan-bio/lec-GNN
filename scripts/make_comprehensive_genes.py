import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comprehensive LEC gene list.")
    parser.add_argument(
        "--output",
        default="data/external/lec_comprehensive_genes.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    tf_list = [
        "NFKB1",
        "NFKB2",
        "RELA",
        "RELB",
        "REL",
        "PROX1",
        "SOX18",
        "SOX7",
        "SOX17",
        "FOXC1",
        "FOXC2",
        "GATA2",
        "GATA3",
        "ERG",
        "FLI1",
        "ETV2",
        "KLF2",
        "KLF4",
        "STAT1",
        "STAT3",
        "STAT5A",
        "STAT5B",
        "IRF1",
        "IRF3",
        "IRF7",
        "HIF1A",
        "EPAS1",
        "ARNT",
        "ETS1",
        "ETS2",
        "ELK3",
        "TAL1",
        "GATA4",
    ]

    glyco_genes = [
        "ST3GAL1",
        "ST3GAL2",
        "ST3GAL3",
        "ST3GAL4",
        "ST3GAL5",
        "ST3GAL6",
        "ST6GAL1",
        "ST6GAL2",
        "ST6GALNAC1",
        "ST6GALNAC2",
        "ST6GALNAC3",
        "ST6GALNAC4",
        "ST6GALNAC5",
        "ST6GALNAC6",
        "ST8SIA1",
        "ST8SIA4",
        "GALNT1",
        "GALNT2",
        "GALNT3",
        "GALNT4",
        "GALNT5",
        "GALNT6",
        "GALNT7",
        "GALNT10",
        "GALNT11",
        "GALNT12",
        "GALNT14",
        "B3GALT1",
        "B3GALT2",
        "B3GALT4",
        "B3GALT5",
        "B3GALT6",
        "B4GALT1",
        "B4GALT2",
        "B4GALT3",
        "B4GALT4",
        "B4GALT5",
        "B4GALT6",
        "MGAT1",
        "MGAT2",
        "MGAT3",
        "MGAT4A",
        "MGAT4B",
        "MGAT5",
        "MGAT5B",
        "FUT1",
        "FUT2",
        "FUT3",
        "FUT4",
        "FUT5",
        "FUT6",
        "FUT7",
        "FUT8",
        "FUT9",
        "A3GALT2",
        "B3GNT2",
        "B3GNT3",
        "B3GNT7",
        "B3GNT8",
        "GCNT1",
        "GCNT2",
        "GCNT3",
        "GCNT4",
        "C1GALT1",
        "C1GALT1C1",
        "NEU1",
        "NEU2",
        "NEU3",
        "NEU4",
        "FUCA1",
        "FUCA2",
        "HEXA",
        "HEXB",
        "SLC35A1",
        "SLC35A2",
        "SLC35B1",
        "SLC35C1",
        "SLC35D1",
    ]

    trafficking_genes = [
        "VCAM1",
        "ICAM1",
        "ICAM2",
        "ICAM3",
        "MADCAM1",
        "ALCAM",
        "PECAM1",
        "CD34",
        "SELE",
        "SELP",
        "SELL",
        "CCL19",
        "CCL21",
        "CCL2",
        "CCL5",
        "CXCL9",
        "CXCL10",
        "CXCL11",
        "CXCL12",
        "CXCL13",
        "CX3CL1",
        "CCR7",
        "CXCR4",
        "CXCR5",
        "ACKR2",
        "ACKR4",
        "LYVE1",
        "PDPN",
        "FLT4",
        "NRP2",
        "VEGFC",
        "VEGFD",
        "CD274",
        "PDCD1LG2",
        "CD80",
        "CD86",
        "CTLA4",
        "PLPP3",
        "SPHK1",
        "S1PR1",
        "S1PR2",
        "F3",
        "TNFRSF1A",
        "TNFRSF1B",
        "CD53",
        "CD44",
        "ITGA4",
        "ITGAL",
        "ITGAM",
        "ITGB1",
        "ITGB2",
        "ITGB7",
    ]

    rows = []
    for gene in tf_list:
        rows.append({"symbol": gene, "is_tf": True, "category": "tf"})
    for gene in glyco_genes:
        rows.append({"symbol": gene, "is_tf": False, "category": "glyco"})
    for gene in trafficking_genes:
        rows.append({"symbol": gene, "is_tf": False, "category": "trafficking"})

    df = pd.DataFrame(rows).drop_duplicates(subset=["symbol"]).sort_values("symbol")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    print("Created comprehensive gene list")
    print(f"  Total genes: {df['symbol'].nunique()}")
    print(f"  TFs: {df['is_tf'].sum()}")
    print(f"  Glyco enzymes: {(df['category'] == 'glyco').sum()}")
    print(f"  Trafficking genes: {(df['category'] == 'trafficking').sum()}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()
