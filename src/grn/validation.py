import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

logger = logging.getLogger(__name__)


def _bh_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg correction for multiple testing."""
    n = len(pvalues)
    if n == 0:
        return np.array([])

    order = np.argsort(pvalues)
    ranked = np.empty(n, dtype=float)
    cumulative = 1.0

    for i in range(n - 1, -1, -1):
        rank = i + 1
        value = pvalues[order[i]] * n / rank
        cumulative = min(cumulative, value)
        ranked[order[i]] = cumulative

    return np.minimum(ranked, 1.0)


class GRNValidator:
    """
    Validate inferred GRN against external databases and literature.
    """

    def __init__(self, grn: pd.DataFrame, background_genes: Optional[Iterable[str]] = None) -> None:
        """
        Args:
            grn: GRN DataFrame with columns [TF, target, importance]
            background_genes: Optional background gene universe
        """
        self.grn = grn
        if background_genes is None:
            genes = set(grn["TF"]).union(set(grn["target"]))
        else:
            genes = set(background_genes)
        self.background_genes = genes

    def load_chipseq_database(self) -> Dict[str, List[str]]:
        """
        Load known TF-target relationships from ChIP-seq experiments.

        Returns:
            Dictionary: {TF: [target1, target2, ...]}
        """
        logger.info("Loading ChIP-seq validation database")

        chipseq_db = {
            "Prox1": ["Lyve1", "Flt4", "Pdpn", "Vegfc", "Nrp2"],
            "Sox18": ["Prox1", "Vegfr3", "Ccl21a"],
            "Nfkb1": ["Ccl21a", "Vcam1", "Icam1", "Cd274"],
            "Foxc2": ["Gja4", "Cx37", "Nrg1"],
            "Gata2": ["Dll4", "Hey1", "Hes1"],
        }

        logger.info(
            "  Loaded %s TFs with %s known targets",
            len(chipseq_db),
            sum(len(values) for values in chipseq_db.values()),
        )

        return chipseq_db

    def compute_chipseq_overlap(self, chipseq_db: Dict[str, List[str]]) -> pd.DataFrame:
        """
        Compute overlap between predicted and known TF-target edges.

        Returns:
            DataFrame with validation statistics per TF
        """
        logger.info("Computing ChIP-seq validation overlap")

        validation_stats = []

        for tf, known_targets in chipseq_db.items():
            predicted = self.grn[self.grn["TF"] == tf]["target"].tolist()

            if not predicted:
                continue

            overlap = set(predicted) & set(known_targets)
            precision = len(overlap) / len(predicted) if predicted else 0
            recall = len(overlap) / len(known_targets) if known_targets else 0

            validation_stats.append(
                {
                    "TF": tf,
                    "predicted_targets": len(predicted),
                    "known_targets": len(known_targets),
                    "overlap": len(overlap),
                    "precision": precision,
                    "recall": recall,
                    "validated_edges": ";".join(sorted(overlap)),
                }
            )

        stats_df = pd.DataFrame(validation_stats)

        if not stats_df.empty:
            logger.info("  Overall precision: %.2f%%", stats_df["precision"].mean() * 100)
            logger.info("  Overall recall: %.2f%%", stats_df["recall"].mean() * 100)

        return stats_df

    def load_go_library(self, go_path: Optional[str] = None, organism: str = "Mouse") -> Dict[str, List[str]]:
        """
        Load GO pathway gene sets from local file or gseapy.

        Args:
            go_path: Optional path to local GO pathway CSV
            organism: Organism name for gseapy

        Returns:
            Dictionary of pathway name to gene list
        """
        if go_path:
            path = Path(go_path)
            if path.exists():
                df = pd.read_csv(path)
                if "pathway" in df.columns and "gene" in df.columns:
                    return df.groupby("pathway")["gene"].apply(lambda x: sorted(set(x))).to_dict()

        try:
            import gseapy as gp
        except ImportError as exc:
            raise ImportError("gseapy is required for GO enrichment when no local file is provided.") from exc

        return gp.get_library("GO_Biological_Process_2023", organism=organism)

    def pathway_enrichment_per_tf(
        self,
        go_path: Optional[str] = "data/external/go_pathways_trafficking.csv",
        organism: str = "Mouse",
    ) -> pd.DataFrame:
        """
        Test if each TF's targets are enriched in specific pathways.

        Returns:
            DataFrame with enrichment results
        """
        logger.info("Running pathway enrichment analysis for TF regulons")

        pathways = self.load_go_library(go_path=go_path, organism=organism)
        background_genes = set(self.background_genes)
        for genes in pathways.values():
            background_genes.update(genes)
        results = []

        for tf in self.grn["TF"].unique():
            targets = set(self.grn[self.grn["TF"] == tf]["target"].tolist())
            if len(targets) < 5:
                continue

            for pathway, genes in pathways.items():
                pathway_genes = set(genes) & background_genes
                if not pathway_genes:
                    continue

                a = len(targets & pathway_genes)
                b = len(targets - pathway_genes)
                c = len(pathway_genes - targets)
                d = len(background_genes - targets - pathway_genes)

                if a == 0:
                    continue

                _, p_value = fisher_exact([[a, b], [c, d]], alternative="greater")
                results.append(
                    {
                        "TF": tf,
                        "pathway": pathway,
                        "p_value": p_value,
                        "targets_in_pathway": a,
                        "n_targets": len(targets),
                    }
                )

        enrichment_df = pd.DataFrame(results)
        if enrichment_df.empty:
            logger.warning("No enrichment results computed")
            return enrichment_df

        enrichment_df["p_value_adj"] = _bh_correction(enrichment_df["p_value"].values)
        enrichment_df = enrichment_df.sort_values("p_value_adj")

        logger.info("  Found %s enriched TF-pathway pairs", len(enrichment_df))
        return enrichment_df

    def literature_matches(self) -> pd.DataFrame:
        """
        Compare GRN against curated LEC literature edges.

        Returns:
            DataFrame of known edges and whether they are present
        """
        known_edges = [
            ("Prox1", "Lyve1"),
            ("Prox1", "Flt4"),
            ("Foxc2", "Gja4"),
            ("Sox18", "Prox1"),
            ("Nfkb1", "Ccl21a"),
        ]
        predicted = set(zip(self.grn["TF"], self.grn["target"]))
        rows = []
        for tf, target in known_edges:
            rows.append({"TF": tf, "target": target, "inferred": (tf, target) in predicted})
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Validate GRN")
    parser.add_argument("--grn", required=True, help="Path to GRN CSV file")
    parser.add_argument("--output", required=True, help="Output path for validation report")
    parser.add_argument("--go_path", default="data/external/go_pathways_trafficking.csv", help="Path to GO genes")

    args = parser.parse_args()

    grn_df = pd.read_csv(args.grn)

    validator = GRNValidator(grn_df)

    chipseq_db = validator.load_chipseq_database()
    overlap_stats = validator.compute_chipseq_overlap(chipseq_db)
    enrichment = validator.pathway_enrichment_per_tf(go_path=args.go_path)
    literature = validator.literature_matches()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_path) as writer:
        overlap_stats.to_excel(writer, sheet_name="ChIPseq_Validation", index=False)
        enrichment.to_excel(writer, sheet_name="Pathway_Enrichment", index=False)
        literature.to_excel(writer, sheet_name="Literature", index=False)

    print(f"\nValidation complete. Report saved to {output_path}")
