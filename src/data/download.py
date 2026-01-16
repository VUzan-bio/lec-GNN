import logging
import re
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import requests
import yaml

try:
    import GEOparse
except ImportError as exc:  # pragma: no cover - optional at runtime
    GEOparse = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class DatasetDownloader:
    """
    Unified downloader for LEC scRNA-seq datasets and gene annotations.

    Attributes:
        config_path: Path to data_sources.yaml
        output_dir: Directory to save downloaded files
    """

    def __init__(self, config_path: str = "config/data_sources.yaml", output_dir: str = "data/raw"):
        """
        Initialize downloader with configuration.

        Args:
            config_path: Path to YAML config with dataset accessions
            output_dir: Base directory for raw data storage
        """
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Configure logging to file and stdout."""
        root_logger = logging.getLogger()
        if not root_logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                handlers=[
                    logging.FileHandler("data_download.log"),
                    logging.StreamHandler(),
                ],
            )
            return

        has_file = any(
            isinstance(handler, logging.FileHandler)
            and handler.baseFilename.endswith("data_download.log")
            for handler in root_logger.handlers
        )
        if not has_file:
            file_handler = logging.FileHandler("data_download.log")
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            )
            root_logger.addHandler(file_handler)

    def _load_config(self, config_path: str) -> Dict:
        """Load dataset configuration from YAML."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle)

    def _download_file(self, url: str, destination: Path, timeout: int = 60) -> None:
        """
        Stream download a file from a URL.

        Args:
            url: Remote file URL
            destination: Local file path to write
            timeout: Timeout in seconds
        """
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

    def _looks_like_html(self, path: Path) -> bool:
        """Check if a downloaded file looks like an HTML error page."""
        try:
            with path.open("rb") as handle:
                snippet = handle.read(2048).lower()
        except OSError:
            return False
        return b"<html" in snippet or b"<!doctype html" in snippet

    def _extract_file_names(self, payload: object) -> List[str]:
        """Recursively collect file-like names from a BioStudies response."""
        names: List[str] = []

        def _walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_lower = key.lower()
                    if key_lower in {"filename", "file_name", "name"} and isinstance(value, str):
                        if "." in value:
                            names.append(value)
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return sorted(set(names))

    def _fetch_biostudies_file_list(self, accession: str) -> List[str]:
        """Fetch file list from BioStudies API or HTML listing."""
        api_urls = [
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}?format=json",
        ]
        for url in api_urls:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue
            file_names = self._extract_file_names(payload)
            if file_names:
                return file_names

        listing_url = f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}/files"
        try:
            response = requests.get(listing_url, timeout=60)
            response.raise_for_status()
        except Exception:
            return []

        matches = re.findall(r'href="([^"]+)"', response.text)
        file_names = []
        for match in matches:
            if accession in match:
                clean = match.split("?", 1)[0]
                file_names.append(Path(clean).name)
        return sorted(set(file_names))

    def _choose_arrayexpress_files(self, accession: str, file_names: Iterable[str]) -> List[str]:
        """Select processed counts and SDRF files from available names."""
        names = list(file_names)
        sdrf_candidates = [name for name in names if name.lower().endswith(".sdrf.txt")]
        processed_candidates = [name for name in names if "processed" in name.lower()]

        selected: List[str] = []
        if sdrf_candidates:
            selected.append(sdrf_candidates[0])
        else:
            selected.append(f"{accession}.sdrf.txt")

        if processed_candidates:
            zip_candidates = [name for name in processed_candidates if name.lower().endswith(".zip")]
            selected.append(zip_candidates[0] if zip_candidates else processed_candidates[0])
        else:
            selected.append(f"{accession}.processed.1.zip")

        return selected

    def _get_tf_urls(self) -> List[str]:
        """Collect TF database URLs from config with fallbacks."""
        tf_config = self.config.get("gene_databases", {}).get("transcription_factors", {})
        urls: List[str] = []
        if isinstance(tf_config.get("urls"), list):
            urls.extend([str(url) for url in tf_config["urls"]])
        if isinstance(tf_config.get("url"), str):
            urls.append(tf_config["url"])
        fallback_urls = [
            "http://bioinfo.life.hust.edu.cn/AnimalTFDB/download/TF/Mus_musculus_TF",
            "http://bioinfo.life.hust.edu.cn/AnimalTFDB/download/TF/Mouse_TF",
            "https://www.bioguo.org/AnimalTFDB/download/Mus_musculus_TF",
        ]
        urls.extend(fallback_urls)
        seen = set()
        unique_urls = []
        for url in urls:
            if url and url not in seen:
                unique_urls.append(url)
                seen.add(url)
        return unique_urls

    def download_arrayexpress(self, accession: str) -> Path:
        """
        Download dataset from ArrayExpress (BioStudies interface).

        Args:
            accession: ArrayExpress ID (e.g., "E-MTAB-8414")

        Returns:
            Path to downloaded directory

        Raises:
            requests.HTTPError: If download fails
        """
        logger.info("Downloading ArrayExpress dataset %s", accession)

        base_url = f"https://www.ebi.ac.uk/biostudies/files/{accession}"

        output_path = self.output_dir / accession
        output_path.mkdir(exist_ok=True)

        file_names = self._fetch_biostudies_file_list(accession)
        if file_names:
            logger.info("  Found %s files in BioStudies listing", len(file_names))
        else:
            logger.warning("  No BioStudies file listing found, using default filenames")
        files_to_download = self._choose_arrayexpress_files(accession, file_names)

        for filename in files_to_download:
            file_path = filename.lstrip("/")
            file_url = f"{base_url}/{file_path}"
            local_path = output_path / Path(file_path).name

            if local_path.exists():
                logger.info("  %s already exists, skipping", filename)
                continue

            logger.info("  Downloading %s...", filename)
            try:
                self._download_file(file_url, local_path)
            except Exception as exc:
                logger.error("  Failed to download %s: %s", filename, exc)
                continue

            if self._looks_like_html(local_path):
                logger.error("  %s looks like an HTML error page. Check file listing.", filename)
                local_path.unlink(missing_ok=True)
                continue

        import zipfile

        for zipfile_path in output_path.glob("*.zip"):
            if not zipfile.is_zipfile(zipfile_path):
                logger.warning("  Skipping %s: not a valid zip file", zipfile_path.name)
                continue
            logger.info("  Extracting %s", zipfile_path.name)
            with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
                zip_ref.extractall(output_path)

        logger.info("Downloaded %s to %s", accession, output_path)
        return output_path

    def _save_geo_metadata(self, gse: "GEOparse.GEOTypes.GSE", output_dir: Path) -> Path:
        """
        Save sample metadata from a GEO GSE into a CSV file.

        Args:
            gse: GEOparse GSE object
            output_dir: Directory where metadata will be saved

        Returns:
            Path to saved metadata CSV
        """
        records: List[Dict[str, str]] = []
        for gsm_name, gsm in gse.gsms.items():
            record: Dict[str, str] = {"gsm": gsm_name}
            for key, value in gsm.metadata.items():
                if isinstance(value, list):
                    record[key] = ";".join(value)
                else:
                    record[key] = str(value)
            records.append(record)

        metadata_df = pd.DataFrame(records)
        output_path = output_dir / "sample_metadata.csv"
        metadata_df.to_csv(output_path, index=False)
        return output_path

    def download_geo(self, accession: str, include_suppl: bool = True) -> Tuple["GEOparse.GEOTypes.GSE", Path]:
        """
        Download GEO dataset using GEOparse.

        Args:
            accession: GEO Series ID (e.g., "GSE148730")
            include_suppl: Download supplementary files (counts matrices)

        Returns:
            (GEOparse GSE object, Path to download directory)
        """
        if GEOparse is None:
            raise ImportError("GEOparse is not installed. Install it to download GEO datasets.")
        assert GEOparse is not None

        logger.info("Downloading GEO dataset %s", accession)

        output_path = self.output_dir / accession
        output_path.mkdir(exist_ok=True)

        gse = GEOparse.get_GEO(
            geo=accession,
            destdir=str(output_path),
            silent=False,
            include_data=include_suppl,
        )

        logger.info("  Platform: %s", gse.metadata.get("platform_id"))
        logger.info("  Samples: %s", len(gse.gsms))

        metadata_path = self._save_geo_metadata(gse, output_path)
        logger.info("  Saved sample metadata to %s", metadata_path)

        if include_suppl:
            logger.info("  Downloading supplementary files...")
            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, "download_supplementary_files"):
                    gsm.download_supplementary_files(
                        directory=str(output_path / "supplementary"),
                        download_sra=False,
                    )
                else:
                    logger.warning("  GSM %s has no supplementary files", gsm_name)

        logger.info("Downloaded %s to %s", accession, output_path)
        return gse, output_path

    def download_gene_databases(self) -> Dict[str, pd.DataFrame]:
        """
        Download curated gene annotation databases.

        Returns:
            Dictionary of {database_name: DataFrame}
        """
        logger.info("Downloading gene annotation databases")

        db_dir = self.output_dir.parent / "external"
        db_dir.mkdir(exist_ok=True)

        databases: Dict[str, pd.DataFrame] = {}

        tf_path = db_dir / "mouse_tfs_animalTFDB.csv"

        if not tf_path.exists():
            logger.info("  Downloading mouse TF database...")
            tf_df = None
            tf_urls = self._get_tf_urls()
            for tf_url in tf_urls:
                try:
                    response = requests.get(tf_url, timeout=60)
                    response.raise_for_status()
                    if "<html" in response.text.lower():
                        raise ValueError("HTML response received")
                    tf_df = pd.read_csv(
                        StringIO(response.text),
                        sep="\t",
                        header=None,
                        names=["ensembl", "symbol", "family", "dbds"],
                    )
                    break
                except Exception as exc:
                    logger.warning("  TF download failed from %s: %s", tf_url, exc)
                    tf_df = None

            if tf_df is None:
                raise RuntimeError("Unable to download TF database from configured URLs.")

            tf_df.to_csv(tf_path, index=False)
        else:
            tf_df = pd.read_csv(tf_path)

        databases["transcription_factors"] = tf_df
        logger.info("  Loaded %s mouse TFs", len(tf_df))

        try:
            import gseapy as gp
        except ImportError as exc:  # pragma: no cover - optional at runtime
            raise ImportError("gseapy is not installed. Install it to download GO pathways.") from exc

        go_path = db_dir / "go_pathways_trafficking.csv"

        if not go_path.exists():
            logger.info("  Downloading GO pathway gene sets...")
            go_bp = gp.get_library("GO_Biological_Process_2023", organism="Mouse")

            pathways_of_interest = {
                "migration": "GO:0030334",
                "lymphangiogenesis": "GO:0001946",
                "glycosylation": "GO:0006486",
                "chemotaxis": "GO:0060326",
                "angiogenesis": "GO:0001525",
            }

            pathway_genes: List[Dict[str, str]] = []
            for name, go_id in pathways_of_interest.items():
                matched_sets = {
                    k: v
                    for k, v in go_bp.items()
                    if go_id in k or name.replace("_", " ").lower() in k.lower()
                }
                if not matched_sets:
                    logger.warning("  No GO sets matched for %s (%s)", name, go_id)
                for set_name, genes in matched_sets.items():
                    for gene in genes:
                        pathway_genes.append(
                            {"pathway": name, "go_id": go_id, "gene": gene, "source_set": set_name}
                        )

            go_df = pd.DataFrame(pathway_genes, columns=["pathway", "go_id", "gene", "source_set"])
            go_df.to_csv(go_path, index=False)
        else:
            go_df = pd.read_csv(go_path)

        databases["go_pathways"] = go_df
        logger.info("  Loaded %s unique pathway genes", go_df["gene"].nunique())

        glyco_path = db_dir / "glycosylation_genes_ggdb.csv"
        if glyco_path.exists():
            glyco_df = pd.read_csv(glyco_path)
        else:
            logger.info("  Using manual glycosylation gene list (fallback)")
            glyco_genes = [
                "Galnt1",
                "Galnt2",
                "Galnt3",
                "Galnt4",
                "Galnt6",
                "St3gal1",
                "St3gal4",
                "St3gal6",
                "Gcnt1",
                "Gcnt3",
                "B3gnt7",
                "Fut4",
                "Fut9",
                "C1galt1",
                "Mgat5",
            ]
            glyco_df = pd.DataFrame({"gene": glyco_genes, "category": "O-glycosylation"})
            glyco_df.to_csv(glyco_path, index=False)

        databases["glycosylation"] = glyco_df
        logger.info("  Loaded %s glycosylation genes", len(glyco_df))

        return databases

    def download_all(self) -> Dict[str, Path]:
        """
        Download all datasets specified in config file.

        Returns:
            Dictionary mapping dataset names to local paths
        """
        logger.info("=" * 60)
        logger.info("Starting complete dataset download pipeline")
        logger.info("=" * 60)

        downloaded_paths: Dict[str, Path] = {}

        for dataset in self.config.get("arrayexpress", []):
            accession = dataset["accession"]
            try:
                path = self.download_arrayexpress(accession)
                downloaded_paths[accession] = path
            except Exception as exc:  # pragma: no cover - network
                logger.error("Failed to download %s: %s", accession, exc)

        for dataset in self.config.get("geo", []):
            accession = dataset["accession"]
            try:
                _, path = self.download_geo(accession)
                downloaded_paths[accession] = path
            except Exception as exc:  # pragma: no cover - network
                logger.error("Failed to download %s: %s", accession, exc)

        try:
            self.download_gene_databases()
            downloaded_paths["gene_databases"] = self.output_dir.parent / "external"
        except Exception as exc:  # pragma: no cover - network
            logger.error("Failed to download gene databases: %s", exc)

        logger.info("=" * 60)
        logger.info("Download complete. %s datasets acquired", len(downloaded_paths))
        logger.info("=" * 60)

        return downloaded_paths


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download LEC scRNA-seq datasets")
    parser.add_argument("--config", default="config/data_sources.yaml", help="Path to config file")
    parser.add_argument("--output", default="data/raw", help="Output directory")

    args = parser.parse_args()

    downloader = DatasetDownloader(config_path=args.config, output_dir=args.output)
    paths = downloader.download_all()

    print("\nDownloaded datasets:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
