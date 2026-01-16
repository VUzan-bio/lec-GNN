import logging
import re
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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

    def _normalize_geo_url(self, url: str) -> str:
        """Normalize GEO supplementary URLs for HTTP downloads."""
        if url.startswith("ftp://"):
            return "https://" + url[len("ftp://") :]
        return url

    def _download_geo_supplementary_urls(self, urls: Iterable[str], output_dir: Path) -> None:
        """Download supplementary files listed at the GEO series level."""
        for url in urls:
            if not url:
                continue
            normalized = self._normalize_geo_url(url)
            filename = Path(normalized).name
            destination = output_dir / filename
            if destination.exists() and self._is_valid_download(destination):
                logger.info("  %s already exists, skipping", destination.name)
                continue
            if destination.exists():
                destination.unlink(missing_ok=True)
            try:
                logger.info("  Downloading %s...", filename)
                self._download_file(normalized, destination, timeout=120)
            except Exception as exc:
                logger.warning("  Failed to download %s: %s", normalized, exc)

    def _download_geo_series_fallback(self, accession: str, output_dir: Path) -> None:
        """Try common GEO series supplementary files when metadata is empty."""
        series_root = f"{accession[:-3]}nnn"
        base_url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{series_root}/{accession}/suppl"
        candidates = [
            f"{base_url}/{accession}_RAW.tar",
            f"{base_url}/{accession}_RAW.tar.gz",
            f"{base_url}/{accession}_series_matrix.txt.gz",
            f"{base_url}/{accession}_series_matrix.txt",
        ]

        for url in candidates:
            filename = Path(url).name
            destination = output_dir / filename
            if destination.exists() and self._is_valid_download(destination):
                continue
            try:
                logger.info("  Attempting GEO series download: %s", filename)
                self._download_file(url, destination, timeout=120)
            except Exception as exc:
                logger.info("  GEO series file not available: %s (%s)", filename, exc)
    def _extract_archives(self, directory: Path) -> None:
        """Extract common archives (zip, tar, tar.gz) in a directory."""
        import tarfile
        import zipfile

        for archive in directory.rglob("*"):
            if not archive.is_file():
                continue
            name = archive.name.lower()
            if name.endswith(".zip"):
                if not zipfile.is_zipfile(archive):
                    continue
                logger.info("  Extracting %s", archive.name)
                with zipfile.ZipFile(archive, "r") as zip_ref:
                    zip_ref.extractall(directory)
            elif name.endswith((".tar", ".tar.gz", ".tgz")):
                try:
                    with tarfile.open(archive, "r:*") as tar_ref:
                        logger.info("  Extracting %s", archive.name)
                        tar_ref.extractall(directory)
                except tarfile.TarError as exc:
                    logger.warning("  Failed to extract %s: %s", archive.name, exc)

    def _is_valid_download(self, path: Path) -> bool:
        """Validate downloaded file is non-empty and not HTML."""
        if not path.exists():
            return False
        if path.stat().st_size == 0:
            return False
        if self._looks_like_html(path):
            return False
        return True

    def _looks_like_html(self, path: Path) -> bool:
        """Check if a downloaded file looks like an HTML error page."""
        try:
            with path.open("rb") as handle:
                snippet = handle.read(2048).lower()
        except OSError:
            return False
        return b"<html" in snippet or b"<!doctype html" in snippet

    def _looks_like_filename(self, value: str) -> bool:
        """Check if a value looks like a data filename (no spaces, has extension)."""
        if not value or " " in value:
            return False
        name = Path(value).name
        if "." not in name:
            return False
        allowed_suffixes = (
            ".zip",
            ".txt",
            ".txt.gz",
            ".tsv",
            ".tsv.gz",
            ".csv",
            ".csv.gz",
            ".mtx",
            ".mtx.gz",
            ".h5",
            ".h5ad",
        )
        lower = name.lower()
        return lower.endswith(allowed_suffixes)

    def _strip_url_artifacts(self, value: str) -> str:
        """Clean query strings and fragments from a URL-like string."""
        return value.split("?", 1)[0].split("#", 1)[0]

    def _filter_biostudies_files(self, accession: str, names: List[str]) -> List[str]:
        """Filter BioStudies listings to likely data files."""
        if not names:
            return []
        cleaned = [self._strip_url_artifacts(name) for name in names]
        filtered = [name for name in cleaned if self._looks_like_filename(name)]
        accession_matches = [name for name in filtered if accession in name]
        if accession_matches:
            return sorted(set(accession_matches))
        return sorted(set(filtered))

    def _extract_files_from_html(self, accession: str, html_text: str) -> List[str]:
        """Extract file links from HTML pages."""
        candidates: List[str] = []
        hrefs = re.findall(r'href="([^"]+)"', html_text)
        data_attrs = re.findall(r'data-file(?:name|path)="([^"]+)"', html_text)

        for value in hrefs + data_attrs:
            if accession in value or self._looks_like_filename(value):
                candidates.append(value)

        pattern = rf"(?:https?://[^\s\"']+|/biostudies/files/[^\s\"']+|/arrayexpress/files/[^\s\"']+|{re.escape(accession)}[^\s\"']+)"
        for match in re.findall(pattern, html_text):
            if self._looks_like_filename(match):
                candidates.append(match)

        return self._filter_biostudies_files(accession, candidates)

    def _extract_xml_filenames(self, xml_text: str) -> List[str]:
        """Extract filename-like strings from XML payloads."""
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError:
            return []
        names: List[str] = []
        for elem in root.iter():
            if elem.text:
                value = elem.text.strip()
                if self._looks_like_filename(value):
                    names.append(value)
        return sorted(set(names))

    def _extract_file_names(self, payload: object) -> List[str]:
        """Recursively collect file-like names from a BioStudies response."""
        names: List[str] = []

        def _walk(node: object) -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    key_lower = key.lower()
                    if key_lower in {"filename", "file_name", "name", "url", "uri", "path"} and isinstance(
                        value, str
                    ):
                        if self._looks_like_filename(value):
                            names.append(value)
                    else:
                        _walk(value)
            elif isinstance(node, list):
                for item in node:
                    _walk(item)

        _walk(payload)
        return sorted(set(names))

    def _fetch_arrayexpress_legacy_file_list(self, accession: str) -> List[str]:
        """Fetch file list using legacy ArrayExpress endpoints."""
        json_url = f"https://www.ebi.ac.uk/arrayexpress/json/v3/files/{accession}"
        try:
            response = requests.get(json_url, timeout=60)
            response.raise_for_status()
            payload = response.json()
            file_names = self._extract_file_names(payload)
            file_names = self._filter_biostudies_files(accession, file_names)
            if file_names:
                return file_names
        except Exception:
            pass

        xml_url = f"https://www.ebi.ac.uk/arrayexpress/xml/v3/files/{accession}"
        try:
            response = requests.get(xml_url, timeout=60)
            response.raise_for_status()
            file_names = self._extract_xml_filenames(response.text)
            file_names = self._filter_biostudies_files(accession, file_names)
            if file_names:
                return file_names
        except Exception:
            pass

        listing_url = f"https://www.ebi.ac.uk/arrayexpress/files/{accession}/"
        try:
            response = requests.get(listing_url, timeout=60)
            response.raise_for_status()
            matches = re.findall(r'href="([^"]+)"', response.text)
            file_names = [Path(match.split("?", 1)[0]).name for match in matches]
            file_names = self._filter_biostudies_files(accession, file_names)
            if file_names:
                return file_names
        except Exception:
            pass

        return []

    def _normalize_explicit_files(self, files_override: object) -> List[Dict[str, str]]:
        """Normalize explicit file overrides from config."""
        if not isinstance(files_override, list):
            return []

        normalized: List[Dict[str, str]] = []
        for entry in files_override:
            if isinstance(entry, str):
                name = Path(entry).name
                normalized.append({"name": name, "path": entry})
                continue
            if isinstance(entry, dict):
                url = entry.get("url")
                path = entry.get("path")
                name = entry.get("name") or (Path(url).name if url else None) or (Path(path).name if path else None)
                if name:
                    record: Dict[str, str] = {"name": name}
                    if url:
                        record["url"] = str(url)
                    if path:
                        record["path"] = str(path)
                    normalized.append(record)
        return normalized

    def _download_explicit_files(self, entries: List[Dict[str, str]], output_path: Path, accession: str) -> None:
        """Download explicitly configured files."""
        base_url = f"https://www.ebi.ac.uk/biostudies/files/{accession}"
        for entry in entries:
            name = entry["name"]
            local_path = output_path / name

            if local_path.exists() and self._is_valid_download(local_path):
                logger.info("  %s already exists, skipping", local_path.name)
                continue
            if local_path.exists():
                logger.warning("  Removing invalid file %s", local_path.name)
                local_path.unlink(missing_ok=True)

            urls: List[str] = []
            if "url" in entry:
                urls.append(entry["url"])
            if "path" in entry:
                urls.extend(self._build_arrayexpress_urls(base_url, accession, entry["path"]))
            if not urls:
                urls.extend(self._build_arrayexpress_urls(base_url, accession, name))

            logger.info("  Downloading %s (explicit)...", name)
            if not self._download_with_fallback(urls, local_path):
                logger.error("  Failed to download %s from explicit URLs", name)

    def _write_fastq_manifest(self, sdrf_path: Path, output_path: Path) -> Optional[Path]:
        """Extract FASTQ URIs from SDRF into a manifest file."""
        if not sdrf_path.exists():
            return None

        df = pd.read_csv(sdrf_path, sep="\t")
        fastq_col = None
        for col in df.columns:
            if col.strip().lower() == "comment[fastq_uri]":
                fastq_col = col
                break
        if fastq_col is None:
            return None

        manifest_cols = [fastq_col]
        for optional in ["Source Name", "Comment[ENA_RUN]", "Comment[ENA_EXPERIMENT]", "Comment[SUBMITTED_FILE_NAME]"]:
            if optional in df.columns:
                manifest_cols.insert(0, optional)

        manifest = df[manifest_cols].copy()
        manifest = manifest[manifest[fastq_col].notna()]
        if manifest.empty:
            return None

        manifest_path = output_path / "fastq_manifest.tsv"
        manifest.to_csv(manifest_path, sep="\t", index=False)
        logger.info("  Wrote FASTQ manifest with %s entries to %s", len(manifest), manifest_path)
        return manifest_path

    def _build_arrayexpress_urls(self, base_url: str, accession: str, file_path: str) -> List[str]:
        """Generate candidate download URLs for ArrayExpress files."""
        if file_path.startswith(("http://", "https://", "ftp://")):
            return [file_path]

        name = Path(file_path).name
        mtab_accession = accession.replace("E-MTAB-", "MTAB-")
        ftp_bases = [
            f"https://ftp.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}",
            f"https://ftp.ebi.ac.uk/biostudies/arrayexpress/studies/{mtab_accession}",
            f"https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/MTAB/{accession}",
            f"https://ftp.ebi.ac.uk/pub/databases/arrayexpress/data/experiment/MTAB/{mtab_accession}",
        ]
        candidates = [
            f"{base_url}/{file_path}",
            f"{base_url}/{name}",
            f"{base_url}/{accession}/{name}",
            f"{base_url}/Files/{name}",
            f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}/files/{name}",
            f"https://www.ebi.ac.uk/arrayexpress/files/{accession}/{name}",
            f"https://www.ebi.ac.uk/arrayexpress/files/{mtab_accession}/{name}",
        ]
        for ftp_base in ftp_bases:
            candidates.append(f"{ftp_base}/{name}")
        seen = set()
        unique = []
        for url in candidates:
            if url not in seen:
                unique.append(url)
                seen.add(url)
        return unique

    def _download_with_fallback(self, urls: List[str], destination: Path) -> bool:
        """Try multiple URLs until one succeeds."""
        for url in urls:
            try:
                self._download_file(url, destination)
            except Exception as exc:
                logger.warning("  Download failed from %s: %s", url, exc)
                destination.unlink(missing_ok=True)
                continue

            if self._is_valid_download(destination):
                logger.info("  Downloaded from %s (%s bytes)", url, destination.stat().st_size)
                return True

            logger.warning("  Invalid download from %s", url)
            destination.unlink(missing_ok=True)

        return False

    def _fetch_biostudies_file_list(self, accession: str) -> List[str]:
        """Fetch file list from BioStudies API or HTML listing."""
        api_urls = [
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}?format=json",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/info?format=json",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/files",
            f"https://www.ebi.ac.uk/biostudies/api/v1/studies/{accession}/files?format=json",
        ]
        for url in api_urls:
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                payload = response.json()
            except Exception:
                continue
            file_names = self._extract_file_names(payload)
            file_names = self._filter_biostudies_files(accession, file_names)
            if file_names:
                return file_names

        listing_urls = [
            f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}/files",
            f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}/files?format=json",
            f"https://www.ebi.ac.uk/biostudies/arrayexpress/studies/{accession}/files?format=tsv",
        ]
        for listing_url in listing_urls:
            try:
                response = requests.get(listing_url, timeout=60)
                response.raise_for_status()
            except Exception:
                continue

            if listing_url.endswith("json") and response.headers.get("content-type", "").startswith("application/json"):
                try:
                    payload = response.json()
                except Exception:
                    payload = None
                if payload:
                    file_names = self._extract_file_names(payload)
                    file_names = self._filter_biostudies_files(accession, file_names)
                    if file_names:
                        return file_names

            if listing_url.endswith("tsv"):
                lines = [line.strip() for line in response.text.splitlines() if line.strip()]
                if lines and len(lines) > 1:
                    header = [col.lower() for col in lines[0].split("\t")]
                    if "filename" in header or "file_name" in header:
                        idx = header.index("filename") if "filename" in header else header.index("file_name")
                        names = [line.split("\t")[idx] for line in lines[1:] if "\t" in line]
                        names = self._filter_biostudies_files(accession, names)
                        if names:
                            return names

            matches = re.findall(r'href="([^"]+)"', response.text)
            file_names = []
            for match in matches:
                clean = match.split("?", 1)[0]
                name = Path(clean).name
                if name and self._looks_like_filename(name):
                    file_names.append(name)
            file_names = self._filter_biostudies_files(accession, file_names)
            if file_names:
                return file_names

            html_files = self._extract_files_from_html(accession, response.text)
            if html_files:
                return html_files

        return self._fetch_arrayexpress_legacy_file_list(accession)

    def _is_candidate_data_file(self, name: str) -> bool:
        """Heuristic filter for processed expression files."""
        lower = name.lower()
        if lower.endswith((".fastq", ".fastq.gz", ".fq", ".fq.gz", ".sra", ".bam", ".cram", ".fasta", ".fa")):
            return False
        if lower.endswith(".sdrf.txt"):
            return False
        return any(
            token in lower
            for token in ["processed", "normalized", "counts", "matrix", ".mtx", ".h5", ".h5ad", ".csv", ".tsv", ".txt", ".zip"]
        )

    def _choose_arrayexpress_files(self, accession: str, file_names: Iterable[str]) -> List[str]:
        """Select processed counts and SDRF files from available names."""
        names = list(file_names)
        sdrf_candidates = [name for name in names if name.lower().endswith(".sdrf.txt")]
        data_candidates = [name for name in names if self._is_candidate_data_file(name)]

        selected: List[str] = []
        if sdrf_candidates:
            selected.append(sdrf_candidates[0])
        else:
            selected.append(f"{accession}.sdrf.txt")

        if data_candidates:
            processed = [name for name in data_candidates if "processed" in name.lower()]
            if processed:
                selected.extend(sorted(processed))
            else:
                selected.extend(sorted(data_candidates))
        else:
            fallback_suffixes = [
                "processed.1.zip",
                "processed.2.zip",
                "processed.3.zip",
                "processed.zip",
                "processed.1.txt",
                "processed.1.txt.gz",
                "processed.1.tsv",
                "processed.1.tsv.gz",
                "processed.1.csv",
                "processed.1.csv.gz",
                "processed.txt",
                "processed.txt.gz",
                "processed.tsv",
                "processed.tsv.gz",
                "processed.csv",
                "processed.csv.gz",
            ]
            selected.extend([f"{accession}.{suffix}" for suffix in fallback_suffixes])

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

    def download_arrayexpress(self, accession: str, files_override: Optional[List[object]] = None) -> Path:
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
            listing_path = output_path / "biostudies_file_listing.txt"
            listing_path.write_text("\n".join(file_names), encoding="utf-8")
        else:
            logger.warning("  No BioStudies file listing found, using default filenames")
        explicit_files = self._normalize_explicit_files(files_override)
        if explicit_files:
            logger.info("  Using %s explicit file entries for %s", len(explicit_files), accession)
            self._download_explicit_files(explicit_files, output_path, accession)
            files_to_download = []
        else:
            files_to_download = self._choose_arrayexpress_files(accession, file_names)

        for filename in files_to_download:
            file_path = filename.lstrip("/")
            local_path = output_path / Path(file_path).name

            if local_path.exists() and self._is_valid_download(local_path):
                logger.info("  %s already exists, skipping", local_path.name)
                continue
            if local_path.exists():
                logger.warning("  Removing invalid file %s", local_path.name)
                local_path.unlink(missing_ok=True)

            logger.info("  Downloading %s...", filename)
            urls = self._build_arrayexpress_urls(base_url, accession, file_path)
            if not self._download_with_fallback(urls, local_path):
                logger.error("  Failed to download %s from any candidate URL", filename)
                continue

        import zipfile

        for zipfile_path in output_path.glob("*.zip"):
            if not zipfile.is_zipfile(zipfile_path):
                logger.warning("  Skipping %s: not a valid zip file", zipfile_path.name)
                continue
            logger.info("  Extracting %s", zipfile_path.name)
            with zipfile.ZipFile(zipfile_path, "r") as zip_ref:
                zip_ref.extractall(output_path)

        sdrf_path = output_path / f"{accession}.sdrf.txt"
        if sdrf_path.exists():
            self._write_fastq_manifest(sdrf_path, output_path)

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
            supp_dir = output_path / "supplementary"
            supp_dir.mkdir(exist_ok=True)

            gse_suppl = gse.metadata.get("supplementary_file", [])
            if isinstance(gse_suppl, str):
                gse_suppl = [gse_suppl]
            self._download_geo_supplementary_urls(gse_suppl, supp_dir)
            if not gse_suppl:
                self._download_geo_series_fallback(accession, supp_dir)

            for gsm_name, gsm in gse.gsms.items():
                if hasattr(gsm, "download_supplementary_files"):
                    gsm.download_supplementary_files(
                        directory=str(supp_dir),
                        download_sra=False,
                    )
                else:
                    logger.warning("  GSM %s has no supplementary files", gsm_name)

            self._extract_archives(supp_dir)

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

    def download_all(self, sources: Optional[List[str]] = None) -> Dict[str, Path]:
        """
        Download all datasets specified in config file.

        Args:
            sources: Optional list of sources to download (arrayexpress, geo, gene_databases).

        Returns:
            Dictionary mapping dataset names to local paths
        """
        source_set = {source.strip().lower() for source in sources} if sources else None

        logger.info("=" * 60)
        logger.info("Starting complete dataset download pipeline")
        logger.info("=" * 60)

        downloaded_paths: Dict[str, Path] = {}

        if source_set is None or "arrayexpress" in source_set:
            for dataset in self.config.get("arrayexpress", []):
                if isinstance(dataset, dict):
                    status = str(dataset.get("status", "")).lower()
                    if "deferred" in status:
                        logger.info("Skipping %s (status=%s)", dataset.get("accession"), dataset.get("status"))
                        continue
                accession = dataset["accession"]
                try:
                    files_override = dataset.get("files") if isinstance(dataset, dict) else None
                    path = self.download_arrayexpress(accession, files_override=files_override)
                    downloaded_paths[accession] = path
                except Exception as exc:  # pragma: no cover - network
                    logger.error("Failed to download %s: %s", accession, exc)

        if source_set is None or "geo" in source_set:
            for dataset in self.config.get("geo", []):
                if isinstance(dataset, dict):
                    status = str(dataset.get("status", "")).lower()
                    if "deferred" in status:
                        logger.info("Skipping %s (status=%s)", dataset.get("accession"), dataset.get("status"))
                        continue
                accession = dataset["accession"]
                try:
                    _, path = self.download_geo(accession)
                    downloaded_paths[accession] = path
                except Exception as exc:  # pragma: no cover - network
                    logger.error("Failed to download %s: %s", accession, exc)

        if source_set is None or "gene_databases" in source_set:
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
    parser.add_argument(
        "--source",
        default="all",
        help="Comma-separated sources to download (arrayexpress, geo, gene_databases, all)",
    )

    args = parser.parse_args()

    downloader = DatasetDownloader(config_path=args.config, output_dir=args.output)
    sources = None
    if args.source and args.source.lower() != "all":
        sources = [source.strip() for source in args.source.split(",") if source.strip()]
    paths = downloader.download_all(sources=sources)

    print("\nDownloaded datasets:")
    for name, path in paths.items():
        print(f"  {name}: {path}")
