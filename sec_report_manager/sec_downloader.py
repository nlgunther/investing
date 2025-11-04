"""
SEC Filing Downloader Module

Downloads SEC filings and returns either file paths or IO streams.
Supports multiple filing types through strategy pattern.
"""

import requests
import time
from pathlib import Path
from typing import List, Optional, Protocol, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum
from io import BytesIO
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


# ============================================================================
# SHARED TYPES & CONFIGURATION
# ============================================================================

class VerbosityLevel(IntEnum):
    """Verbosity levels for output control."""
    SILENT = 0
    ERROR = 1
    NORMAL = 2
    VERBOSE = 3
    DEBUG = 4


@dataclass
class DownloadConfig:
    """Configuration for SEC downloads."""
    user_agent: str = "Research Tool research@example.com"
    base_url: str = "https://www.sec.gov"
    data_url: str = "https://data.sec.gov"
    rate_limit_delay: float = 0.2
    request_delay: float = 0.1
    timeout: int = 30
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL


def _log(message: str, level: VerbosityLevel, config: DownloadConfig, prefix: str = ""):
    """Internal logging function."""
    if config.verbosity >= level:
        print(f"{prefix}{message}")


# ============================================================================
# EXCEPTIONS
# ============================================================================

class DownloadError(Exception):
    """Base exception for download operations."""
    pass


class FilingNotFoundError(DownloadError):
    """Raised when a filing cannot be found."""
    pass


# ============================================================================
# HTTP CLIENT PROTOCOL
# ============================================================================

class HTTPClient(Protocol):
    """Protocol for HTTP operations (enables testing without real requests)."""
    def get(self, url: str, timeout: int = 30) -> requests.Response: ...


class DefaultHTTPClient:
    """Default HTTP client using requests library."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.headers = {"User-Agent": config.user_agent}
    
    def get(self, url: str, timeout: Optional[int] = None) -> requests.Response:
        """Make HTTP GET request with configured headers."""
        response = requests.get(
            url, 
            headers=self.headers, 
            timeout=timeout or self.config.timeout
        )
        response.raise_for_status()
        return response


# ============================================================================
# FILING METADATA
# ============================================================================

@dataclass
class FilingMetadata:
    """Metadata about a single SEC filing."""
    cik: str
    accession_number: str
    filing_date: str
    form_type: str
    
    @property
    def accession_no_dash(self) -> str:
        """Accession number without dashes (used in URLs)."""
        return self.accession_number.replace("-", "")
    
    def archive_url(self, base_url: str) -> str:
        """Construct SEC archive URL for this filing."""
        return f"{base_url}/Archives/edgar/data/{int(self.cik)}/{self.accession_no_dash}"


@dataclass
class FilingResult:
    """Result of downloading a filing."""
    metadata: FilingMetadata
    files: Dict[str, BytesIO]  # filename -> stream
    local_path: Optional[Path] = None  # Set if saved locally


# ============================================================================
# FILE DISCOVERY
# ============================================================================

class FileDiscoverer:
    """Discovers XML files in SEC filings using multiple fallback methods."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        self.http = http
        self.config = config
    
    def discover(self, archive_url: str) -> List[str]:
        """Discover all XML files in a filing."""
        _log(f"Discovering files at {archive_url}", VerbosityLevel.DEBUG, self.config, "    ")
        
        # Try methods in order of preference
        files = (self._try_index_json(archive_url) or
                self._try_index_html(archive_url) or
                self._get_default_files())
        
        return files
    
    def _try_index_json(self, url: str) -> Optional[List[str]]:
        """Try to get file list from index.json (preferred method)."""
        try:
            data = self.http.get(f"{url}/index.json").json()
            files = [item['name'] for item in data['directory']['item'] 
                    if item['name'].endswith('.xml')]
            _log(f"Found {len(files)} XML files via index.json", 
                VerbosityLevel.DEBUG, self.config, "      ")
            return files
        except Exception:
            return None
    
    def _try_index_html(self, url: str) -> Optional[List[str]]:
        """Try to parse file list from index.html (fallback method)."""
        try:
            soup = BeautifulSoup(self.http.get(f"{url}/index.html").text, 'html.parser')
            files = [link.get('href', '').split('/')[-1] for link in soup.find_all('a')]
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                _log(f"Found {len(xml_files)} XML files via index.html", 
                    VerbosityLevel.DEBUG, self.config, "      ")
            return xml_files or None
        except Exception:
            return None
    
    def _get_default_files(self) -> List[str]:
        """Return common XML filenames as last resort."""
        return ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"]


# ============================================================================
# DOWNLOAD STRATEGIES
# ============================================================================

class DownloadStrategy(ABC):
    """Abstract download strategy defining interface for different filing types."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        self.http = http
        self.config = config
    
    @abstractmethod
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Determine if filing should be downloaded (e.g., series filtering)."""
        pass
    
    @abstractmethod
    def get_required_files(self, archive_url: str) -> List[str]:
        """Get list of required files for this filing type."""
        pass


class Strategy13F(DownloadStrategy):
    """Download strategy for 13-F filings (hedge fund holdings)."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        super().__init__(http, config)
        self.discoverer = FileDiscoverer(http, config)
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Download all 13-F filings (no filtering applied)."""
        return True
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """Discover all XML files (13-F has separate holdings file)."""
        return self.discoverer.discover(archive_url)


class StrategyNPORT(DownloadStrategy):
    """Download strategy for NPORT filings (mutual fund holdings) with series filtering."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig, series_id: Optional[str] = None):
        super().__init__(http, config)
        self.series_id = series_id
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Check if filing matches series ID filter."""
        if not self.series_id:
            return True
        
        _log(f"Checking series ID for {filing.accession_number}", 
            VerbosityLevel.DEBUG, self.config, "    ")
        
        try:
            # Fetch primary_doc.xml to check series ID
            response = self.http.get(f"{archive_url}/primary_doc.xml")
            root = ET.fromstring(response.content)
            
            # Remove namespaces for easier querying
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]
            
            series_elem = root.find('.//seriesId')
            if series_elem is not None and series_elem.text:
                matches = series_elem.text.strip() == self.series_id
                _log(f"Series ID {series_elem.text.strip()} {'matches' if matches else 'does not match'}", 
                    VerbosityLevel.DEBUG, self.config, "      ")
                return matches
            
            return False
            
        except Exception as e:
            _log(f"Warning: Could not check series ID: {e}", 
                VerbosityLevel.VERBOSE, self.config, "    ")
            return False
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """NPORT uses single file (primary_doc.xml contains everything)."""
        return ["primary_doc.xml"]


# ============================================================================
# DOWNLOADER
# ============================================================================

class SECDownloader:
    """Downloads SEC filings with optional local storage.
    
    Features:
    - Returns IO streams for immediate parsing
    - Optional local storage for caching
    - Pluggable strategies for different filing types
    - Automatic rate limiting
    """
    
    # Registry of supported form types
    STRATEGIES = {
        '13F-HR': Strategy13F,
        'NPORT-P': StrategyNPORT,
        'NPORT-N': StrategyNPORT,
    }
    
    def __init__(self, 
                 form_type: str, 
                 config: Optional[DownloadConfig] = None,
                 series_id: Optional[str] = None,
                 http_client: Optional[HTTPClient] = None):
        """Initialize downloader.
        
        Args:
            form_type: Form type (e.g., "13F-HR", "NPORT-P")
            config: Download configuration (uses defaults if None)
            series_id: Series ID for NPORT filings (optional)
            http_client: Custom HTTP client (uses default if None, mainly for testing)
        """
        if form_type not in self.STRATEGIES:
            raise ValueError(
                f"Unsupported form type: {form_type}. "
                f"Supported: {', '.join(self.STRATEGIES.keys())}"
            )
        
        self.form_type = form_type
        self.config = config or DownloadConfig()
        
        # Create HTTP client
        self.http = http_client or DefaultHTTPClient(self.config)
        
        # Create appropriate strategy
        strategy_cls = self.STRATEGIES[form_type]
        if form_type in ['NPORT-P', 'NPORT-N']:
            self.strategy = strategy_cls(self.http, self.config, series_id)
        else:
            self.strategy = strategy_cls(self.http, self.config)
    
    def download(self, 
                cik: str, 
                num_reports: int = 5, 
                save_to_disk: bool = False, 
                output_dir: str = "sec_filings") -> List[FilingResult]:
        """Download filings.
        
        Args:
            cik: Central Index Key (company identifier)
            num_reports: Maximum number of filings to download
            save_to_disk: If True, save files locally in addition to returning streams
            output_dir: Directory for local storage (if save_to_disk=True)
            
        Returns:
            List of FilingResult objects containing:
            - metadata: Filing metadata
            - files: Dict of filename -> BytesIO stream
            - local_path: Path if saved to disk, None otherwise
        """
        # Normalize CIK to 10 digits with leading zeros
        cik = cik.strip().replace('-', '').zfill(10)
        
        _log(f"Fetching {self.form_type} filings for CIK {cik}...", 
            VerbosityLevel.NORMAL, self.config)
        
        # Fetch list of available filings
        filings = self._fetch_filing_list(cik)
        if not filings:
            _log(f"No {self.form_type} filings found", VerbosityLevel.NORMAL, self.config)
            return []
        
        _log(f"Found {len(filings)} filing(s), applying filters...", 
            VerbosityLevel.NORMAL, self.config)
        
        # Prepare output directory if saving to disk
        output_path = Path(output_dir) if save_to_disk else None
        if output_path:
            output_path.mkdir(exist_ok=True)
        
        # Download filings that pass strategy's filter
        results = []
        for filing in filings:
            if len(results) >= num_reports:
                break
            
            archive_url = filing.archive_url(self.config.base_url)
            
            # Check if this filing should be downloaded
            if not self.strategy.should_download(filing, archive_url):
                _log(f"Skipping {filing.filing_date} (filtered out)", 
                    VerbosityLevel.VERBOSE, self.config, "  ")
                continue
            
            # Download the filing
            result = self._download_filing(filing, archive_url, output_path)
            if result:
                results.append(result)
            
            # Rate limiting
            time.sleep(self.config.rate_limit_delay)
        
        _log(f"Downloaded {len(results)} filing(s)", 
            VerbosityLevel.NORMAL, self.config, "\n✓ ")
        
        return results
    
    def _fetch_filing_list(self, cik: str) -> List[FilingMetadata]:
        """Fetch list of filings from SEC submissions API."""
        try:
            url = f"{self.config.data_url}/submissions/CIK{cik}.json"
            _log(f"Fetching filing list from {url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            data = self.http.get(url).json()
            filings_data = data.get("filings", {}).get("recent", {})
            
            # Filter to requested form type
            result = [
                FilingMetadata(
                    cik=cik,
                    accession_number=filings_data["accessionNumber"][i],
                    filing_date=filings_data["filingDate"][i],
                    form_type=self.form_type
                )
                for i, form in enumerate(filings_data.get("form", [])) 
                if form == self.form_type
            ]
            
            return result
            
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch filings for CIK {cik}: {e}")
    
    def _download_filing(self, 
                        filing: FilingMetadata, 
                        archive_url: str,
                        output_path: Optional[Path]) -> Optional[FilingResult]:
        """Download a single filing's files."""
        _log(f"{filing.filing_date} ({filing.accession_number})", 
            VerbosityLevel.NORMAL, self.config, "\n")
        _log(f"Archive URL: {archive_url}", VerbosityLevel.VERBOSE, self.config, "  ")
        
        files = {}
        local_dir = None
        
        # Create local directory if saving to disk
        if output_path:
            local_dir = output_path / f"{filing.cik}_{filing.accession_no_dash}"
            local_dir.mkdir(exist_ok=True)
        
        # Download each required file
        for filename in self.strategy.get_required_files(archive_url):
            file_url = f"{archive_url}/{filename}"
            _log(f"Downloading {file_url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            try:
                response = self.http.get(file_url)
                content = response.content
                
                # Store in memory stream (always available)
                files[filename] = BytesIO(content)
                
                # Optionally save to disk
                if local_dir:
                    (local_dir / filename).write_bytes(content)
                
                _log(f"✓ {filename}", VerbosityLevel.NORMAL, self.config, "  ")
                
            except Exception as e:
                _log(f"✗ {filename}: {e}", VerbosityLevel.VERBOSE, self.config, "  ")
            
            # Rate limiting between files
            time.sleep(self.config.request_delay)
        
        if not files:
            _log("No files downloaded", VerbosityLevel.ERROR, self.config, "  ")
            return None
        
        return FilingResult(
            metadata=filing,
            files=files,
            local_path=local_dir if output_path else None
        )


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Download to memory (no disk storage)
    print("Example 1: Download to memory streams")
    print("=" * 60)
    
    downloader = SECDownloader(
        "13F-HR", 
        config=DownloadConfig(verbosity=VerbosityLevel.NORMAL)
    )
    results = downloader.download("0001037389", num_reports=1, save_to_disk=False)
    
    if results:
        result = results[0]
        print(f"\nDownloaded {len(result.files)} files to memory")
        print(f"Files: {list(result.files.keys())}")
        print(f"CIK: {result.metadata.cik}")
        print(f"Filing date: {result.metadata.filing_date}")
    
    
    # Example 2: Download with local storage
    print("\n\nExample 2: Download with local storage")
    print("=" * 60)
    
    downloader = SECDownloader(
        "13F-HR", 
        config=DownloadConfig(verbosity=VerbosityLevel.VERBOSE)
    )
    results = downloader.download("0001037389", num_reports=1, save_to_disk=True)
    
    if results:
        result = results[0]
        print(f"\nSaved to: {result.local_path}")
        print(f"Also available as streams: {list(result.files.keys())}")
    
    
    # Example 3: NPORT with series filtering
    print("\n\nExample 3: NPORT with series filtering")
    print("=" * 60)
    
    downloader = SECDownloader(
        "NPORT-P", 
        series_id="S000002839",
        config=DownloadConfig(verbosity=VerbosityLevel.NORMAL)
    )
    results = downloader.download("0000036405", num_reports=1, save_to_disk=False)
    
    if results:
        print(f"\nDownloaded {len(results)} NPORT filing(s) for series S000002839")
