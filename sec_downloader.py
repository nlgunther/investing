"""
SEC Filing Downloader Module

Downloads SEC filings and returns either file paths or IO streams.
Supports multiple filing types through strategy pattern.
"""

import requests
import time
from pathlib import Path
from typing import List, Optional, Protocol, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum
from io import BytesIO
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup


# ============================================================================
# CONFIGURATION
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


def log(message: str, level: VerbosityLevel, config: DownloadConfig, prefix: str = ""):
    """Centralized logging function."""
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
# HTTP CLIENT
# ============================================================================

class HTTPClient(Protocol):
    """Protocol for HTTP operations."""
    def get(self, url: str, timeout: int = 30) -> requests.Response: ...


class DefaultHTTPClient:
    """Default HTTP client implementation."""
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.headers = {"User-Agent": config.user_agent}
    
    def get(self, url: str, timeout: int = None) -> requests.Response:
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
        """Accession number without dashes."""
        return self.accession_number.replace("-", "")
    
    def archive_url(self, base_url: str) -> str:
        """Construct SEC archive URL."""
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
    """Discovers XML files in SEC filings."""
    
    def __init__(self, http_client: HTTPClient, config: DownloadConfig):
        self.http = http_client
        self.config = config
    
    def discover(self, archive_url: str) -> List[str]:
        """Discover all XML files in a filing."""
        log(f"Discovering files at {archive_url}", VerbosityLevel.DEBUG, self.config, "    ")
        
        return (self._try_index_json(archive_url) or
                self._try_index_html(archive_url) or
                ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"])
    
    def _try_index_json(self, url: str) -> Optional[List[str]]:
        """Try to get file list from index.json."""
        try:
            data = self.http.get(f"{url}/index.json").json()
            files = [item['name'] for item in data['directory']['item'] 
                    if item['name'].endswith('.xml')]
            log(f"Found {len(files)} XML files via index.json", 
                VerbosityLevel.DEBUG, self.config, "      ")
            return files
        except:
            return None
    
    def _try_index_html(self, url: str) -> Optional[List[str]]:
        """Try to parse file list from index.html."""
        try:
            soup = BeautifulSoup(self.http.get(f"{url}/index.html").text, 'html.parser')
            files = [link.get('href', '').split('/')[-1] for link in soup.find_all('a')]
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                log(f"Found {len(xml_files)} XML files via index.html", 
                    VerbosityLevel.DEBUG, self.config, "      ")
            return xml_files or None
        except:
            return None


# ============================================================================
# DOWNLOAD STRATEGIES
# ============================================================================

class DownloadStrategy(ABC):
    """Abstract download strategy."""
    
    def __init__(self, http_client: HTTPClient, config: DownloadConfig):
        self.http = http_client
        self.config = config
    
    @abstractmethod
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Determine if filing should be downloaded."""
        pass
    
    @abstractmethod
    def get_required_files(self, archive_url: str) -> List[str]:
        """Get list of required files for this filing type."""
        pass


class Strategy13F(DownloadStrategy):
    """Download strategy for 13-F filings."""
    
    def __init__(self, http_client: HTTPClient, config: DownloadConfig):
        super().__init__(http_client, config)
        self.discoverer = FileDiscoverer(http_client, config)
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Download all 13-F filings."""
        return True
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """Discover all XML files."""
        return self.discoverer.discover(archive_url)


class StrategyNPORT(DownloadStrategy):
    """Download strategy for NPORT filings with series filtering."""
    
    def __init__(self, http_client: HTTPClient, config: DownloadConfig, 
                 series_id: Optional[str] = None):
        super().__init__(http_client, config)
        self.series_id = series_id
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Check if filing matches series ID filter."""
        if not self.series_id:
            return True
        
        log(f"Checking series ID for {filing.accession_number}", 
            VerbosityLevel.DEBUG, self.config, "    ")
        
        try:
            response = self.http.get(f"{archive_url}/primary_doc.xml")
            root = ET.fromstring(response.content)
            
            # Remove namespaces
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]
            
            series_elem = root.find('.//seriesId')
            if series_elem is not None and series_elem.text:
                matches = series_elem.text.strip() == self.series_id
                log(f"Series ID {series_elem.text.strip()} {'matches' if matches else 'does not match'}", 
                    VerbosityLevel.DEBUG, self.config, "      ")
                return matches
            
            return False
        except Exception as e:
            log(f"Warning: Could not check series ID: {e}", 
                VerbosityLevel.VERBOSE, self.config, "    ")
            return False
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """NPORT uses single file."""
        return ["primary_doc.xml"]


# ============================================================================
# DOWNLOADER
# ============================================================================

class SECDownloader:
    """Downloads SEC filings with optional local storage."""
    
    STRATEGIES = {
        '13F-HR': Strategy13F,
        'NPORT-P': StrategyNPORT,
        'NPORT-N': StrategyNPORT,
    }
    
    def __init__(self, form_type: str, config: Optional[DownloadConfig] = None,
                 series_id: Optional[str] = None):
        """Initialize downloader.
        
        Args:
            form_type: Form type (e.g., "13F-HR", "NPORT-P")
            config: Download configuration
            series_id: Series ID for NPORT filings
        """
        if form_type not in self.STRATEGIES:
            raise ValueError(f"Unsupported form type: {form_type}. "
                           f"Supported: {list(self.STRATEGIES.keys())}")
        
        self.form_type = form_type
        self.config = config or DownloadConfig()
        
        # Create HTTP client and strategy
        http_client = DefaultHTTPClient(self.config)
        strategy_cls = self.STRATEGIES[form_type]
        
        if form_type in ['NPORT-P', 'NPORT-N']:
            self.strategy = strategy_cls(http_client, self.config, series_id)
        else:
            self.strategy = strategy_cls(http_client, self.config)
        
        self.http = http_client
    
    def download(self, cik: str, num_reports: int = 5, 
                save_to_disk: bool = False, output_dir: str = "sec_filings") -> List[FilingResult]:
        """Download filings.
        
        Args:
            cik: Central Index Key
            num_reports: Maximum number of filings to download
            save_to_disk: If True, save files locally and set local_path
            output_dir: Directory for local storage (if save_to_disk=True)
            
        Returns:
            List of FilingResult objects with streams (and paths if saved)
        """
        cik = cik.strip().replace('-', '').zfill(10)
        log(f"Fetching {self.form_type} filings for CIK {cik}...", 
            VerbosityLevel.NORMAL, self.config)
        
        filings = self._fetch_filing_list(cik)
        if not filings:
            log(f"No {self.form_type} filings found", VerbosityLevel.NORMAL, self.config)
            return []
        
        log(f"Found {len(filings)} filing(s), checking filters...", 
            VerbosityLevel.NORMAL, self.config)
        
        results = []
        output_path = Path(output_dir) if save_to_disk else None
        if output_path:
            output_path.mkdir(exist_ok=True)
        
        for filing in filings:
            if len(results) >= num_reports:
                break
            
            archive_url = filing.archive_url(self.config.base_url)
            
            if not self.strategy.should_download(filing, archive_url):
                log(f"Skipping {filing.filing_date} (filter)", 
                    VerbosityLevel.VERBOSE, self.config, "  ")
                continue
            
            result = self._download_filing(filing, archive_url, output_path)
            if result:
                results.append(result)
            
            time.sleep(self.config.rate_limit_delay)
        
        log(f"Downloaded {len(results)} filing(s)", VerbosityLevel.NORMAL, self.config, "\n✓ ")
        return results
    
    def _fetch_filing_list(self, cik: str) -> List[FilingMetadata]:
        """Fetch list of filings from SEC API."""
        try:
            url = f"{self.config.data_url}/submissions/CIK{cik}.json"
            log(f"Fetching filing list from {url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            data = self.http.get(url).json()
            filings = data.get("filings", {}).get("recent", {})
            
            result = [
                FilingMetadata(cik, filings["accessionNumber"][i], 
                             filings["filingDate"][i], self.form_type)
                for i, form in enumerate(filings.get("form", [])) 
                if form == self.form_type
            ]
            
            return result
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch filings: {e}")
    
    def _download_filing(self, filing: FilingMetadata, archive_url: str,
                        output_path: Optional[Path]) -> Optional[FilingResult]:
        """Download a single filing's files."""
        log(f"{filing.filing_date} ({filing.accession_number})", 
            VerbosityLevel.NORMAL, self.config, "\n")
        log(f"Archive URL: {archive_url}", VerbosityLevel.VERBOSE, self.config, "  ")
        
        files = {}
        local_dir = None
        
        if output_path:
            local_dir = output_path / f"{filing.cik}_{filing.accession_no_dash}"
            local_dir.mkdir(exist_ok=True)
        
        for filename in self.strategy.get_required_files(archive_url):
            file_url = f"{archive_url}/{filename}"
            log(f"Downloading {file_url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            try:
                response = self.http.get(file_url)
                content = response.content
                
                # Store in memory stream
                files[filename] = BytesIO(content)
                
                # Optionally save to disk
                if local_dir:
                    (local_dir / filename).write_bytes(content)
                
                log(f"✓ {filename}", VerbosityLevel.NORMAL, self.config, "  ")
            except Exception as e:
                log(f"✗ {filename}: {e}", VerbosityLevel.VERBOSE, self.config, "  ")
            
            time.sleep(self.config.request_delay)
        
        if not files:
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
    downloader = SECDownloader("13F-HR", config=DownloadConfig(verbosity=VerbosityLevel.NORMAL))
    results = downloader.download("0001037389", num_reports=1, save_to_disk=False)
    
    if results:
        result = results[0]
        print(f"\nDownloaded {len(result.files)} files to memory")
        print(f"Files: {list(result.files.keys())}")
    
    # Example 2: Download with local storage
    print("\n\nExample 2: Download with local storage")
    print("=" * 60)
    downloader = SECDownloader("13F-HR", config=DownloadConfig(verbosity=VerbosityLevel.VERBOSE))
    results = downloader.download("0001037389", num_reports=1, save_to_disk=True)
    
    if results:
        result = results[0]
        print(f"\nSaved to: {result.local_path}")
        print(f"Also available as streams: {list(result.files.keys())}")
