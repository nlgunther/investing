"""
SEC EDGAR Filing Downloader with Manifest Integration

Downloads SEC filings with automatic manifest-based tracking for incremental
downloads, deduplication, and local caching.

Architecture:
    FilingMetadata → Immutable filing identity
    HTTPClient → Network abstraction (testable)
    FileDiscoverer → Multi-strategy file detection
    DownloadStrategy → Filing-type-specific logic
    SECDownloader → Main orchestrator

Key Features:
    - Incremental downloads: Only fetch new filings via manifest
    - Dual-mode: Memory streams (always) + optional disk storage
    - Rate limiting: SEC-compliant request throttling
    - Strategy pattern: Easy addition of new filing types
    - Manifest integration: Automatic state tracking

Supported Filing Types:
    - 13F-HR: Hedge fund quarterly holdings
    - NPORT-P/NPORT-N: Mutual fund monthly holdings (with series filtering)

Extension Points:
    - Add new strategies to STRATEGIES registry
    - Override FileDiscoverer methods for new discovery patterns
    - Subclass DownloadStrategy for custom filtering logic

Example:
    >>> from downloader import SECDownloader
    >>> from manifest import Manifest
    >>> 
    >>> # Download with manifest tracking
    >>> downloader = SECDownloader("13F-HR")
    >>> results = downloader.download(
    ...     cik="0001037389",
    ...     num_reports=5,
    ...     save_to_disk=True
    ... )
    >>> 
    >>> # Only new filings are downloaded
    >>> for result in results:
    ...     print(f"Downloaded: {result.metadata.filing_date}")
    ...     print(f"Files: {list(result.files.keys())}")
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Protocol, Dict, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum
from io import BytesIO
from datetime import datetime
from functools import lru_cache
from bs4 import BeautifulSoup
import pandas as pd

# Import manifest system
from manifest import Manifest, ManifestEntry


# ============================================================================
# CONFIGURATION & LOGGING
# ============================================================================

class VerbosityLevel(IntEnum):
    """Logging verbosity control."""
    SILENT, ERROR, NORMAL, VERBOSE, DEBUG = range(5)


@dataclass
class DownloadConfig:
    """Configuration for SEC downloads with sensible defaults.
    
    Attributes:
        user_agent: Required by SEC (must identify your organization)
        base_url: SEC EDGAR web interface
        data_url: SEC data API endpoint
        rate_limit_delay: Seconds between filing downloads (SEC compliance)
        request_delay: Seconds between file downloads within a filing
        timeout: HTTP request timeout in seconds
        verbosity: Logging level
    
    SEC Requirements:
        - User-Agent must identify requester (include email)
        - Rate limit: Max 10 requests/second
        - See: https://www.sec.gov/os/accessing-edgar-data
    """
    user_agent: str = "Research Tool research@example.com"
    base_url: str = "https://www.sec.gov"
    data_url: str = "https://data.sec.gov"
    rate_limit_delay: float = 0.2  # 5 requests/second (conservative)
    request_delay: float = 0.1
    timeout: int = 30
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL


def _log(message: str, level: VerbosityLevel, config: DownloadConfig, indent: int = 0):
    """Centralized logging with verbosity control."""
    if config.verbosity >= level:
        print("  " * indent + message)


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SECError(Exception):
    """Base exception for SEC operations."""
    pass


class FilingNotFoundError(SECError):
    """Raised when filing cannot be found in SEC database."""
    pass


# ============================================================================
# HTTP CLIENT
# ============================================================================

class HTTPClient(Protocol):
    """Protocol for HTTP operations (enables testing without real network).
    
    Protocol Pattern:
        Defines interface without requiring inheritance.
        Any class with a get() method matching this signature works.
        Enables easy mocking in tests without subclassing.
    """
    def get(self, url: str, timeout: int = 30) -> requests.Response: ...


class DefaultHTTPClient:
    """Production HTTP client with connection pooling and SEC compliance.
    
    Features:
        - Session pooling: Reuses TCP connections
        - User-Agent injection: SEC requirement
        - Automatic error handling
        - Timeout management
    """
    
    def __init__(self, config: DownloadConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": config.user_agent,
            "Accept-Encoding": "gzip, deflate",
        })
    
    def get(self, url: str, timeout: Optional[int] = None) -> requests.Response:
        """Execute GET request with error handling."""
        try:
            response = self.session.get(url, timeout=timeout or self.config.timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise SECError(f"Request failed: {e}")


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass(frozen=True)
class FilingMetadata:
    """Immutable metadata for a single SEC filing."""
    cik: str
    accession_number: str
    filing_date: str  # ISO: YYYY-MM-DD
    form_type: str
    
    @property
    def accession_no_dash(self) -> str:
        """Accession number without dashes (required for SEC URLs)."""
        return self.accession_number.replace("-", "")
    
    def archive_url(self, base_url: str) -> str:
        """Construct SEC archive URL for this filing."""
        return f"{base_url}/Archives/edgar/data/{int(self.cik)}/{self.accession_no_dash}"
    
    def to_dict(self) -> dict:
        """Convert to dict for manifest storage."""
        return {
            'cik': self.cik,
            'accession_number': self.accession_number,
            'filing_date': self.filing_date,
            'form_type': self.form_type
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'FilingMetadata':
        """Restore from dict."""
        return cls(**data)


@dataclass
class FilingResult:
    """Result of a successful filing download."""
    metadata: FilingMetadata
    files: Dict[str, BytesIO]  # filename → stream (always available)
    local_path: Optional[Path] = None  # Set if saved to disk


# ============================================================================
# METADATA FETCHER
# ============================================================================

class FilingMetadataFetcher:
    """Unified interface for fetching filing metadata with manifest awareness."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig, manifest: Manifest):
        self.http = http
        self.config = config
        self.manifest = manifest
    
    @lru_cache(maxsize=32)
    def fetch_remote(self, cik: str, form_type: str) -> Tuple[FilingMetadata, ...]:
        """Fetch filing list from SEC API with caching."""
        try:
            url = f"{self.config.data_url}/submissions/CIK{cik}.json"
            _log(f"Fetching from SEC API: CIK{cik}", VerbosityLevel.DEBUG, self.config, 1)
            
            data = self.http.get(url).json()
            recent = data.get("filings", {}).get("recent", {})
            
            filings = [
                FilingMetadata(
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    form_type=form_type
                )
                for i, form in enumerate(recent.get("form", []))
                if form == form_type
            ]
            
            _log(f"Found {len(filings)} {form_type} filings", VerbosityLevel.DEBUG, self.config, 2)
            return tuple(filings)  # Tuple for caching
            
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch CIK {cik}: {e}")
    
    def get_new_filings(self, cik: str, form_type: str) -> List[FilingMetadata]:
        """Identify filings that need downloading (manifest-aware)."""
        remote = list(self.fetch_remote(cik, form_type))
        latest_date = self.manifest.get_latest_date()
        
        _log(f"Latest manifest date: {latest_date or 'None (empty)'}", VerbosityLevel.DEBUG, self.config, 1)
        
        # Filter: not in manifest AND date > latest
        new = [
            filing for filing in remote
            if not self.manifest.has_entry(filing.accession_number)
            and (latest_date is None or pd.to_datetime(filing.filing_date) > latest_date)
        ]
        
        return sorted(new, key=lambda f: f.filing_date)


# ============================================================================
# FILE DISCOVERY
# ============================================================================

class FileDiscoverer:
    """Discovers XML files in SEC filings using cascading fallback strategies."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        self.http = http
        self.config = config
        self._strategies = [
            self._try_index_json,
            self._try_index_html,
            self._get_defaults
        ]
    
    def discover(self, archive_url: str) -> List[str]:
        """Discover XML files using fallback strategies."""
        for strategy in self._strategies:
            files = strategy(archive_url)
            if files:
                return files
        return []
    
    def _try_index_json(self, url: str) -> Optional[List[str]]:
        """Strategy 1: Parse structured index.json."""
        try:
            data = self.http.get(f"{url}/index.json").json()
            files = [item['name'] for item in data['directory']['item'] if item['name'].endswith('.xml')]
            if files:
                _log(f"Found {len(files)} files via index.json", VerbosityLevel.DEBUG, self.config, 3)
            return files if files else None
        except:
            return None
    
    def _try_index_html(self, url: str) -> Optional[List[str]]:
        """Strategy 2: Parse HTML directory listing."""
        try:
            html = self.http.get(f"{url}/index.html").text
            soup = BeautifulSoup(html, 'html.parser')
            files = [link.get('href', '').split('/')[-1] for link in soup.find_all('a')]
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                _log(f"Found {len(xml_files)} files via index.html", VerbosityLevel.DEBUG, self.config, 3)
            return xml_files or None
        except:
            return None
    
    def _get_defaults(self, url: str) -> List[str]:
        """Strategy 3: Return common XML filenames."""
        return ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"]


# ============================================================================
# DOWNLOAD STRATEGIES
# ============================================================================

class DownloadStrategy(ABC):
    """Abstract strategy defining filing-type-specific logic."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        self.http = http
        self.config = config
    
    @abstractmethod
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Determine if this filing passes type-specific filters."""
        pass
    
    @abstractmethod
    def get_required_files(self, archive_url: str) -> List[str]:
        """Get list of files required for this filing type."""
        pass


class Strategy13F(DownloadStrategy):
    """Strategy for 13-F filings (hedge fund quarterly holdings)."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig):
        super().__init__(http, config)
        self.discoverer = FileDiscoverer(http, config)
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """13-F: Download all filings (no filtering)."""
        return True
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """13-F: Discover all XML files (includes holdings table)."""
        return self.discoverer.discover(archive_url)


class StrategyNPORT(DownloadStrategy):
    """Strategy for NPORT filings (mutual fund monthly holdings)."""
    
    def __init__(self, http: HTTPClient, config: DownloadConfig, series_id: Optional[str] = None):
        super().__init__(http, config)
        self.series_id = series_id
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """NPORT: Filter by series_id if specified."""
        if not self.series_id:
            return True
        
        try:
            response = self.http.get(f"{archive_url}/primary_doc.xml")
            root = ET.fromstring(response.content)
            
            # Strip namespaces
            for elem in root.iter():
                if '}' in elem.tag:
                    elem.tag = elem.tag.split('}', 1)[1]
            
            series_elem = root.find('.//seriesId')
            if series_elem is not None and series_elem.text:
                matches = series_elem.text.strip() == self.series_id
                _log(f"Series {series_elem.text.strip()} {'matches' if matches else 'does not match'}",
                     VerbosityLevel.DEBUG, self.config, 3)
                return matches
            
            return False
        except Exception as e:
            _log(f"Series check failed: {e}", VerbosityLevel.VERBOSE, self.config, 2)
            return False
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """NPORT: Single file contains everything."""
        return ["primary_doc.xml"]


# ============================================================================
# MAIN DOWNLOADER
# ============================================================================

class SECDownloader:
    """Main orchestrator for SEC filing downloads with manifest integration."""
    
    # Registry: form_type → (strategy_class, needs_series_id)
    STRATEGIES: Dict[str, Tuple[type, bool]] = {
        '13F-HR': (Strategy13F, False),
        'NPORT-P': (StrategyNPORT, True),
        'NPORT-N': (StrategyNPORT, True),
    }
    
    def __init__(self,
                 form_type: str,
                 config: Optional[DownloadConfig] = None,
                 series_id: Optional[str] = None,
                 http_client: Optional[HTTPClient] = None,
                 manifest_path: Optional[Path] = None,
                 use_manifest: bool = True):
        """Initialize downloader with dependency injection."""
        if form_type not in self.STRATEGIES:
            raise ValueError(f"Unsupported form type: {form_type}. "
                           f"Supported: {', '.join(self.STRATEGIES)}")
        
        self.form_type = form_type
        self.config = config or DownloadConfig()
        self.use_manifest = use_manifest
        
        # Initialize HTTP client
        self.http = http_client or DefaultHTTPClient(self.config)
        
        # Initialize manifest system (if enabled)
        if use_manifest:
            self.manifest = Manifest(manifest_path)
            self.fetcher = FilingMetadataFetcher(self.http, self.config, self.manifest)
        else:
            self.manifest = None
            self.fetcher = None
        
        # Create appropriate strategy
        strategy_cls, needs_series = self.STRATEGIES[form_type]
        kwargs = {'http': self.http, 'config': self.config}
        if needs_series and series_id:
            kwargs['series_id'] = series_id
        self.strategy = strategy_cls(**kwargs)
    
    def download(self,
                 cik: str,
                 num_reports: int = 5,
                 save_to_disk: bool = False,
                 output_dir: str = "sec_filings",
                 fetch_all: bool = False) -> List[FilingResult]:
        """Download SEC filings with automatic manifest tracking."""
        cik = cik.strip().replace('-', '').zfill(10)
        
        _log(f"Fetching {self.form_type} for CIK {cik}", VerbosityLevel.NORMAL, self.config)
        
        # Get filings (manifest-aware or all)
        if self.use_manifest and not fetch_all:
            filings = self.fetcher.get_new_filings(cik, self.form_type)
            _log(f"Found {len(filings)} new filing(s)", VerbosityLevel.NORMAL, self.config)
        else:
            filings = self._fetch_all_filings(cik)
            _log(f"Found {len(filings)} total filing(s)", VerbosityLevel.NORMAL, self.config)
        
        if not filings:
            return []
        
        # Setup output directory
        output_path = Path(output_dir) if save_to_disk else None
        if output_path:
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Download with filtering
        results = []
        for filing in filings:
            if len(results) >= num_reports:
                break
            
            archive_url = filing.archive_url(self.config.base_url)
            
            if not self.strategy.should_download(filing, archive_url):
                _log(f"Skipped {filing.filing_date} (filtered)", VerbosityLevel.VERBOSE, self.config, 1)
                continue
            
            result = self._download_filing(filing, archive_url, output_path)
            if result:
                results.append(result)
                if self.use_manifest and save_to_disk:
                    self._update_manifest(result)
            
            time.sleep(self.config.rate_limit_delay)
        
        _log(f"\n✓ Downloaded {len(results)} filing(s)", VerbosityLevel.NORMAL, self.config)
        return results
    
    def _fetch_all_filings(self, cik: str) -> List[FilingMetadata]:
        """Fetch complete filing list from SEC API (bypasses manifest)."""
        try:
            url = f"{self.config.data_url}/submissions/CIK{cik}.json"
            data = self.http.get(url).json()
            recent = data.get("filings", {}).get("recent", {})
            
            return [
                FilingMetadata(
                    cik=cik,
                    accession_number=recent["accessionNumber"][i],
                    filing_date=recent["filingDate"][i],
                    form_type=self.form_type
                )
                for i, form in enumerate(recent.get("form", []))
                if form == self.form_type
            ]
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch filings: {e}")
    
    def _download_filing(self,
                        filing: FilingMetadata,
                        archive_url: str,
                        output_path: Optional[Path]) -> Optional[FilingResult]:
        """Download a single filing's files."""
        _log(f"\n  {filing.filing_date} ({filing.accession_number})", VerbosityLevel.NORMAL, self.config, 1)
        _log(f"URL: {archive_url}", VerbosityLevel.VERBOSE, self.config, 2)
        
        files: Dict[str, BytesIO] = {}
        local_dir: Optional[Path] = None
        
        if output_path:
            local_dir = output_path / f"{filing.cik}_{filing.accession_no_dash}"
            local_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in self.strategy.get_required_files(archive_url):
            file_url = f"{archive_url}/{filename}"
            
            try:
                content = self.http.get(file_url).content
                files[filename] = BytesIO(content)
                
                if local_dir:
                    (local_dir / filename).write_bytes(content)
                
                _log(f"✓ {filename}", VerbosityLevel.NORMAL, self.config, 2)
            except Exception as e:
                _log(f"✗ {filename}: {e}", VerbosityLevel.VERBOSE, self.config, 2)
            
            time.sleep(self.config.request_delay)
        
        return FilingResult(filing, files, local_dir) if files else None
    
    def _update_manifest(self, result: FilingResult):
        """Update manifest with newly downloaded filing."""
        if not result.local_path:
            return
        
        entry = ManifestEntry(
            metadata=result.metadata.to_dict(),
            local_path=result.local_path,
            timestamp=datetime.now().isoformat(),
            files=list(result.files.keys())
        )
        
        self.manifest.add_entry(entry)
        self.manifest.save()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def download_filings(form_type: str, cik: str, num_reports: int = 5, **kwargs) -> List[FilingResult]:
    """Convenience function for one-line downloads."""
    downloader = SECDownloader(form_type, **kwargs)
    return downloader.download(cik, num_reports)


# ============================================================================
# MODULE METADATA
# ============================================================================

__version__ = "2.0.0"
__all__ = [
    'SECDownloader',
    'FilingMetadata',
    'FilingResult',
    'DownloadConfig',
    'VerbosityLevel',
    'HTTPClient',
    'DefaultHTTPClient',
    'DownloadStrategy',
    'Strategy13F',
    'StrategyNPORT',
    'FileDiscoverer',
    'FilingMetadataFetcher',
    'SECError',
    'FilingNotFoundError',
    'download_filings',
]