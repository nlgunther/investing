"""
SEC Filing Downloader and Parser Module

A flexible, extensible system for downloading and parsing SEC filings.
Currently supports:
- 13-F filings (hedge fund holdings)
- NPORT-P/NPORT-N filings (mutual fund holdings with series filtering)

Architecture:
- Strategy pattern for different filing types (easy to extend)
- Composition over inheritance for flexibility
- Protocol-based HTTP client for testability
- Centralized configuration and verbosity control

To add a new filing type:
1. Create DownloaderStrategy subclass
2. Create ParserStrategy subclass  
3. Add to SECFilingManager.STRATEGIES registry
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Tuple, Optional, List, Protocol
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import IntEnum


# ============================================================================
# CONFIGURATION & VERBOSITY
# ============================================================================

class VerbosityLevel(IntEnum):
    """Verbosity levels for output control."""
    SILENT = 0   # No output
    ERROR = 1    # Errors only
    NORMAL = 2   # Standard progress messages
    VERBOSE = 3  # Detailed information
    DEBUG = 4    # Debug information including URLs


@dataclass
class SECConfig:
    """Configuration for SEC API interactions.
    
    Attributes:
        user_agent: User agent string for SEC requests (required by SEC)
        base_url: Base URL for SEC website
        data_url: Base URL for SEC data API
        rate_limit_delay: Delay between filing downloads (seconds)
        request_delay: Delay between file downloads within a filing (seconds)
        timeout: HTTP request timeout (seconds)
        verbosity: Output verbosity level
    """
    user_agent: str = "Research Tool research@example.com"
    base_url: str = "https://www.sec.gov"
    data_url: str = "https://data.sec.gov"
    rate_limit_delay: float = 0.2
    request_delay: float = 0.1
    timeout: int = 30
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL


def log(message: str, level: VerbosityLevel, config: SECConfig, prefix: str = ""):
    """Centralized logging function respecting verbosity settings.
    
    Args:
        message: Message to log
        level: Minimum verbosity level required to show this message
        config: Configuration containing current verbosity setting
        prefix: Optional prefix for indentation/formatting
    """
    if config.verbosity >= level:
        print(f"{prefix}{message}")


# ============================================================================
# EXCEPTIONS
# ============================================================================

class SECFilingError(Exception):
    """Base exception for SEC filing operations."""
    pass


class FilingNotFoundError(SECFilingError):
    """Raised when a filing cannot be found."""
    pass


class ParsingError(SECFilingError):
    """Raised when XML parsing fails."""
    pass


# ============================================================================
# HTTP CLIENT
# ============================================================================

class HTTPClient(Protocol):
    """Protocol for HTTP operations.
    
    Enables dependency injection for testing without making real HTTP calls.
    Any class implementing get() can be used as HTTP client.
    """
    def get(self, url: str, timeout: int = 30) -> requests.Response: ...


class DefaultHTTPClient:
    """Default HTTP client implementation using requests library."""
    
    def __init__(self, config: SECConfig):
        self.config = config
        self.headers = {"User-Agent": config.user_agent}
    
    def get(self, url: str, timeout: int = None) -> requests.Response:
        """Make HTTP GET request with configured headers.
        
        Raises:
            requests.HTTPError: If request fails
        """
        response = requests.get(url, headers=self.headers, timeout=timeout or self.config.timeout)
        response.raise_for_status()
        return response


# ============================================================================
# XML UTILITIES
# ============================================================================

class XMLParser:
    """Utility class for XML parsing operations.
    
    All methods are static as this is a pure utility class with no state.
    Handles namespace removal and hierarchical data extraction.
    """
    
    @staticmethod
    def parse_file(path: Path) -> Optional[ET.Element]:
        """Parse XML file and remove namespaces.
        
        Returns:
            Root element with namespaces removed, or None if file missing/invalid
        """
        if not path.exists():
            return None
        try:
            tree = ET.parse(path)
            return XMLParser.remove_namespaces(tree.getroot())
        except ET.ParseError:
            return None
    
    @staticmethod
    def remove_namespaces(root: ET.Element) -> ET.Element:
        """Remove XML namespaces from element tree.
        
        SEC XML files use namespaces like xmlns="http://www.sec.gov/edgar/..."
        This strips them for easier querying.
        """
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        return root
    
    @staticmethod
    def to_dict(root: ET.Element) -> Dict[str, str]:
        """Extract all leaf node values with hierarchical keys.
        
        Converts nested XML like:
            <parent><child>value</child></parent>
        To:
            {'parent_child': 'value'}
        
        This flattened structure makes it easy to access values without
        navigating the XML tree structure.
        """
        data = {}
        def traverse(element: ET.Element, path: List[str] = []):
            for child in element:
                tag = child.tag.split('}')[-1]  # Remove namespace if present
                key = '_'.join(path + [tag]).lower()
                if len(child) == 0 and child.text and child.text.strip():
                    # Leaf node - store the value
                    data[key] = child.text.strip()
                else:
                    # Branch node - recurse
                    traverse(child, path + [tag])
        traverse(root)
        return data
    
    @staticmethod
    def find_all(root: ET.Element, tag: str) -> List[ET.Element]:
        """Find all elements with tag (namespace-agnostic).
        
        Tries multiple strategies to find elements regardless of namespace:
        1. Using root's namespace
        2. Without namespace
        3. By checking tag endings
        """
        return (root.findall(f'.//{{{root.tag.split("}")[0][1:]}}}{tag}') or
                root.findall(f'.//{tag}') or
                [e for e in root.findall('.//*') if e.tag.endswith(tag)])


# ============================================================================
# FILE DISCOVERY
# ============================================================================

class FilingFileDiscoverer:
    """Discovers XML files in SEC filings.
    
    SEC filings can have various XML files. This class tries multiple
    methods to discover them:
    1. index.json (structured data)
    2. index.html (fallback)
    3. Common filenames (last resort)
    """
    
    def __init__(self, http_client: HTTPClient, config: SECConfig):
        self.http_client = http_client
        self.config = config
    
    def discover(self, archive_url: str) -> List[str]:
        """Discover all XML files in a filing."""
        log(f"Discovering files at {archive_url}", VerbosityLevel.DEBUG, self.config, "    ")
        
        return (self._try_index_json(archive_url) or
                self._try_index_html(archive_url) or
                ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"])
    
    def _try_index_json(self, url: str) -> Optional[List[str]]:
        """Try to get file list from index.json (preferred method)."""
        try:
            data = self.http_client.get(f"{url}/index.json").json()
            files = [item['name'] for item in data['directory']['item'] if item['name'].endswith('.xml')]
            log(f"Found {len(files)} XML files via index.json", VerbosityLevel.DEBUG, self.config, "      ")
            return files
        except:
            return None
    
    def _try_index_html(self, url: str) -> Optional[List[str]]:
        """Try to parse file list from index.html (fallback method)."""
        try:
            soup = BeautifulSoup(self.http_client.get(f"{url}/index.html").text, 'html.parser')
            files = [link.get('href', '').split('/')[-1] for link in soup.find_all('a')]
            xml_files = [f for f in files if f.endswith('.xml')]
            if xml_files:
                log(f"Found {len(xml_files)} XML files via index.html", VerbosityLevel.DEBUG, self.config, "      ")
            return xml_files or None
        except:
            return None


# ============================================================================
# FILING METADATA
# ============================================================================

@dataclass
class FilingMetadata:
    """Metadata about a single SEC filing.
    
    Contains the minimal information needed to identify and download a filing.
    The archive URL is constructed from these fields.
    """
    cik: str                 # Central Index Key (company identifier)
    accession_number: str    # Unique filing identifier (e.g., "0001037389-24-000123")
    filing_date: str         # Date filing was submitted to SEC
    form_type: str           # Type of form (e.g., "13F-HR", "NPORT-P")
    
    @property
    def accession_no_dash(self) -> str:
        """Accession number without dashes (used in URLs)."""
        return self.accession_number.replace("-", "")
    
    def archive_url(self, base_url: str) -> str:
        """Construct SEC archive URL for this filing."""
        return f"{base_url}/Archives/edgar/data/{int(self.cik)}/{self.accession_no_dash}"


# ============================================================================
# STRATEGIES
# ============================================================================

class FilingStrategy(ABC):
    """Base strategy class providing common functionality.
    
    Both downloader and parser strategies inherit from this to get
    access to HTTP client, config, and XML parser utilities.
    """
    
    def __init__(self, http_client: HTTPClient, config: SECConfig):
        self.http = http_client
        self.config = config
        self.xml = XMLParser()


class DownloaderStrategy(FilingStrategy):
    """Abstract downloader strategy.
    
    Subclasses implement specific logic for different filing types:
    - Which filings to download (filtering logic)
    - Which files are needed for this filing type
    """
    
    @abstractmethod
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Determine if filing should be downloaded (e.g., series filtering)."""
        pass
    
    @abstractmethod
    def get_files(self, archive_url: str) -> List[str]:
        """Get list of required files for this filing type."""
        pass


class ParserStrategy(FilingStrategy):
    """Abstract parser strategy.
    
    Subclasses implement specific logic for different filing types:
    - How to extract metadata from primary_doc.xml
    - Where to find holdings data
    - How to process holdings into DataFrame
    """
    
    @abstractmethod
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Parse metadata from primary document."""
        pass
    
    @abstractmethod
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse holdings data into DataFrame."""
        pass
    
    @abstractmethod
    def get_aliases(self) -> Dict[str, str]:
        """Get metadata field aliases (short names for common fields)."""
        pass
    
    def _to_numeric(self, df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
        """Convert columns matching patterns to numeric types.
        
        Shared utility method for all parsers. Silently ignores conversion errors.
        """
        for col in df.columns:
            if any(p in col for p in patterns):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass  # Keep as string if conversion fails
        return df


# ============================================================================
# 13-F STRATEGIES
# ============================================================================

class Filing13FDownloader(DownloaderStrategy):
    """Downloads 13-F filings (hedge fund holdings).
    
    13-F filings report quarterly holdings for institutional investment managers
    with over $100M in assets. No filtering is applied - all 13-F filings
    for a CIK are downloaded.
    """
    
    def __init__(self, http_client: HTTPClient, config: SECConfig):
        super().__init__(http_client, config)
        self.discoverer = FilingFileDiscoverer(http_client, config)
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Download all 13-F filings (no filtering)."""
        return True
    
    def get_files(self, archive_url: str) -> List[str]:
        """Discover all XML files (13-F has separate holdings file)."""
        return self.discoverer.discover(archive_url)


class Filing13FParser(ParserStrategy):
    """Parses 13-F filings.
    
    13-F structure:
    - primary_doc.xml: Metadata (filer info, period, etc.)
    - *_holdings.xml or infotable.xml: Holdings data (one infoTable per holding)
    """
    
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml."""
        root = self.xml.parse_file(xml_path)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        # Add convenient aliases for commonly used fields
        return {**metadata, **{alias: metadata.get(key, '') for alias, key in self.get_aliases().items()}}
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse holdings from separate holdings XML file.
        
        Searches for files with 'holding' in name or named 'infotable.xml'.
        Each <infoTable> element represents one holding.
        """
        # Find holdings file (naming varies by filer)
        holdings_file = next((f for f in filing_path.glob("*") 
                             if 'holding' in f.name.lower() or f.name == 'infotable.xml'), None)
        if not holdings_file:
            log("No holdings file found", VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        root = self.xml.parse_file(holdings_file)
        if not root:
            return pd.DataFrame()
        
        # Extract all infoTable elements (one per holding)
        holdings = [self.xml.to_dict(table) for table in self.xml.find_all(root, 'infoTable')]
        if not holdings:
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['value', 'sshprnamt'])
        
        # Calculate unit value (price per share) if possible
        if 'value' in df.columns and 'shrsorprnamt_sshprnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['shrsorprnamt_sshprnamt'])
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys for 13-F metadata."""
        return {
            'cik': 'headerdata_filerinfo_filer_credentials_cik',
            'periodofreport': 'headerdata_filerinfo_periodofreport',
            'name': 'formdata_coverpage_filingmanager_name'
        }


# ============================================================================
# NPORT STRATEGIES
# ============================================================================

class FilingNPORTDownloader(DownloaderStrategy):
    """Downloads NPORT filings (mutual fund holdings) with series filtering.
    
    NPORT-P filings report monthly holdings for registered investment companies
    (mutual funds). A single CIK can have multiple series (different funds),
    so we support filtering by series ID.
    """
    
    def __init__(self, http_client: HTTPClient, config: SECConfig, series_id: Optional[str] = None):
        super().__init__(http_client, config)
        self.series_id = series_id
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Check if filing matches series ID filter.
        
        If no series_id specified, download all filings.
        Otherwise, fetch primary_doc.xml and check the series ID.
        """
        if not self.series_id:
            return True
        
        log(f"Checking series ID for {filing.accession_number}", VerbosityLevel.DEBUG, self.config, "    ")
        
        try:
            # Fetch primary_doc.xml to check series
            response = self.http.get(f"{archive_url}/primary_doc.xml")
            root = ET.fromstring(response.content)
            root = self.xml.remove_namespaces(root)
            
            series_elem = root.find('.//seriesId')
            if series_elem is not None and series_elem.text:
                matches = series_elem.text.strip() == self.series_id
                log(f"Series ID {series_elem.text.strip()} {'matches' if matches else 'does not match'}", 
                    VerbosityLevel.DEBUG, self.config, "      ")
                return matches
            
            return False
        except Exception as e:
            log(f"Warning: Could not check series ID: {e}", VerbosityLevel.VERBOSE, self.config, "    ")
            return False
    
    def get_files(self, archive_url: str) -> List[str]:
        """NPORT uses single file (primary_doc.xml contains everything)."""
        return ["primary_doc.xml"]


class FilingNPORTParser(ParserStrategy):
    """Parses NPORT filings.
    
    NPORT structure (single file):
    - primary_doc.xml contains both metadata and holdings
    - Metadata in <headerData> and <genInfo> sections
    - Holdings in <invstOrSecs> section (one <invstOrSec> per holding)
    """
    
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml."""
        root = self.xml.parse_file(xml_path)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        # Add convenient aliases
        return {**metadata, **{alias: metadata.get(key, '') for alias, key in self.get_aliases().items()}}
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse holdings from primary_doc.xml (single file structure).
        
        Each <invstOrSec> element represents one security holding.
        """
        root = self.xml.parse_file(filing_path / "primary_doc.xml")
        if not root:
            return pd.DataFrame()
        
        # Extract all invstOrSec elements
        holdings = [self.xml.to_dict(sec) for sec in self.xml.find_all(root, 'invstOrSec')]
        if not holdings:
            log("No holdings found in invstOrSecs section", VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['valusd', 'balance', 'pctval'])
        
        # Calculate unit value (price per unit) if possible
        if 'valusd' in df.columns and 'balance' in df.columns:
            df = df.assign(unitValue=lambda x: x['valusd'] / x['balance'])
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys for NPORT metadata."""
        return {
            'cik': 'headerdata_filerinfo_filer_issuercredentials_cik',
            'periodofreport': 'formdata_geninfo_reppddate',  # Report date, not period end
            'name': 'formdata_geninfo_seriesname',
            'seriesid': 'headerdata_filerinfo_seriesclassinfo_seriesid'
        }


# ============================================================================
# DOWNLOADER & PARSER
# ============================================================================

class SECFilingDownloader:
    """Downloads SEC filings using pluggable strategy pattern.
    
    Responsibilities:
    - Fetch list of available filings from SEC API
    - Apply filters via strategy
    - Download files for each filing
    - Manage rate limiting
    """
    
    def __init__(self, strategy: DownloaderStrategy, config: SECConfig, output_dir: str = "sec_filings"):
        self.strategy = strategy
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download(self, cik: str, form_type: str, num_reports: int = 5) -> List[str]:
        """Download filings for specified CIK.
        
        Args:
            cik: Central Index Key (company identifier)
            form_type: Form type to download (e.g., "13F-HR", "NPORT-P")
            num_reports: Maximum number of filings to download
            
        Returns:
            List of paths to downloaded filing directories
        """
        # Normalize CIK to 10 digits with leading zeros
        cik = cik.strip().replace('-', '').zfill(10)
        log(f"Fetching {form_type} filings for CIK {cik}...", VerbosityLevel.NORMAL, self.config)
        
        filings = self._fetch_filing_list(cik, form_type)
        if not filings:
            log(f"No {form_type} filings found", VerbosityLevel.NORMAL, self.config)
            return []
        
        log(f"Found {len(filings)} filing(s), checking filters...", VerbosityLevel.NORMAL, self.config)
        
        # Download filings that pass strategy's filter
        downloaded = []
        for filing in filings:
            if len(downloaded) >= num_reports:
                break
            
            archive_url = filing.archive_url(self.config.base_url)
            
            if not self.strategy.should_download(filing, archive_url):
                log(f"Skipping {filing.filing_date} (filter)", VerbosityLevel.VERBOSE, self.config, "  ")
                continue
            
            path = self._download_filing(filing, archive_url)
            if path:
                downloaded.append(path)
            
            time.sleep(self.config.rate_limit_delay)
        
        log(f"Downloaded {len(downloaded)} filing(s)", VerbosityLevel.NORMAL, self.config, "\n✓ ")
        return downloaded
    
    def _fetch_filing_list(self, cik: str, form_type: str) -> List[FilingMetadata]:
        """Fetch list of filings from SEC submissions API."""
        try:
            url = f"{self.config.data_url}/submissions/CIK{cik}.json"
            log(f"Fetching filing list from {url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            data = self.strategy.http.get(url).json()
            filings = data.get("filings", {}).get("recent", {})
            
            # Filter to requested form type
            result = [FilingMetadata(cik, filings["accessionNumber"][i], filings["filingDate"][i], form_type)
                     for i, form in enumerate(filings.get("form", [])) if form == form_type]
            
            return result
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch filings: {e}")
    
    def _download_filing(self, filing: FilingMetadata, archive_url: str) -> Optional[str]:
        """Download a single filing's files.
        
        Creates directory named: {cik}_{accession_no_dash}
        Downloads all required files per strategy.
        """
        filing_dir = self.output_dir / f"{filing.cik}_{filing.accession_no_dash}"
        filing_dir.mkdir(exist_ok=True)
        
        log(f"{filing.filing_date} ({filing.accession_number})", VerbosityLevel.NORMAL, self.config, "\n")
        log(f"Archive URL: {archive_url}", VerbosityLevel.VERBOSE, self.config, "  ")
        
        downloaded = 0
        for filename in self.strategy.get_files(archive_url):
            file_url = f"{archive_url}/{filename}"
            log(f"Downloading {file_url}", VerbosityLevel.DEBUG, self.config, "  ")
            
            try:
                response = self.strategy.http.get(file_url)
                (filing_dir / filename).write_bytes(response.content)
                log(f"✓ {filename}", VerbosityLevel.NORMAL, self.config, "  ")
                downloaded += 1
            except Exception as e:
                log(f"✗ {filename}: {e}", VerbosityLevel.VERBOSE, self.config, "  ")
            
            time.sleep(self.config.request_delay)
        
        return str(filing_dir) if downloaded > 0 else None


class SECFilingParser:
    """Parses SEC filings using pluggable strategy pattern.
    
    Responsibilities:
    - Parse metadata from primary_doc.xml
    - Parse holdings data into DataFrame
    - Convert dates for dictionary keys
    """
    
    def __init__(self, strategy: ParserStrategy):
        self.strategy = strategy
    
    def parse_filing(self, filing_dir: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse single filing directory.
        
        Returns:
            Tuple of (metadata_dict, holdings_dataframe)
        """
        path = Path(filing_dir)
        return self.strategy.parse_metadata(path / "primary_doc.xml"), self.strategy.parse_holdings(path)
    
    def parse_multiple(self, filing_dirs: List[str]) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse multiple filings into dictionary keyed by (cik, period_date).
        
        This structure allows easy lookup by date and handles multiple CIKs.
        """
        results = {}
        for filing_dir in filing_dirs:
            try:
                metadata, holdings = self.parse_filing(filing_dir)
                cik = metadata.get('cik', 'unknown')
                period_str = metadata.get('periodofreport', 'unknown')
                
                try:
                    period_date = pd.to_datetime(period_str)
                    log(f"✓ {filing_dir}: CIK={cik}, Period={period_date.date()}, Holdings={len(holdings)}", 
                        VerbosityLevel.NORMAL, self.strategy.config)
                except Exception as e:
                    log(f"Warning: Date parse error '{period_str}': {e}", 
                        VerbosityLevel.VERBOSE, self.strategy.config, "  ")
                    period_date = pd.NaT
                
                results[(cik, period_date)] = (metadata, holdings)
            except Exception as e:
                log(f"✗ {filing_dir}: {e}", VerbosityLevel.ERROR, self.strategy.config)
        
        return results


# ============================================================================
# MANAGER (FACADE)
# ============================================================================

class SECFilingManager:
    """Facade providing simple interface for SEC filing operations.
    
    This is the main entry point for users of this module. It:
    - Selects appropriate strategies based on form type
    - Provides unified interface regardless of filing type
    - Stores parsed filings for later retrieval
    
    Example:
        manager = SECFilingManager(form_type="13F-HR", verbosity=VerbosityLevel.VERBOSE)
        filings = manager.download_and_parse("0001037389", num_reports=5)
    """
    
    # Strategy registry - add new form types here
    STRATEGIES = {
        '13F-HR': (Filing13FDownloader, Filing13FParser),
        'NPORT-P': (FilingNPORTDownloader, FilingNPORTParser),
        'NPORT-N': (FilingNPORTDownloader, FilingNPORTParser),
    }
    
    def __init__(self, form_type: str = "13F-HR", series_id: Optional[str] = None,
                 output_dir: str = "sec_filings", config: Optional[SECConfig] = None,
                 verbosity: Optional[VerbosityLevel] = None):
        """Initialize manager with appropriate strategies.
        
        Args:
            form_type: Form type to download (e.g., "13F-HR", "NPORT-P")
            series_id: Series ID for NPORT filings (optional)
            output_dir: Directory to save downloaded files
            config: Custom configuration (optional, defaults created if None)
            verbosity: Output verbosity level (overrides config if provided)
        """
        if form_type not in self.STRATEGIES:
            raise ValueError(f"Unsupported form type: {form_type}. Supported: {list(self.STRATEGIES.keys())}")
        
        self.form_type = form_type
        self.config = config or SECConfig()
        
        # Allow verbosity override without creating new config
        if verbosity is not None:
            self.config.verbosity = verbosity
        
        # Create HTTP client and strategies
        http_client = DefaultHTTPClient(self.config)
        downloader_cls, parser_cls = self.STRATEGIES[form_type]
        
        # Create downloader strategy with appropriate parameters
        if form_type in ['NPORT-P', 'NPORT-N']:
            download_strategy = downloader_cls(http_client, self.config, series_id)
        else:
            download_strategy = downloader_cls(http_client, self.config)
        
        parse_strategy = parser_cls(http_client, self.config)
        
        # Create composed objects
        self.downloader = SECFilingDownloader(download_strategy, self.config, output_dir)
        self.parser = SECFilingParser(parse_strategy)
        self.filings = {}
    
    def download_and_parse(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Download and parse filings in one operation.
        
        This is the main method for retrieving filing data.
        
        Returns:
            Dictionary keyed by (cik, period_date) with values of (metadata, holdings_df)
        """
        paths = self.downloader.download(cik, self.form_type, num_reports)
        self.filings = self.parser.parse_multiple(paths)
        return self.filings
    
    def get_filing_by_date(self, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date from stored filings.
        
        Args:
            date_str: Date string in any pandas-parseable format (e.g., '2024-09-30')
            
        Returns:
            Tuple of ((cik, period_date), (metadata, holdings_df))
            
        Raises:
            KeyError: If no filing found for the given date
        """
        target = pd.to_datetime(date_str).date()
        for (cik, period_date), value in self.filings.items():
            if pd.notna(period_date) and period_date.date() == target:
                return ((cik, period_date), value)
        raise KeyError(f"No filing found for date {date_str}")


# ============================================================================
# LOCAL FILE PARSER
# ============================================================================

class LocalFileParser:
    """Parse SEC filings from local XML files without downloading.
    
    Useful for:
    - Analyzing previously downloaded filings
    - Testing with sample files
    - Batch processing existing archives
    
    Example:
        parser = LocalFileParser(form_type="13F-HR")
        
        # Single file
        metadata, holdings = parser.parse_file("path/to/filing_dir")
        
        # Multiple directories
        results = parser.parse_directories(["dir1", "dir2", "dir3"])
    """
    
    def __init__(self, form_type: str = "13F-HR", config: Optional[SECConfig] = None):
        """Initialize parser for specified form type.
        
        Args:
            form_type: Form type - "13F-HR", "NPORT-P", etc.
            config: Optional config (uses default if None)
        """
        if form_type not in SECFilingManager.STRATEGIES:
            raise ValueError(f"Unsupported form type: {form_type}")
        
        self.form_type = form_type
        self.config = config or SECConfig(verbosity=VerbosityLevel.NORMAL)
        
        # Create parser strategy (no HTTP needed for local files)
        http_client = DefaultHTTPClient(self.config)
        _, parser_cls = SECFilingManager.STRATEGIES[form_type]
        self.strategy = parser_cls(http_client, self.config)
    
    def parse_file(self, path: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse single filing directory.
        
        Args:
            path: Path to filing directory containing XML files
            
        Returns:
            Tuple of (metadata_dict, holdings_dataframe)
        """
        filing_path = Path(path)
        if not filing_path.exists():
            raise FileNotFoundError(f"Directory not found: {path}")
        
        log(f"Parsing {filing_path}", VerbosityLevel.VERBOSE, self.config)
        
        metadata = self.strategy.parse_metadata(filing_path / "primary_doc.xml")
        holdings = self.strategy.parse_holdings(filing_path)
        
        log(f"✓ Found {len(holdings)} holdings", VerbosityLevel.NORMAL, self.config)
        
        return metadata, holdings
    
    def parse_files(self, paths: List[str]) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse multiple filing directories.
        
        Args:
            paths: List of paths to filing directories
            
        Returns:
            Dictionary keyed by (cik, period_date)
        """
        results = {}
        
        for path in paths:
            try:
                metadata, holdings = self.parse_file(path)
                
                cik = metadata.get('cik', 'unknown')
                period_str = metadata.get('periodofreport', 'unknown')
                
                try:
                    period_date = pd.to_datetime(period_str)
                except:
                    period_date = pd.NaT
                
                results[(cik, period_date)] = (metadata, holdings)
            except Exception as e:
                log(f"✗ {path}: {e}", VerbosityLevel.ERROR, self.config)
        
        log(f"Parsed {len(results)} filing(s)", VerbosityLevel.NORMAL, self.config, "\n✓ ")
        return results
    
    def parse_directory(self, directory: str, pattern: str = "*") -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse all filing subdirectories in a directory.
        
        Args:
            directory: Parent directory containing filing subdirectories
            pattern: Glob pattern for subdirectory names (default: "*")
            
        Returns:
            Dictionary keyed by (cik, period_date)
            
        Example:
            # Parse all filings in sec_filings/
            results = parser.parse_directory("sec_filings")
            
            # Parse only specific CIK
            results = parser.parse_directory("sec_filings", "0001037389_*")
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all subdirectories matching pattern
        subdirs = [str(p) for p in dir_path.glob(pattern) if p.is_dir()]
        
        if not subdirs:
            log(f"No subdirectories found matching '{pattern}' in {directory}", 
                VerbosityLevel.NORMAL, self.config)
            return {}
        
        log(f"Found {len(subdirs)} filing(s) in {directory}", VerbosityLevel.NORMAL, self.config)
        return self.parse_files(subdirs)
    
    def parse_directories(self, directories: List[str], pattern: str = "*") -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse all filings from multiple parent directories.
        
        Args:
            directories: List of parent directories
            pattern: Glob pattern for subdirectory names
            
        Returns:
            Dictionary keyed by (cik, period_date)
            
        Example:
            results = parser.parse_directories([
                "archive_2023",
                "archive_2024",
                "archive_2025"
            ])
        """
        all_results = {}
        
        for directory in directories:
            try:
                results = self.parse_directory(directory, pattern)
                all_results.update(results)
            except Exception as e:
                log(f"✗ {directory}: {e}", VerbosityLevel.ERROR, self.config)
        
        log(f"Total: {len(all_results)} filing(s) from {len(directories)} directories", 
            VerbosityLevel.NORMAL, self.config, "\n✓ ")
        return all_results
    
    def get_filing_by_date(self, filings: Dict, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date from parsed filings.
        
        Convenience method that wraps the module-level function.
        """
        return get_filing_by_date(filings, date_str)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_filing_by_date(filings: Dict, date_str: str) -> Tuple:
    """Get filing by date from filings dictionary.
    
    Standalone function for when you have a filings dict but no manager instance.
    
    Args:
        filings: Dictionary returned by download_and_parse() or parse_multiple()
        date_str: Date string to search for
        
    Returns:
        Tuple of ((cik, period_date), (metadata, holdings_df))
    """
    target = pd.to_datetime(date_str).date()
    for (cik, period_date), value in filings.items():
        if pd.notna(period_date) and period_date.date() == target:
            return ((cik, period_date), value)
    raise KeyError(f"No filing found for date {date_str}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: 13-F filing with verbose output
    print("13-F: Renaissance Technologies")
    print("=" * 60)
    mgr = SECFilingManager(
        form_type="13F-HR",
        verbosity=VerbosityLevel.VERBOSE  # Show detailed progress
    )
    filings = mgr.download_and_parse("0001037389", num_reports=1)
    
    if filings:
        key, (metadata, holdings) = list(filings.items())[0]
        print(f"\nRetrieved {len(holdings)} holdings for {metadata.get('name')}")
    
    # Example 2: NPORT-P with series filtering and normal output
    print("\n\nNPORT-P: Vanguard 500 Index Fund")
    print("=" * 60)
    mgr = SECFilingManager(
        form_type="NPORT-P",
        series_id="S000002839",  # Specific series
        verbosity=VerbosityLevel.NORMAL  # Standard output
    )
    filings = mgr.download_and_parse("0000036405", num_reports=1)
    
    if filings:
        key, (metadata, holdings) = list(filings.items())[0]
        print(f"\nRetrieved {len(holdings)} holdings for {metadata.get('name')}")
        print(f"Series ID: {metadata.get('seriesid')}")
    
    # Example 3: Silent mode (no output except errors)
    print("\n\nSilent mode example:")
    print("=" * 60)
    mgr_silent = SECFilingManager(
        form_type="13F-HR",
        verbosity=VerbosityLevel.SILENT
    )
    # This will run with no output
    filings_silent = mgr_silent.download_and_parse("0001037389", num_reports=1)
    print(f"Downloaded {len(filings_silent)} filing(s) silently")