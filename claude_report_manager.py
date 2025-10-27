"""
SEC Filing Downloader and Parser Module

Improved design with better separation of concerns, testability, and extensibility.
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


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class SECConfig:
    """Configuration for SEC API interactions."""
    user_agent: str = "Research Tool research@example.com"
    base_url: str = "https://www.sec.gov"
    data_url: str = "https://data.sec.gov"
    rate_limit_delay: float = 0.2
    request_delay: float = 0.1
    timeout: int = 30


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
# HTTP CLIENT PROTOCOL
# ============================================================================

class HTTPClient(Protocol):
    """Protocol for HTTP operations (enables dependency injection)."""
    
    def get(self, url: str, timeout: int = 30) -> requests.Response:
        """Make HTTP GET request."""
        ...


class DefaultHTTPClient:
    """Default HTTP client implementation."""
    
    def __init__(self, config: SECConfig):
        self.config = config
        self.headers = {"User-Agent": config.user_agent}
    
    def get(self, url: str, timeout: int = None) -> requests.Response:
        """Make HTTP GET request with configured headers."""
        timeout = timeout or self.config.timeout
        response = requests.get(url, headers=self.headers, timeout=timeout)
        response.raise_for_status()
        return response


# ============================================================================
# XML UTILITIES
# ============================================================================

class XMLParser:
    """Utility class for XML parsing operations."""
    
    @staticmethod
    def remove_namespaces(root: ET.Element) -> ET.Element:
        """Remove XML namespaces from element tree."""
        for elem in root.iter():
            if '}' in elem.tag:
                elem.tag = elem.tag.split('}', 1)[1]
        return root
    
    @staticmethod
    def extract_hierarchical_data(root: ET.Element) -> Dict[str, str]:
        """Extract all leaf node values with hierarchical keys."""
        data = {}
        
        def traverse(element: ET.Element, path: List[str] = []):
            for child in element:
                tag = child.tag.split('}')[-1]
                current_path = path + [tag]
                
                if len(child) == 0 and child.text and child.text.strip():
                    key = '_'.join(current_path).lower()
                    data[key] = child.text.strip()
                else:
                    traverse(child, current_path)
        
        traverse(root)
        return data
    
    @staticmethod
    def find_elements(root: ET.Element, tag: str) -> List[ET.Element]:
        """Find all elements with given tag (namespace-agnostic)."""
        # Try with namespace wildcard
        elements = root.findall(f'.//{{{root.tag.split("}")[0][1:]}}}{tag}')
        if not elements:
            # Try without namespace
            elements = root.findall(f'.//{tag}')
        if not elements:
            # Try with wildcard
            elements = root.findall(f'.//*')
            elements = [e for e in elements if e.tag.endswith(tag)]
        return elements


# ============================================================================
# FILE DISCOVERY
# ============================================================================

class FilingFileDiscoverer:
    """Discovers XML files in SEC filings."""
    
    def __init__(self, http_client: HTTPClient, config: SECConfig):
        self.http_client = http_client
        self.config = config
    
    def discover_xml_files(self, archive_base: str) -> List[str]:
        """Discover all XML files in a filing."""
        # Try index.json first
        xml_files = self._try_index_json(archive_base)
        if xml_files:
            return xml_files
        
        # Try index.html
        xml_files = self._try_index_html(archive_base)
        if xml_files:
            return xml_files
        
        # Fallback to common names
        return ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"]
    
    def _try_index_json(self, archive_base: str) -> Optional[List[str]]:
        """Try to get file list from index.json."""
        try:
            response = self.http_client.get(f"{archive_base}/index.json")
            index_data = response.json()
            return [
                item['name'] for item in index_data['directory']['item']
                if item['name'].endswith('.xml')
            ]
        except:
            return None
    
    def _try_index_html(self, archive_base: str) -> Optional[List[str]]:
        """Try to get file list from index.html."""
        try:
            response = self.http_client.get(f"{archive_base}/index.html")
            soup = BeautifulSoup(response.text, 'html.parser')
            xml_files = []
            for link in soup.find_all('a'):
                href = link.get('href', '')
                if href.endswith('.xml'):
                    filename = href.split('/')[-1]
                    if filename not in xml_files:
                        xml_files.append(filename)
            return xml_files if xml_files else None
        except:
            return None


# ============================================================================
# FILING METADATA
# ============================================================================

@dataclass
class FilingMetadata:
    """Metadata about a filing."""
    cik: str
    accession_number: str
    filing_date: str
    form_type: str
    
    @property
    def accession_no_dash(self) -> str:
        """Accession number without dashes."""
        return self.accession_number.replace("-", "")
    
    def archive_url(self, base_url: str) -> str:
        """Construct archive URL for this filing."""
        return f"{base_url}/Archives/edgar/data/{int(self.cik)}/{self.accession_no_dash}"


# ============================================================================
# DOWNLOADER STRATEGIES
# ============================================================================

class FilingDownloaderStrategy(ABC):
    """Abstract strategy for downloading different filing types."""
    
    def __init__(self, http_client: HTTPClient, config: SECConfig):
        self.http_client = http_client
        self.config = config
    
    @abstractmethod
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Determine if filing should be downloaded."""
        pass
    
    @abstractmethod
    def get_required_files(self, archive_url: str) -> List[str]:
        """Get list of required files for this filing type."""
        pass


class Filing13FDownloaderStrategy(FilingDownloaderStrategy):
    """Strategy for downloading 13-F filings."""
    
    def __init__(self, http_client: HTTPClient, config: SECConfig, 
                 file_discoverer: FilingFileDiscoverer):
        super().__init__(http_client, config)
        self.file_discoverer = file_discoverer
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """All 13-F filings are downloaded."""
        return True
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """Discover all XML files for 13-F filing."""
        return self.file_discoverer.discover_xml_files(archive_url)


class FilingNPORTDownloaderStrategy(FilingDownloaderStrategy):
    """Strategy for downloading NPORT filings with optional series filtering."""
    
    def __init__(self, http_client: HTTPClient, config: SECConfig, 
                 series_id: Optional[str] = None):
        super().__init__(http_client, config)
        self.series_id = series_id
        self.xml_parser = XMLParser()
    
    def should_download(self, filing: FilingMetadata, archive_url: str) -> bool:
        """Check if filing matches series ID filter."""
        if not self.series_id:
            return True
        
        try:
            # Fetch primary_doc.xml to check series
            response = self.http_client.get(f"{archive_url}/primary_doc.xml")
            root = ET.fromstring(response.content)
            root = self.xml_parser.remove_namespaces(root)
            
            series_elem = root.find('.//seriesId')
            if series_elem is not None and series_elem.text:
                return series_elem.text.strip() == self.series_id
            
            return False
        except Exception as e:
            print(f"  Warning: Could not check series ID: {e}")
            return False
    
    def get_required_files(self, archive_url: str) -> List[str]:
        """NPORT only needs primary_doc.xml."""
        return ["primary_doc.xml"]


# ============================================================================
# PARSER STRATEGIES
# ============================================================================

class FilingParserStrategy(ABC):
    """Abstract strategy for parsing different filing types."""
    
    def __init__(self, xml_parser: XMLParser):
        self.xml_parser = xml_parser
    
    @abstractmethod
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Parse metadata from primary document."""
        pass
    
    @abstractmethod
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse holdings data."""
        pass
    
    @abstractmethod
    def add_metadata_aliases(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """Add convenient aliases to metadata."""
        pass


class Filing13FParserStrategy(FilingParserStrategy):
    """Strategy for parsing 13-F filings."""
    
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Parse 13-F primary_doc.xml."""
        if not xml_path.exists():
            return {}
        
        try:
            tree = ET.parse(xml_path)
            root = self.xml_parser.remove_namespaces(tree.getroot())
            metadata = self.xml_parser.extract_hierarchical_data(root)
            return self.add_metadata_aliases(metadata)
        except ET.ParseError as e:
            raise ParsingError(f"Failed to parse 13-F metadata: {e}")
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse 13-F holdings from separate XML file."""
        holdings_files = (list(filing_path.glob("*holding*.xml")) + 
                         list(filing_path.glob("infotable.xml")))
        
        if not holdings_files:
            return pd.DataFrame()
        
        return self._parse_holdings_xml(holdings_files[0])
    
    def _parse_holdings_xml(self, xml_path: Path) -> pd.DataFrame:
        """Parse 13-F holdings XML file."""
        try:
            tree = ET.parse(xml_path)
            root = self.xml_parser.remove_namespaces(tree.getroot())
            info_tables = self.xml_parser.find_elements(root, 'infoTable')
            
            if not info_tables:
                return pd.DataFrame()
            
            holdings = []
            for table in info_tables:
                holding = self.xml_parser.extract_hierarchical_data(table)
                if holding:
                    holdings.append(holding)
            
            df = pd.DataFrame(holdings)
            return self._post_process_holdings(df)
        except Exception as e:
            raise ParsingError(f"Failed to parse 13-F holdings: {e}")
    
    def _post_process_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns and calculate derived values."""
        if df.empty:
            return df
        
        # Convert numeric columns
        for col in df.columns:
            if any(nc in col for nc in ['value', 'sshprnamt']):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value
        if 'value' in df.columns and 'shrsorprnamt_sshprnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['shrsorprnamt_sshprnamt'])
        
        return df
    
    def add_metadata_aliases(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """Add convenient aliases for 13-F metadata."""
        aliases = {
            'cik': 'headerdata_filerinfo_filer_credentials_cik',
            'periodofreport': 'headerdata_filerinfo_periodofreport',
            'name': 'formdata_coverpage_filingmanager_name'
        }
        
        for alias, full_key in aliases.items():
            if full_key in metadata:
                metadata[alias] = metadata[full_key]
        
        return metadata


class FilingNPORTParserStrategy(FilingParserStrategy):
    """Strategy for parsing NPORT filings."""
    
    def parse_metadata(self, xml_path: Path) -> Dict[str, str]:
        """Parse NPORT primary_doc.xml."""
        if not xml_path.exists():
            return {}
        
        try:
            tree = ET.parse(xml_path)
            root = self.xml_parser.remove_namespaces(tree.getroot())
            metadata = self.xml_parser.extract_hierarchical_data(root)
            return self.add_metadata_aliases(metadata)
        except ET.ParseError as e:
            raise ParsingError(f"Failed to parse NPORT metadata: {e}")
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse NPORT holdings from primary_doc.xml."""
        xml_path = filing_path / "primary_doc.xml"
        if not xml_path.exists():
            return pd.DataFrame()
        
        return self._parse_holdings_from_primary(xml_path)
    
    def _parse_holdings_from_primary(self, xml_path: Path) -> pd.DataFrame:
        """Parse NPORT holdings from single file."""
        try:
            tree = ET.parse(xml_path)
            root = self.xml_parser.remove_namespaces(tree.getroot())
            holdings_elements = self.xml_parser.find_elements(root, 'invstOrSec')
            
            if not holdings_elements:
                return pd.DataFrame()
            
            holdings = []
            for sec in holdings_elements:
                holding = self.xml_parser.extract_hierarchical_data(sec)
                if holding:
                    holdings.append(holding)
            
            df = pd.DataFrame(holdings)
            return self._post_process_holdings(df)
        except Exception as e:
            raise ParsingError(f"Failed to parse NPORT holdings: {e}")
    
    def _post_process_holdings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert numeric columns and calculate derived values."""
        if df.empty:
            return df
        
        # Convert numeric columns
        for col in df.columns:
            if any(nc in col for nc in ['valusd', 'balance', 'pctval']):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value
        if 'valusd' in df.columns and 'balance' in df.columns:
            df = df.assign(unitValue=lambda x: x['valusd'] / x['balance'])
        
        return df
    
    def add_metadata_aliases(self, metadata: Dict[str, str]) -> Dict[str, str]:
        """Add convenient aliases for NPORT metadata."""
        aliases = {
            'cik': 'headerdata_filerinfo_filer_issuercredentials_cik',
            'periodofreport': 'formdata_geninfo_reppdend',
            'name': 'formdata_geninfo_seriesname',
            'seriesid': 'headerdata_filerinfo_seriesclassinfo_seriesid'
        }
        
        for alias, full_key in aliases.items():
            if full_key in metadata:
                metadata[alias] = metadata[full_key]
        
        return metadata


# ============================================================================
# DOWNLOADER
# ============================================================================

class SECFilingDownloader:
    """Downloads SEC filings using pluggable strategy."""
    
    def __init__(self, strategy: FilingDownloaderStrategy, 
                 http_client: HTTPClient, config: SECConfig, output_dir: str = "sec_filings"):
        self.strategy = strategy
        self.http_client = http_client
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def download(self, cik: str, form_type: str, num_reports: int = 5) -> List[str]:
        """Download filings for specified CIK."""
        cik = cik.strip().replace('-', '').zfill(10)
        
        print(f"Fetching {form_type} filings for CIK {cik}...")
        
        # Get filing list
        filings = self._fetch_filing_list(cik, form_type)
        
        if not filings:
            print(f"No {form_type} filings found")
            return []
        
        print(f"Found {len(filings)} {form_type} filing(s), checking filters...")
        
        # Download filings that pass filter
        downloaded_paths = []
        for filing in filings:
            if len(downloaded_paths) >= num_reports:
                break
            
            archive_url = filing.archive_url(self.config.base_url)
            
            if not self.strategy.should_download(filing, archive_url):
                print(f"  Skipping {filing.filing_date} (filter)")
                continue
            
            path = self._download_filing(filing, archive_url)
            if path:
                downloaded_paths.append(path)
            
            time.sleep(self.config.rate_limit_delay)
        
        print(f"\n✓ Downloaded {len(downloaded_paths)} filing(s)")
        return downloaded_paths
    
    def _fetch_filing_list(self, cik: str, form_type: str) -> List[FilingMetadata]:
        """Fetch list of filings from SEC API."""
        url = f"{self.config.data_url}/submissions/CIK{cik}.json"
        
        try:
            response = self.http_client.get(url)
            data = response.json()
            
            filings = data.get("filings", {}).get("recent", {})
            
            result = []
            for i, form in enumerate(filings.get("form", [])):
                if form == form_type:
                    result.append(FilingMetadata(
                        cik=cik,
                        accession_number=filings["accessionNumber"][i],
                        filing_date=filings["filingDate"][i],
                        form_type=form
                    ))
            
            return result
        except Exception as e:
            raise FilingNotFoundError(f"Failed to fetch filings: {e}")
    
    def _download_filing(self, filing: FilingMetadata, archive_url: str) -> Optional[str]:
        """Download a single filing."""
        filing_dir = self.output_dir / f"{filing.cik}_{filing.accession_no_dash}"
        filing_dir.mkdir(exist_ok=True)
        
        print(f"\nDownloading {filing.filing_date} ({filing.accession_number})...")
        
        files = self.strategy.get_required_files(archive_url)
        downloaded = 0
        
        for filename in files:
            if self._download_file(f"{archive_url}/{filename}", filing_dir / filename):
                print(f"  ✓ {filename}")
                downloaded += 1
            time.sleep(self.config.request_delay)
        
        return str(filing_dir) if downloaded > 0 else None
    
    def _download_file(self, url: str, filepath: Path) -> bool:
        """Download single file."""
        try:
            response = self.http_client.get(url)
            filepath.write_bytes(response.content)
            return True
        except:
            return False


# ============================================================================
# PARSER
# ============================================================================

class SECFilingParser:
    """Parses SEC filings using pluggable strategy."""
    
    def __init__(self, strategy: FilingParserStrategy):
        self.strategy = strategy
    
    def parse_filing(self, filing_dir: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse single filing."""
        filing_path = Path(filing_dir)
        
        metadata = self.strategy.parse_metadata(filing_path / "primary_doc.xml")
        holdings = self.strategy.parse_holdings(filing_path)
        
        return metadata, holdings
    
    def parse_multiple(self, filing_dirs: List[str]) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse multiple filings."""
        results = {}
        
        for filing_dir in filing_dirs:
            try:
                metadata, holdings = self.parse_filing(filing_dir)
                
                cik = metadata.get('cik', 'unknown')
                period_str = metadata.get('periodofreport', 'unknown')
                
                try:
                    period_date = pd.to_datetime(period_str)
                except Exception:
                    period_date = pd.NaT
                
                key = (cik, period_date)
                results[key] = (metadata, holdings)
                print(f"✓ {filing_dir}: {len(holdings)} holdings")
            except Exception as e:
                print(f"✗ {filing_dir}: {e}")
        
        return results


# ============================================================================
# MANAGER (FACADE)
# ============================================================================

class SECFilingManager:
    """Facade for SEC filing operations."""
    
    def __init__(self, form_type: str = "13F-HR", series_id: Optional[str] = None,
                 output_dir: str = "sec_filings", config: Optional[SECConfig] = None):
        """Initialize with appropriate strategies for form type."""
        self.form_type = form_type
        self.series_id = series_id
        self.config = config or SECConfig()
        
        # Create shared dependencies
        http_client = DefaultHTTPClient(self.config)
        xml_parser = XMLParser()
        
        # Create strategies based on form type
        if form_type == "13F-HR":
            file_discoverer = FilingFileDiscoverer(http_client, self.config)
            download_strategy = Filing13FDownloaderStrategy(http_client, self.config, file_discoverer)
            parse_strategy = Filing13FParserStrategy(xml_parser)
        elif form_type in ["NPORT-P", "NPORT-N"]:
            download_strategy = FilingNPORTDownloaderStrategy(http_client, self.config, series_id)
            parse_strategy = FilingNPORTParserStrategy(xml_parser)
        else:
            raise ValueError(f"Unsupported form type: {form_type}")
        
        # Create composed objects
        self.downloader = SECFilingDownloader(download_strategy, http_client, self.config, output_dir)
        self.parser = SECFilingParser(parse_strategy)
        self.filings = {}
    
    def download_and_parse(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Download and parse filings."""
        paths = self.downloader.download(cik, self.form_type, num_reports)
        self.filings = self.parser.parse_multiple(paths)
        return self.filings
    
    def get_filing_by_date(self, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date."""
        target_date = pd.to_datetime(date_str)
        
        for key, value in self.filings.items():
            cik, period_date = key
            if pd.notna(period_date) and period_date.date() == target_date.date():
                return (key, value)
        
        raise KeyError(f"No filing found for date {date_str}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def get_filing_by_date(filings_dict: Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]], 
                       date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
    """Get filing by date from filings dictionary."""
    target_date = pd.to_datetime(date_str)
    
    for key, value in filings_dict.items():
        cik, period_date = key
        if pd.notna(period_date) and period_date.date() == target_date.date():
            return (key, value)
    
    raise KeyError(f"No filing found for date {date_str}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: 13-F filing
    print("=" * 60)
    print("13-F: Renaissance Technologies")
    print("=" * 60)
    manager_13f = SECFilingManager(form_type="13F-HR")
    filings_13f = manager_13f.download_and_parse("0001037389", num_reports=2)
    
    # Example 2: NPORT-P with series ID
    print("\n" + "=" * 60)
    print("NPORT-P: Vanguard 500 Index Fund")
    print("=" * 60)
    manager_nport = SECFilingManager(form_type="NPORT-P", series_id="S000002839")
    filings_nport = manager_nport.download_and_parse("0000036405", num_reports=2)
    
    if filings_nport:
        key, (metadata, holdings) = list(filings_nport.items())[0]
        print(f"\nResults:")
        print(f"  Series: {metadata.get('name')}")
        print(f"  Holdings: {len(holdings)}")