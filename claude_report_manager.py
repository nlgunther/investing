"""
SEC Filing Downloader and Parser Module

Uses composition to support different filing types (13-F and NPORT-P).
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Tuple, Optional, List
from abc import ABC, abstractmethod


class FilingDownloaderStrategy(ABC):
    """Abstract strategy for downloading different filing types."""
    
    @abstractmethod
    def should_download_filing(self, cik: str, accession: str, **kwargs) -> bool:
        """Determine if a specific filing should be downloaded."""
        pass
    
    @abstractmethod
    def get_xml_files(self, archive_base: str) -> List[str]:
        """Get list of XML files to download for this filing type."""
        pass


class Filing13FDownloaderStrategy(FilingDownloaderStrategy):
    """Strategy for downloading 13-F filings."""
    
    def __init__(self, headers: Dict):
        self.headers = headers
    
    def should_download_filing(self, cik: str, accession: str, **kwargs) -> bool:
        """All 13-F filings are downloaded (no filtering needed)."""
        return True
    
    def get_xml_files(self, archive_base: str) -> List[str]:
        """Get XML files for 13-F filings."""
        return self._discover_xml_files(archive_base)
    
    def _discover_xml_files(self, archive_base: str) -> List[str]:
        """Discover all XML files in a filing."""
        xml_files = []
        
        # Try index.json
        try:
            index_response = requests.get(f"{archive_base}/index.json", headers=self.headers)
            if index_response.status_code == 200:
                index_data = index_response.json()
                xml_files = [
                    item['name'] for item in index_data['directory']['item']
                    if item['name'].endswith('.xml')
                ]
                return xml_files
        except:
            pass
        
        # Try parsing index.html
        try:
            html_response = requests.get(f"{archive_base}/index.html", headers=self.headers)
            if html_response.status_code == 200:
                soup = BeautifulSoup(html_response.text, 'html.parser')
                for link in soup.find_all('a'):
                    href = link.get('href', '')
                    if href.endswith('.xml'):
                        filename = href.split('/')[-1]
                        if filename not in xml_files:
                            xml_files.append(filename)
                return xml_files
            time.sleep(0.1)
        except:
            pass
        
        # Fallback
        return ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"]


class FilingNPORTDownloaderStrategy(FilingDownloaderStrategy):
    """Strategy for downloading NPORT-P filings with series filtering."""
    
    def __init__(self, headers: Dict, series_id: Optional[str] = None):
        self.headers = headers
        self.series_id = series_id
    
    def should_download_filing(self, cik: str, accession: str, **kwargs) -> bool:
        """Check if filing matches the desired series ID."""
        if not self.series_id:
            return True  # No filter, download all
        
        archive_base = kwargs.get('archive_base')
        if not archive_base:
            return False
        
        # Fetch primary_doc.xml to check series ID
        try:
            url = f"{archive_base}/primary_doc.xml"
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                # Parse XML to find series ID
                root = ET.fromstring(response.content)
                # Remove namespace
                for elem in root.iter():
                    if '}' in elem.tag:
                        elem.tag = elem.tag.split('}', 1)[1]
                
                # Find seriesId in the filing
                series_elem = root.find('.//seriesId')
                if series_elem is not None and series_elem.text:
                    filing_series_id = series_elem.text.strip()
                    return filing_series_id == self.series_id
            time.sleep(0.1)
        except Exception as e:
            print(f"  Error checking series ID: {e}")
        
        return False
    
    def get_xml_files(self, archive_base: str) -> List[str]:
        """For NPORT-P, only need primary_doc.xml (contains everything)."""
        return ["primary_doc.xml"]


class FilingParserStrategy(ABC):
    """Abstract strategy for parsing different filing types."""
    
    @abstractmethod
    def parse_metadata(self, xml_path: Path) -> Dict:
        """Parse metadata from primary document."""
        pass
    
    @abstractmethod
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse holdings data."""
        pass


class Filing13FParserStrategy(FilingParserStrategy):
    """Strategy for parsing 13-F filings."""
    
    def __init__(self, method: str = 'ET'):
        self.method = method
    
    def parse_metadata(self, xml_path: Path) -> Dict:
        """Parse 13-F primary_doc.xml metadata."""
        if not xml_path.exists():
            return {}
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        metadata = {}
        
        def extract_with_path(element, path=[]):
            for child in element:
                tag = child.tag.split('}')[-1]
                current_path = path + [tag]
                
                if len(child) == 0 and child.text and child.text.strip():
                    key = '_'.join(current_path).lower()
                    metadata[key] = child.text.strip()
                else:
                    extract_with_path(child, current_path)
        
        extract_with_path(root)
        
        # Add aliases
        if 'headerdata_filerinfo_filer_credentials_cik' in metadata:
            metadata['cik'] = metadata['headerdata_filerinfo_filer_credentials_cik']
        if 'headerdata_filerinfo_periodofreport' in metadata:
            metadata['periodofreport'] = metadata['headerdata_filerinfo_periodofreport']
        if 'formdata_coverpage_filingmanager_name' in metadata:
            metadata['name'] = metadata['formdata_coverpage_filingmanager_name']
        
        return metadata
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse 13-F holdings from separate holdings XML file."""
        # Find holdings file
        holdings_files = (list(filing_path.glob("*holding*.xml")) + 
                         list(filing_path.glob("infotable.xml")))
        
        if not holdings_files:
            return pd.DataFrame()
        
        return self._parse_holdings_xml(holdings_files[0])
    
    def _parse_holdings_xml(self, xml_path: Path) -> pd.DataFrame:
        """Parse 13-F holdings XML."""
        tree = ET.parse(xml_path)
        info_tables = tree.findall('.//{*}infoTable')
        
        if not info_tables:
            return pd.DataFrame()
        
        holdings = []
        for table in info_tables:
            holding = {}
            
            def extract_leaf_values(element, prefix=''):
                for child in element:
                    tag = child.tag.split('}')[-1]
                    
                    if len(child) == 0:
                        key = f"{prefix}_{tag}" if prefix else tag
                        if child.text and child.text.strip():
                            holding[key] = child.text.strip()
                    else:
                        new_prefix = f"{prefix}_{tag}" if prefix else tag
                        extract_leaf_values(child, new_prefix)
            
            extract_leaf_values(table)
            if holding:
                holdings.append(holding)
        
        df = pd.DataFrame(holdings)
        
        # Convert numeric columns
        for col in df.columns:
            if any(nc in col for nc in ['value', 'sshPrnamt']):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value
        if 'value' in df.columns and 'shrsOrPrnAmt_sshPrnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['shrsOrPrnAmt_sshPrnamt'])
        
        return df


class FilingNPORTParserStrategy(FilingParserStrategy):
    """Strategy for parsing NPORT-P filings."""
    
    def __init__(self, method: str = 'ET'):
        self.method = method
    
    def parse_metadata(self, xml_path: Path) -> Dict:
        """Parse NPORT-P primary_doc.xml metadata."""
        if not xml_path.exists():
            return {}
        
        tree = ET.parse(xml_path)
        root = tree.getroot()
        metadata = {}
        
        def extract_with_path(element, path=[]):
            for child in element:
                tag = child.tag.split('}')[-1]
                current_path = path + [tag]
                
                if len(child) == 0 and child.text and child.text.strip():
                    key = '_'.join(current_path).lower()
                    metadata[key] = child.text.strip()
                else:
                    extract_with_path(child, current_path)
        
        extract_with_path(root)
        
        # Add aliases for NPORT-P
        if 'headerdata_filerinfo_filer_issuercredentials_cik' in metadata:
            metadata['cik'] = metadata['headerdata_filerinfo_filer_issuercredentials_cik']
        if 'formdata_geninfo_reppdend' in metadata:
            metadata['periodofreport'] = metadata['formdata_geninfo_reppdend']
        if 'formdata_geninfo_seriesname' in metadata:
            metadata['name'] = metadata['formdata_geninfo_seriesname']
        if 'headerdata_filerinfo_seriesclassinfo_seriesid' in metadata:
            metadata['seriesid'] = metadata['headerdata_filerinfo_seriesclassinfo_seriesid']
        
        return metadata
    
    def parse_holdings(self, filing_path: Path) -> pd.DataFrame:
        """Parse NPORT-P holdings from primary_doc.xml (single file structure)."""
        xml_path = filing_path / "primary_doc.xml"
        
        if not xml_path.exists():
            return pd.DataFrame()
        
        return self._parse_holdings_from_primary(xml_path)
    
    def _parse_holdings_from_primary(self, xml_path: Path) -> pd.DataFrame:
        """Parse NPORT-P holdings from primary_doc.xml."""
        tree = ET.parse(xml_path)
        
        # Find all invstOrSec elements
        holdings_elements = tree.findall('.//{*}invstOrSec')
        
        if not holdings_elements:
            return pd.DataFrame()
        
        print(f"  Found {len(holdings_elements)} invstOrSec elements")
        
        holdings = []
        for sec in holdings_elements:
            holding = {}
            
            def extract_leaf_values(element, prefix=''):
                for child in element:
                    tag = child.tag.split('}')[-1]
                    
                    if len(child) == 0:
                        key = f"{prefix}_{tag}" if prefix else tag
                        if child.text and child.text.strip():
                            holding[key] = child.text.strip()
                    else:
                        new_prefix = f"{prefix}_{tag}" if prefix else tag
                        extract_leaf_values(child, new_prefix)
            
            extract_leaf_values(sec)
            if holding:
                holdings.append(holding)
        
        df = pd.DataFrame(holdings)
        
        # Convert numeric columns
        for col in df.columns:
            if any(nc in col for nc in ['valUSD', 'balance', 'pctVal']):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value
        if 'valUSD' in df.columns and 'balance' in df.columns:
            df = df.assign(unitValue=lambda x: x['valUSD'] / x['balance'])
        
        return df


class SECFilingDownloader:
    """Generic SEC filing downloader using composition."""
    
    def __init__(self, strategy: FilingDownloaderStrategy, output_dir: str = "sec_filings", 
                 user_agent: str = "Research Tool research@example.com"):
        self.strategy = strategy
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {"User-Agent": user_agent}
    
    def download(self, cik: str, form_type: str, num_reports: int = 5) -> List[str]:
        """Download filings using the configured strategy."""
        cik = cik.strip().replace('-', '').zfill(10)
        
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"Fetching {form_type} filings for CIK {cik}...")
        
        response = requests.get(filings_url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        filings = data.get("filings", {}).get("recent", {})
        form_indices = [
            i for i, form in enumerate(filings.get("form", []))
            if form == form_type
        ]
        
        if not form_indices:
            print(f"No {form_type} filings found for CIK {cik}")
            return []
        
        print(f"Found {len(form_indices)} {form_type} filing(s), checking filters...")
        
        downloaded_paths = []
        checked = 0
        
        for idx in form_indices:
            if len(downloaded_paths) >= num_reports:
                break
            
            checked += 1
            accession = filings["accessionNumber"][idx]
            accession_no_dash = accession.replace("-", "")
            filing_date = filings["filingDate"][idx]
            archive_base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}"
            
            # Check if this filing should be downloaded
            if not self.strategy.should_download_filing(
                cik, accession, archive_base=archive_base
            ):
                print(f"  Skipping filing {filing_date} (doesn't match filter)")
                continue
            
            filing_dir = self.output_dir / f"{cik}_{accession_no_dash}"
            filing_dir.mkdir(exist_ok=True)
            
            print(f"\nDownloading filing {filing_date} (Accession: {accession})...")
            
            xml_files = self.strategy.get_xml_files(archive_base)
            
            downloaded_count = 0
            for filename in xml_files:
                if self._download_file(f"{archive_base}/{filename}", filing_dir / filename):
                    print(f"  ✓ Downloaded {filename}")
                    downloaded_count += 1
                time.sleep(0.1)
            
            if downloaded_count > 0:
                downloaded_paths.append(str(filing_dir))
                print(f"  Total: {downloaded_count} file(s) downloaded")
            
            time.sleep(0.2)
        
        print(f"\n✓ Downloaded {len(downloaded_paths)} filing(s) (checked {checked}) to {self.output_dir}/")
        return downloaded_paths
    
    def _download_file(self, url: str, filepath: Path) -> bool:
        """Download a single file."""
        try:
            response = requests.get(url, headers=self.headers)
            if response.status_code == 200:
                filepath.write_bytes(response.content)
                return True
        except:
            pass
        return False


class SECFilingParser:
    """Generic SEC filing parser using composition."""
    
    def __init__(self, strategy: FilingParserStrategy):
        self.strategy = strategy
    
    def parse_filing(self, filing_dir: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse a filing using the configured strategy."""
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
                
                print(f"  Parsed metadata - CIK: {cik}, Period: {period_str}, Holdings: {len(holdings)}")
                
                try:
                    period_date = pd.to_datetime(period_str)
                except Exception as e:
                    print(f"  Warning: Could not parse date '{period_str}': {e}")
                    period_date = pd.NaT
                
                key = (cik, period_date)
                results[key] = (metadata, holdings)
                print(f"✓ {filing_dir}: {len(holdings)} holdings")
            except Exception as e:
                print(f"✗ {filing_dir}: {e}")
                import traceback
                traceback.print_exc()
        
        return results


class SECFilingManager:
    """High-level manager for SEC filing operations."""
    
    def __init__(self, form_type: str = "13F-HR", series_id: Optional[str] = None,
                 output_dir: str = "sec_filings", parse_method: str = 'ET'):
        """Initialize manager with appropriate strategies.
        
        Args:
            form_type: Form type - "13F-HR" or "NPORT-P"
            series_id: Series ID for NPORT-P filings (optional)
            output_dir: Directory for downloaded files
            parse_method: Parsing method (currently only 'ET' supported)
        """
        self.form_type = form_type
        self.series_id = series_id
        
        headers = {"User-Agent": "Research Tool research@example.com"}
        
        # Create appropriate strategies based on form type
        if form_type == "13F-HR":
            download_strategy = Filing13FDownloaderStrategy(headers)
            parse_strategy = Filing13FParserStrategy(parse_method)
        elif form_type in ["NPORT-P", "NPORT-N"]:
            download_strategy = FilingNPORTDownloaderStrategy(headers, series_id)
            parse_strategy = FilingNPORTParserStrategy(parse_method)
        else:
            raise ValueError(f"Unsupported form type: {form_type}")
        
        self.downloader = SECFilingDownloader(download_strategy, output_dir)
        self.parser = SECFilingParser(parse_strategy)
        self.filings = {}
    
    def download_and_parse(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Download and parse filings."""
        paths = self.downloader.download(cik, self.form_type, num_reports)
        self.filings = self.parser.parse_multiple(paths)
        return self.filings
    
    def get_filing_by_date(self, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date string."""
        target_date = pd.to_datetime(date_str)
        
        for key, value in self.filings.items():
            cik, period_date = key
            if pd.notna(period_date) and period_date.date() == target_date.date():
                return (key, value)
        
        raise KeyError(f"No filing found for date {date_str}")


# Convenience function
def get_filing_by_date(filings_dict: Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]], 
                       date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
    """Get filing by date string from a filings dictionary."""
    target_date = pd.to_datetime(date_str)
    
    for key, value in filings_dict.items():
        cik, period_date = key
        if pd.notna(period_date) and period_date.date() == target_date.date():
            return (key, value)
    
    raise KeyError(f"No filing found for date {date_str}")


# Example usage
if __name__ == "__main__":
    # Example 1: 13-F filing (hedge fund)
    print("=" * 60)
    print("13-F: Renaissance Technologies")
    print("=" * 60)
    manager_13f = SECFilingManager(form_type="13F-HR")
    filings_13f = manager_13f.download_and_parse("0001037389", num_reports=2)
    
    # Example 2: NPORT-P with series ID (mutual fund)
    print("\n" + "=" * 60)
    print("NPORT-P: Vanguard 500 Index Fund (specific series)")
    print("=" * 60)
    manager_nport = SECFilingManager(
        form_type="NPORT-P",
        series_id="S000002839"  # Vanguard 500 Index Fund
    )
    filings_nport = manager_nport.download_and_parse("0000036405", num_reports=2)
    
    # Display results
    if filings_nport:
        key, (metadata, holdings) = list(filings_nport.items())[0]
        print(f"\nVanguard 500 Index Fund details:")
        print(f"  CIK: {key[0]}")
        print(f"  Period: {key[1]}")
        print(f"  Series: {metadata.get('name', 'N/A')}")
        print(f"  Series ID: {metadata.get('seriesid', 'N/A')}")
        print(f"  Holdings: {len(holdings)}")
        if not holdings.empty and 'n' in holdings.columns:
            print(f"  Sample holdings: {holdings['n'].head(3).tolist()}")