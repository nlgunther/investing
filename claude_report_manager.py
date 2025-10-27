"""
SEC Filing Downloader and Parser Module

Supports Form 13-F (hedge funds) and Form NPORT-P (mutual funds).
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Tuple, Optional


class SECFilingDownloader:
    """Downloads SEC filings (13-F, NPORT-P, etc.)."""
    
    def __init__(self, output_dir: str = "sec_filings", user_agent: str = "Research Tool research@example.com"):
        """Initialize downloader.
        
        Args:
            output_dir: Directory to save downloaded files
            user_agent: User agent string for SEC requests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.headers = {"User-Agent": user_agent}
    
    def download(self, cik: str, form_type: str = "13F-HR", num_reports: int = 5) -> list[str]:
        """Download the most recent filings for a specified entity.
        
        Args:
            cik: The entity's CIK identifier
            form_type: The form type - "13F-HR", "NPORT-P", etc.
            num_reports: Number of most recent reports to download
        
        Returns:
            List of paths to downloaded filing directories
        """
        # Normalize CIK to 10 digits with leading zeros
        cik = cik.strip().replace('-', '').zfill(10)
        
        # Get company filings metadata
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"Fetching {form_type} filings for CIK {cik}...")
        
        response = requests.get(filings_url, headers=self.headers)
        response.raise_for_status()
        data = response.json()
        
        # Extract recent filings of the specified type
        filings = data.get("filings", {}).get("recent", {})
        form_indices = [
            i for i, form in enumerate(filings.get("form", []))
            if form == form_type
        ][:num_reports]
        
        if not form_indices:
            print(f"No {form_type} filings found for CIK {cik}")
            return []
        
        print(f"Found {len(form_indices)} {form_type} filing(s)")
        
        downloaded_paths = []
        
        for idx in form_indices:
            accession = filings["accessionNumber"][idx]
            accession_no_dash = accession.replace("-", "")
            filing_date = filings["filingDate"][idx]
            
            filing_dir = self.output_dir / f"{cik}_{accession_no_dash}"
            filing_dir.mkdir(exist_ok=True)
            
            print(f"\nDownloading {form_type} filing {filing_date} (Accession: {accession})...")
            
            archive_base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_no_dash}"
            
            # Get index to find all XML files
            xml_files = self._discover_xml_files(archive_base)
            
            downloaded_count = 0
            for filename in xml_files:
                if self._download_file(f"{archive_base}/{filename}", filing_dir / filename):
                    if 'holding' in filename.lower() or 'infotable' in filename.lower():
                        print(f"  ✓ Downloaded {filename} (holdings file)")
                    else:
                        print(f"  ✓ Downloaded {filename}")
                    downloaded_count += 1
                time.sleep(0.1)
            
            if downloaded_count > 0:
                downloaded_paths.append(str(filing_dir))
                print(f"  Total: {downloaded_count} file(s) downloaded")
            
            time.sleep(0.2)
        
        print(f"\n✓ Downloaded {len(downloaded_paths)} filing(s) to {self.output_dir}/")
        return downloaded_paths
    
    def _discover_xml_files(self, archive_base: str) -> list[str]:
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
        
        # Fallback to common filenames for both 13-F and NPORT-P
        return ["primary_doc.xml", "form13fInfoTable.xml", "infotable.xml"]
    
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
    """Parses SEC filings (13-F and NPORT-P)."""
    
    def __init__(self, method: str = 'ET'):
        """Initialize parser.
        
        Args:
            method: Parsing method - 'ET' (default) or 'pandas' for read_xml
        """
        self.method = method
    
    def parse_filing(self, filing_dir: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse a single filing directory.
        
        Args:
            filing_dir: Path to filing directory
            
        Returns:
            Tuple of (metadata_dict, holdings_dataframe)
        """
        filing_path = Path(filing_dir)
        
        metadata = self._parse_primary_doc(filing_path / "primary_doc.xml")
        
        # Find holdings file (works for both 13-F and NPORT-P)
        holdings_files = (list(filing_path.glob("*holding*.xml")) + 
                         list(filing_path.glob("infotable.xml")))
        holdings_df = self._parse_holdings_xml(holdings_files[0]) if holdings_files else pd.DataFrame()
        
        return metadata, holdings_df
    
    def parse_multiple(self, filing_dirs: list[str]) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse multiple filings.
        
        Args:
            filing_dirs: List of filing directory paths
            
        Returns:
            Dictionary with keys as (cik, period_date) tuples and values as (metadata, holdings) tuples
        """
        results = {}
        
        for filing_dir in filing_dirs:
            try:
                metadata, holdings = self.parse_filing(filing_dir)
                
                cik = metadata.get('cik', 'unknown')
                # Try both 13-F and NPORT-P period field names
                period_str = (metadata.get('periodofreport') or 
                             metadata.get('reppdend') or 
                             metadata.get('unknown'))
                
                print(f"  Parsed metadata - CIK: {cik}, Period string: {period_str}")
                
                try:
                    period_date = pd.to_datetime(period_str)
                except Exception as e:
                    print(f"  Warning: Could not parse date '{period_str}': {e}")
                    period_date = pd.NaT
                
                key = (cik, period_date)
                results[key] = (metadata, holdings)
                print(f"✓ {filing_dir}: {len(holdings)} holdings (Period: {period_date})")
            except Exception as e:
                print(f"✗ {filing_dir}: {e}")
        
        return results
    
    def parse_holdings_file(self, xml_file_path: str) -> pd.DataFrame:
        """Parse a holdings XML file directly (for testing/debugging).
        
        Args:
            xml_file_path: Direct path to holdings XML file
            
        Returns:
            DataFrame with holdings data
        """
        return self._parse_holdings_xml(Path(xml_file_path))
    
    def _parse_primary_doc(self, xml_path: Path) -> Dict:
        """Parse primary_doc.xml into a dictionary."""
        if not xml_path.exists():
            return {}
        
        if self.method == 'pandas':
            try:
                df = pd.read_xml(xml_path, xpath=".//*[not(*)]")
                if not df.empty:
                    metadata = df.iloc[0].to_dict()
                    metadata = {k.lower(): v for k, v in metadata.items() if pd.notna(v)}
                    return metadata
            except Exception as e:
                print(f"pandas read_xml failed for primary_doc: {e}")
                return {}
        
        # Use ET.parse with hierarchical keys
        tree = ET.parse(xml_path)
        root = tree.getroot()
        metadata = {}
        
        def extract_with_path(element, path=[]):
            """Recursively extract leaf values with full path as key."""
            for child in element:
                tag = child.tag.split('}')[-1]  # Remove namespace
                current_path = path + [tag]
                
                # If it's a leaf node (no children), store it with full path
                if len(child) == 0 and child.text and child.text.strip():
                    key = '_'.join(current_path).lower()
                    metadata[key] = child.text.strip()
                else:
                    extract_with_path(child, current_path)
        
        extract_with_path(root)
        
        # Add commonly used aliases for backward compatibility
        # Works for both 13-F and NPORT-P
        if 'headerdata_filerinfo_filer_credentials_cik' in metadata:
            metadata['cik'] = metadata['headerdata_filerinfo_filer_credentials_cik']
        if 'headerdata_filerinfo_periodofreport' in metadata:
            metadata['periodofreport'] = metadata['headerdata_filerinfo_periodofreport']
        if 'formdata_coverpage_filingmanager_name' in metadata:
            metadata['name'] = metadata['formdata_coverpage_filingmanager_name']
        # NPORT-P specific fields
        if 'formdata_geninfo_regcik' in metadata:
            metadata['cik'] = metadata['formdata_geninfo_regcik']
        if 'formdata_geninfo_reppdend' in metadata:
            metadata['reppdend'] = metadata['formdata_geninfo_reppdend']
        if 'formdata_geninfo_regname' in metadata:
            metadata['name'] = metadata['formdata_geninfo_regname']
        
        return metadata
    
    def _parse_holdings_xml(self, xml_path: Path) -> pd.DataFrame:
        """Parse holdings XML into a DataFrame (works for 13-F and NPORT-P)."""
        if not xml_path.exists():
            print(f"Warning: Holdings file not found at {xml_path}")
            return pd.DataFrame()
        
        print(f"  Parsing holdings file: {xml_path.name}")
        
        if self.method == 'pandas':
            # Try different xpath patterns for different filing types
            for xpath in [".//infoTable", ".//*[local-name()='infoTable']",
                         ".//invstOrSec", ".//*[local-name()='invstOrSec']"]:
                try:
                    df = pd.read_xml(xml_path, xpath=xpath)
                    if not df.empty:
                        break
                except:
                    continue
            else:
                print(f"  pandas read_xml failed for holdings")
                return pd.DataFrame()
        else:
            # Use ET.parse
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            print(f"  Root tag: {root.tag}")
            
            # Try to find holdings elements (13-F uses infoTable, NPORT-P uses invstOrSec)
            holdings_elements = (tree.findall('.//{*}infoTable') or 
                               tree.findall('.//infoTable') or
                               tree.findall('.//{*}invstOrSec') or
                               tree.findall('.//invstOrSec'))
            
            if not holdings_elements:
                print(f"  No holdings elements found in {xml_path}")
                return pd.DataFrame()
            
            print(f"  Found {len(holdings_elements)} holding elements")
            
            holdings = []
            for table in holdings_elements:
                holding = {}
                
                # Recursively extract all leaf values with flattened keys
                def extract_leaf_values(element, prefix=''):
                    for child in element:
                        tag = child.tag.split('}')[-1]  # Remove namespace
                        
                        # If child has no children, it's a leaf - store its text
                        if len(child) == 0:
                            key = f"{prefix}_{tag}" if prefix else tag
                            if child.text and child.text.strip():
                                holding[key] = child.text.strip()
                        else:
                            # Child has children, recurse with updated prefix
                            new_prefix = f"{prefix}_{tag}" if prefix else tag
                            extract_leaf_values(child, new_prefix)
                
                extract_leaf_values(table)
                
                if holding:
                    holdings.append(holding)
            
            if not holdings:
                print(f"  Warning: No holdings data extracted")
            
            df = pd.DataFrame(holdings)
        
        print(f"  Extracted {len(df)} holdings with {len(df.columns)} columns")
        if len(df.columns) > 0:
            print(f"  Columns: {', '.join(list(df.columns)[:10])}")
        
        # Convert numeric columns
        numeric_cols = ['value', 'sshPrnamt', 'sshPrnamtType', 'valUSD', 'balance', 'pctVal']
        for col in df.columns:
            if any(nc in col for nc in numeric_cols):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value for 13-F filings
        if 'value' in df.columns and 'shrsOrPrnAmt_sshPrnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['shrsOrPrnAmt_sshPrnamt'])
        # For NPORT-P, use valUSD and balance
        elif 'valUSD' in df.columns and 'balance' in df.columns:
            df = df.assign(unitValue=lambda x: x['valUSD'] / x['balance'])
        
        return df


class SECFilingManager:
    """High-level manager for SEC filing operations."""
    
    def __init__(self, form_type: str = "13F-HR", output_dir: str = "sec_filings", parse_method: str = 'ET'):
        """Initialize manager.
        
        Args:
            form_type: Form type to download - "13F-HR", "NPORT-P", etc.
            output_dir: Directory for downloaded files
            parse_method: Parsing method - 'ET' or 'pandas'
        """
        self.form_type = form_type
        self.downloader = SECFilingDownloader(output_dir)
        self.parser = SECFilingParser(parse_method)
        self.filings = {}
    
    def download_and_parse(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Download and parse filings in one step.
        
        Args:
            cik: Entity's CIK identifier
            num_reports: Number of reports to download
            
        Returns:
            Dictionary of parsed filings keyed by (cik, period_date)
        """
        paths = self.downloader.download(cik, form_type=self.form_type, num_reports=num_reports)
        self.filings = self.parser.parse_multiple(paths)
        return self.filings
    
    def get_filing_by_date(self, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date string.
        
        Args:
            date_str: Date string to search for (e.g., '2024-12-31')
            
        Returns:
            Tuple of (key, value) where key is (cik, period_date) and value is (metadata, holdings)
        """
        target_date = pd.to_datetime(date_str)
        
        print(f"Searching for date: {target_date}")
        print(f"Available filings:")
        for key, _ in self.filings.items():
            cik, period_date = key
            print(f"  CIK: {cik}, Date: {period_date}")
        
        for key, value in self.filings.items():
            cik, period_date = key
            if pd.notna(period_date) and period_date.date() == target_date.date():
                return (key, value)
        
        raise KeyError(f"No filing found for date {date_str}")


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
    # Example 1: Download 13-F filings for a hedge fund
    print("=" * 60)
    print("13-F: Renaissance Technologies")
    print("=" * 60)
    manager_13f = SECFilingManager(form_type="13F-HR")
    filings_13f = manager_13f.download_and_parse("0001037389", num_reports=2)
    
    # Example 2: Download NPORT-P filings for a mutual fund
    print("\n" + "=" * 60)
    print("NPORT-P: Vanguard Index Funds")
    print("=" * 60)
    manager_nport = SECFilingManager(form_type="NPORT-P")
    filings_nport = manager_nport.download_and_parse("0000036405", num_reports=2)
    
    # Display results
    if filings_nport:
        key, (metadata, holdings) = list(filings_nport.items())[0]
        print(f"\nVanguard filing details:")
        print(f"  CIK: {key[0]}")
        print(f"  Period: {key[1]}")
        print(f"  Fund: {metadata.get('name', 'N/A')}")
        print(f"  Holdings: {len(holdings)}")