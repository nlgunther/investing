"""
SEC 13-F Filing Downloader and Parser Module

Provides tools to download and parse SEC Form 13-F filings.
"""

import requests
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from bs4 import BeautifulSoup
import pandas as pd
from typing import Dict, Tuple, Optional


class SEC13FDownloader:
    """Downloads SEC Form 13-F filings."""
    
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
        """Download the most recent 13-F reports for a specified fund.
        
        Args:
            cik: The fund's CIK identifier
            form_type: The form type to download (default: "13F-HR")
            num_reports: Number of most recent reports to download
        
        Returns:
            List of paths to downloaded filing directories
        """
        # Normalize CIK to 10 digits with leading zeros
        cik = cik.strip().replace('-', '').zfill(10)
        
        # Get company filings metadata
        filings_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
        print(f"Fetching filings for CIK {cik}...")
        
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
            
            print(f"\nDownloading filing {filing_date} (Accession: {accession})...")
            
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
        
        # Fallback to common filenames
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


class SEC13FParser:
    """Parses SEC Form 13-F filings."""
    
    def __init__(self, method: str = 'ET'):
        """Initialize parser.
        
        Args:
            method: Parsing method - 'ET' (default) or 'pandas' for read_xml
        """
        self.method = method
    
    def parse_filing(self, filing_dir: str) -> Tuple[Dict, pd.DataFrame]:
        """Parse a single 13-F filing directory.
        
        Args:
            filing_dir: Path to filing directory
            
        Returns:
            Tuple of (metadata_dict, holdings_dataframe)
        """
        filing_path = Path(filing_dir)
        
        metadata = self._parse_primary_doc(filing_path / "primary_doc.xml")
        
        holdings_files = list(filing_path.glob("*holding*.xml"))
        holdings_df = self._parse_holdings_xml(holdings_files[0]) if holdings_files else pd.DataFrame()
        
        return metadata, holdings_df
    
    def parse_multiple(self, filing_dirs: list[str]) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Parse multiple 13-F filings.
        
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
                period_str = metadata.get('periodofreport', 'unknown')
                
                try:
                    period_date = pd.to_datetime(period_str)
                except:
                    period_date = pd.NaT
                
                key = (cik, period_date)
                results[key] = (metadata, holdings)
                print(f"✓ {filing_dir}: {len(holdings)} holdings")
            except Exception as e:
                print(f"✗ {filing_dir}: {e}")
        
        return results
    
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
        
        # Use ET.parse
        tree = ET.parse(xml_path)
        metadata = {}
        for elem in tree.iter():
            if elem.text and elem.text.strip() and len(elem) == 0:
                tag = elem.tag.split('}')[-1].lower()
                metadata[tag] = elem.text.strip()
        
        return metadata
    
    def _parse_holdings_xml(self, xml_path: Path) -> pd.DataFrame:
        """Parse holdings XML into a DataFrame."""
        if not xml_path.exists():
            return pd.DataFrame()
        
        if self.method == 'pandas':
            try:
                df = pd.read_xml(xml_path, xpath=".//infoTable")
            except Exception:
                try:
                    df = pd.read_xml(xml_path, xpath=".//*[local-name()='infoTable']")
                except Exception as e:
                    print(f"pandas read_xml failed for holdings: {e}")
                    return pd.DataFrame()
        else:
            # Use ET.parse
            tree = ET.parse(xml_path)
            info_tables = tree.findall('.//{*}infoTable')
            
            if not info_tables:
                print(f"No infoTable elements found in {xml_path}")
                return pd.DataFrame()
            
            holdings = []
            for table in info_tables:
                holding = {}
                for elem in table.iter():
                    if elem.text and elem.text.strip() and len(elem) == 0:
                        tag = elem.tag.split('}')[-1]
                        holding[tag] = elem.text.strip()
                holdings.append(holding)
            
            df = pd.DataFrame(holdings)
        
        # Convert numeric columns
        numeric_cols = ['value', 'sshPrnamt', 'sshPrnamtType']
        for col in df.columns:
            if any(nc in col for nc in numeric_cols):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        
        # Calculate unit value
        if 'value' in df.columns and 'sshPrnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['sshPrnamt'])
        
        return df


class SEC13FManager:
    """High-level manager for SEC 13-F operations."""
    
    def __init__(self, output_dir: str = "sec_filings", parse_method: str = 'ET'):
        """Initialize manager.
        
        Args:
            output_dir: Directory for downloaded files
            parse_method: Parsing method - 'ET' or 'pandas'
        """
        self.downloader = SEC13FDownloader(output_dir)
        self.parser = SEC13FParser(parse_method)
        self.filings = {}
    
    def download_and_parse(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Download and parse filings in one step.
        
        Args:
            cik: Fund's CIK identifier
            num_reports: Number of reports to download
            
        Returns:
            Dictionary of parsed filings keyed by (cik, period_date)
        """
        paths = self.downloader.download(cik, num_reports=num_reports)
        self.filings = self.parser.parse_multiple(paths)
        return self.filings
    
    def get_filing_by_date(self, date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
        """Get filing by date string.
        
        Args:
            date_str: Date string to search for (e.g., '2024-12-31')
            
        Returns:
            Tuple of (key, value) where key is (cik, period_date) and value is (metadata, holdings)
            
        Raises:
            KeyError: If no filing found for the given date
        """
        target_date = pd.to_datetime(date_str)
        
        for key, value in self.filings.items():
            cik, period_date = key
            if period_date == target_date:
                return (key, value)
        
        raise KeyError(f"No filing found for date {date_str}")


# Convenience function for standalone use
def get_filing_by_date(filings_dict: Dict[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]], 
                       date_str: str) -> Tuple[Tuple[str, pd.Timestamp], Tuple[Dict, pd.DataFrame]]:
    """Get filing by date string from a filings dictionary.
    
    Args:
        filings_dict: Dictionary returned by parse_multiple_filings
        date_str: Date string to search for
        
    Returns:
        Tuple of (key, value) where key is (cik, period_date) and value is (metadata, holdings)
    """
    target_date = pd.to_datetime(date_str)
    
    for key, value in filings_dict.items():
        cik, period_date = key
        if period_date == target_date:
            return (key, value)
    
    raise KeyError(f"No filing found for date {date_str}")


# Example usage
if __name__ == "__main__":
    # Using the high-level manager
    manager = SEC13FManager()
    filings = manager.download_and_parse("0001037389", num_reports=3)
    
    # Get specific filing
    try:
        key, (metadata, holdings) = manager.get_filing_by_date("2024-09-30")
        print(f"\nFiling for {key[1]}:")
        print(f"  Name: {metadata.get('name')}")
        print(f"  Holdings: {len(holdings)}")
    except KeyError as e:
        print(e)
    
    # Or use individual components
    # downloader = SEC13FDownloader()
    # paths = downloader.download("0001037389", num_reports=5)
    # parser = SEC13FParser(method='ET')
    # filings = parser.parse_multiple(paths)
