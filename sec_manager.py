"""
SEC Filing Manager Module

Thin coordination layer between downloader and parser.
Provides flexible options for storage and retrieval.
"""

from typing import Dict, List, Optional, Tuple
import pandas as pd
from pathlib import Path
from enum import Enum

from sec_downloader import SECDownloader, DownloadConfig, VerbosityLevel as DownloadVerbosity
from sec_parser import SECParser, ParseConfig, ParsingResult, VerbosityLevel as ParseVerbosity


# ============================================================================
# STORAGE OPTIONS
# ============================================================================

class StorageMode(Enum):
    """Storage mode for downloaded/parsed data."""
    MEMORY_ONLY = "memory"          # No storage, return DataFrames only
    SAVE_XML = "xml"                # Save XML files locally
    SAVE_DATAFRAMES = "dataframes"  # Save DataFrames as CSV/Parquet
    SAVE_BOTH = "both"              # Save both XML and DataFrames


# ============================================================================
# MANAGER
# ============================================================================

class SECFilingManager:
    """Coordinates downloading and parsing of SEC filings.
    
    This is a thin layer that:
    1. Downloads filings using SECDownloader
    2. Parses them using SECParser
    3. Optionally stores XML and/or DataFrames
    4. Returns results to user
    
    Example:
        # Memory only - no storage
        manager = SECFilingManager("13F-HR", storage_mode=StorageMode.MEMORY_ONLY)
        results = manager.fetch("0001037389", num_reports=5)
        
        # Save XML files
        manager = SECFilingManager("13F-HR", storage_mode=StorageMode.SAVE_XML)
        results = manager.fetch("0001037389", num_reports=5)
        
        # Save DataFrames as CSV
        manager = SECFilingManager("13F-HR", storage_mode=StorageMode.SAVE_DATAFRAMES)
        results = manager.fetch("0001037389", num_reports=5)
    """
    
    def __init__(self, 
                 form_type: str,
                 storage_mode: StorageMode = StorageMode.MEMORY_ONLY,
                 series_id: Optional[str] = None,
                 xml_dir: str = "sec_filings",
                 dataframe_dir: str = "sec_dataframes",
                 dataframe_format: str = "csv",  # or "parquet"
                 verbosity: Optional[int] = None):
        """Initialize manager.
        
        Args:
            form_type: Form type (e.g., "13F-HR", "NPORT-P")
            storage_mode: How to store data (memory/xml/dataframes/both)
            series_id: Series ID for NPORT filings (optional)
            xml_dir: Directory for XML files (if saving)
            dataframe_dir: Directory for DataFrames (if saving)
            dataframe_format: Format for saved DataFrames ("csv" or "parquet")
            verbosity: Verbosity level (0-4, optional)
        """
        self.form_type = form_type
        self.storage_mode = storage_mode
        self.xml_dir = Path(xml_dir)
        self.dataframe_dir = Path(dataframe_dir)
        self.dataframe_format = dataframe_format
        
        # Convert single verbosity to separate configs
        if verbosity is not None:
            download_config = DownloadConfig(verbosity=DownloadVerbosity(verbosity))
            parse_config = ParseConfig(verbosity=ParseVerbosity(verbosity))
        else:
            download_config = DownloadConfig()
            parse_config = ParseConfig()
        
        # Initialize downloader and parser
        self.downloader = SECDownloader(form_type, download_config, series_id)
        self.parser = SECParser(form_type, parse_config)
        
        # Storage for results
        self.results: Dict[Tuple[str, pd.Timestamp], ParsingResult] = {}
    
    def fetch(self, cik: str, num_reports: int = 5) -> Dict[Tuple[str, pd.Timestamp], ParsingResult]:
        """Fetch and parse SEC filings.
        
        This is the main method that coordinates downloading and parsing.
        
        Args:
            cik: Central Index Key
            num_reports: Maximum number of filings to fetch
            
        Returns:
            Dictionary keyed by (cik, period_date) containing ParsingResult objects
            Each ParsingResult has .metadata (dict) and .holdings (DataFrame)
        """
        # Determine if we need to save XML files
        save_xml = self.storage_mode in [StorageMode.SAVE_XML, StorageMode.SAVE_BOTH]
        
        # Download filings
        filing_results = self.downloader.download(
            cik=cik,
            num_reports=num_reports,
            save_to_disk=save_xml,
            output_dir=str(self.xml_dir)
        )
        
        if not filing_results:
            return {}
        
        # Parse each filing
        parsed_results = {}
        
        for filing_result in filing_results:
            try:
                # Parse from streams (works whether saved to disk or not)
                parse_result = self.parser.parse_streams(filing_result.files)
                
                # Store result
                key = (parse_result.cik, parse_result.period_date)
                parsed_results[key] = parse_result
                
                # Optionally save DataFrame
                if self.storage_mode in [StorageMode.SAVE_DATAFRAMES, StorageMode.SAVE_BOTH]:
                    self._save_dataframe(parse_result)
                
            except Exception as e:
                print(f"✗ Error parsing filing: {e}")
        
        # Store in manager
        self.results.update(parsed_results)
        
        return parsed_results
    
    def _save_dataframe(self, result: ParsingResult):
        """Save DataFrame to disk."""
        if result.holdings.empty:
            return
        
        self.dataframe_dir.mkdir(exist_ok=True)
        
        # Create filename
        date_str = result.period_date.strftime("%Y%m%d") if pd.notna(result.period_date) else "unknown"
        filename = f"{result.cik}_{date_str}_holdings.{self.dataframe_format}"
        filepath = self.dataframe_dir / filename
        
        # Save in specified format
        if self.dataframe_format == "csv":
            result.holdings.to_csv(filepath, index=False)
        elif self.dataframe_format == "parquet":
            result.holdings.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported format: {self.dataframe_format}")
        
        print(f"  ✓ Saved DataFrame: {filepath}")
    
    def get_filing_by_date(self, date_str: str) -> Optional[ParsingResult]:
        """Get filing by date from stored results.
        
        Args:
            date_str: Date string (e.g., '2024-09-30')
            
        Returns:
            ParsingResult if found, None otherwise
        """
        target = pd.to_datetime(date_str).date()
        
        for (cik, period_date), result in self.results.items():
            if pd.notna(period_date) and period_date.date() == target:
                return result
        
        return None
    
    def get_all_holdings(self) -> pd.DataFrame:
        """Combine all holdings into single DataFrame with date/CIK columns.
        
        Returns:
            DataFrame with all holdings plus 'filing_date' and 'cik' columns
        """
        all_dfs = []
        
        for (cik, period_date), result in self.results.items():
            if not result.holdings.empty:
                df = result.holdings.copy()
                df['filing_date'] = period_date
                df['cik'] = cik
                all_dfs.append(df)
        
        if not all_dfs:
            return pd.DataFrame()
        
        return pd.concat(all_dfs, ignore_index=True)
    
    def load_saved_dataframes(self, pattern: str = "*_holdings.csv") -> Dict[Tuple[str, pd.Timestamp], pd.DataFrame]:
        """Load previously saved DataFrames.
        
        Args:
            pattern: Glob pattern for DataFrame files
            
        Returns:
            Dictionary keyed by (cik, period_date)
        """
        if not self.dataframe_dir.exists():
            return {}
        
        results = {}
        
        for filepath in self.dataframe_dir.glob(pattern):
            try:
                # Parse filename: {cik}_{date}_holdings.{ext}
                parts = filepath.stem.split('_')
                cik = parts[0]
                date_str = parts[1]
                period_date = pd.to_datetime(date_str)
                
                # Load DataFrame
                if filepath.suffix == '.csv':
                    df = pd.read_csv(filepath)
                elif filepath.suffix == '.parquet':
                    df = pd.read_parquet(filepath)
                else:
                    continue
                
                results[(cik, period_date)] = df
                print(f"✓ Loaded: {filepath.name}")
                
            except Exception as e:
                print(f"✗ Error loading {filepath}: {e}")
        
        return results


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def fetch_filings(cik: str, 
                 form_type: str = "13F-HR",
                 num_reports: int = 5,
                 series_id: Optional[str] = None,
                 storage_mode: StorageMode = StorageMode.MEMORY_ONLY,
                 verbosity: int = 2) -> Dict[Tuple[str, pd.Timestamp], ParsingResult]:
    """Convenience function to fetch and parse filings in one call.
    
    Args:
        cik: Central Index Key
        form_type: Form type (e.g., "13F-HR", "NPORT-P")
        num_reports: Number of reports to fetch
        series_id: Series ID for NPORT filings
        storage_mode: How to store data
        verbosity: Verbosity level (0-4)
        
    Returns:
        Dictionary of parsed results
        
    Example:
        results = fetch_filings("0001037389", "13F-HR", num_reports=5)
        for (cik, date), result in results.items():
            print(f"{date}: {len(result.holdings)} holdings")
    """
    manager = SECFilingManager(
        form_type=form_type,
        storage_mode=storage_mode,
        series_id=series_id,
        verbosity=verbosity
    )
    return manager.fetch(cik, num_reports)


def parse_local_filings(directory: str,
                       form_type: str = "13F-HR",
                       verbosity: int = 2) -> Dict[Tuple[str, pd.Timestamp], ParsingResult]:
    """Parse previously downloaded XML filings from local directory.
    
    Args:
        directory: Directory containing filing subdirectories
        form_type: Form type
        verbosity: Verbosity level (0-4)
        
    Returns:
        Dictionary of parsed results
        
    Example:
        results = parse_local_filings("sec_filings", "13F-HR")
    """
    parser = SECParser(form_type, ParseConfig(verbosity=ParseVerbosity(verbosity)))
    
    # Find all subdirectories
    dir_path = Path(directory)
    subdirs = [p for p in dir_path.glob("*") if p.is_dir()]
    
    return parser.parse_multiple_directories(subdirs)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("Example 1: Memory only (no storage)")
    print("=" * 60)
    
    results = fetch_filings(
        cik="0001037389",
        form_type="13F-HR",
        num_reports=2,
        storage_mode=StorageMode.MEMORY_ONLY,
        verbosity=2
    )
    
    print(f"\nFetched {len(results)} filing(s)")
    for (cik, date), result in results.items():
        print(f"  {date.date()}: {len(result.holdings)} holdings")
        print(f"    Filer: {result.metadata.get('name', 'Unknown')}")
    
    
    print("\n\nExample 2: Save XML files")
    print("=" * 60)
    
    manager = SECFilingManager(
        form_type="13F-HR",
        storage_mode=StorageMode.SAVE_XML,
        verbosity=2
    )
    results = manager.fetch("0001037389", num_reports=1)
    print(f"\nXML files saved to: {manager.xml_dir}")
    
    
    print("\n\nExample 3: Save DataFrames as CSV")
    print("=" * 60)
    
    manager = SECFilingManager(
        form_type="13F-HR",
        storage_mode=StorageMode.SAVE_DATAFRAMES,
        dataframe_format="csv",
        verbosity=2
    )
    results = manager.fetch("0001037389", num_reports=1)
    print(f"\nCSV files saved to: {manager.dataframe_dir}")
    
    
    print("\n\nExample 4: NPORT with series filtering")
    print("=" * 60)
    
    results = fetch_filings(
        cik="0000036405",
        form_type="NPORT-P",
        series_id="S000002839",
        num_reports=1,
        storage_mode=StorageMode.MEMORY_ONLY,
        verbosity=2
    )
    
    if results:
        result = list(results.values())[0]
        print(f"\nFund: {result.metadata.get('name')}")
        print(f"Series ID: {result.metadata.get('seriesid')}")
        print(f"Holdings: {len(result.holdings)}")
    
    
    print("\n\nExample 5: Parse previously downloaded files")
    print("=" * 60)
    
    # Assuming files were downloaded earlier
    # results = parse_local_filings("sec_filings", "13F-HR", verbosity=2)
    print("Use parse_local_filings() to parse existing XML files")
    
    
    print("\n\nExample 6: Combine all holdings into one DataFrame")
    print("=" * 60)
    
    manager = SECFilingManager("13F-HR", storage_mode=StorageMode.MEMORY_ONLY, verbosity=1)
    results = manager.fetch("0001037389", num_reports=2)
    
    combined_df = manager.get_all_holdings()
    print(f"\nCombined DataFrame shape: {combined_df.shape}")
    if not combined_df.empty:
        print(f"Columns: {list(combined_df.columns)[:10]}...")
        print(f"Date range: {combined_df['filing_date'].min()} to {combined_df['filing_date'].max()}")
