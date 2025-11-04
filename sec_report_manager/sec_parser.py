"""
SEC Filing Parser Module

Parses XML streams into pandas DataFrames.
Works with both file paths and IO streams.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Optional, List, Union, BinaryIO, Callable
import pandas as pd
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass
from io import BytesIO
from itertools import groupby


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def unique_sorted_by_key(objs, f=lambda x: x) -> tuple:
    """
    Returns a tuple:
        (sorted_objects, unique_sorted_attribute_values)
    """
    # 1. Sort the objects once (O(n log n))
    sorted_objs = sorted(objs, key=f)

    # 2. Pull the unique attribute values while preserving order
    uniq_vals = [key for key, _ in groupby(sorted_objs, key=f)]

    return sorted_objs, uniq_vals

# def unique_sorted(lst): OLD
#     if not lst:
#         return []
#     result = [lst[0]]
#     for item in lst[1:]:
#         if item != result[-1]:
#             result.append(item)
#     return result

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
class ParseConfig:
    """Configuration for parsing operations."""
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL


def _log(message: str, level: VerbosityLevel, config: ParseConfig, prefix: str = ""):
    """Internal logging function."""
    if config.verbosity >= level:
        print(f"{prefix}{message}")


# ============================================================================
# EXCEPTIONS
# ============================================================================

class ParseError(Exception):
    """Base exception for parsing operations."""
    pass


# ============================================================================
# XML UTILITIES
# ============================================================================

class XMLParser:
    """Utility class for XML parsing operations.
    
    All methods are static as this is a pure utility class with no state.
    Handles namespace removal and hierarchical data extraction.
    """
    
    @staticmethod
    def parse_stream(stream: BinaryIO) -> Optional[ET.Element]:
        """Parse XML from stream and remove namespaces.
        
        Args:
            stream: Binary stream containing XML data
            
        Returns:
            Root element with namespaces removed, or None if invalid
        """
        try:
            stream.seek(0)  # Reset to beginning
            tree = ET.parse(stream)
            return XMLParser.remove_namespaces(tree.getroot())
        except ET.ParseError:
            return None
    
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
        
        def traverse(element: ET.Element, path: List[str] = None):
            if path is None:
                path = []
            
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
        
        Tries multiple strategies to find elements regardless of namespace.
        """
        # Try with root's namespace
        if '}' in root.tag:
            namespace = root.tag.split('}')[0][1:]
            result = root.findall(f'.//{{{namespace}}}{tag}')
            if result:
                return result
        
        # Try without namespace
        result = root.findall(f'.//{tag}')
        if result:
            return result
        
        # Fallback: check tag endings
        return [e for e in root.findall('.//*') if e.tag.endswith(tag)]


# ============================================================================
# PARSING RESULT
# ============================================================================

@dataclass
class ParsingResult:
    """Result of parsing a filing.
    
    Attributes:
        metadata: Dict of metadata fields (CIK, period, name, etc.)
        holdings: DataFrame of holdings data
        cik: Central Index Key (extracted for convenience)
        period_date: Period date as pandas Timestamp (extracted for convenience)
    """
    metadata: Dict[str, str]
    holdings: pd.DataFrame
    cik: str
    period_date: pd.Timestamp

from typing import List, Optional
import pandas as pd
from dataclasses import dataclass

@dataclass
class ParsingResult:
    metadata: Dict[str, str]
    holdings: pd.DataFrame
    cik: str
    period_date: pd.Timestamp


class ParsingResultManager:
    """Manages sorting and filtering of ParsingResult objects by metadata dates."""
    
    def __init__(self, results: List[ParsingResult],
                 date_key: str = 'filing_date',
                 extracter: Callable = lambda obj, key: obj.metadata[key]):
        """
        Initialize manager with results sorted by specified metadata date key.
        A fast method using pre-sorting and grouping.
        Args:
            results: List of ParsingResult objects
            date_key: Metadata key containing the date string for sorting
        """
        self.date_key = date_key # must be first as used in get_date
        self.extracter = extracter
        self.results, self.dates = unique_sorted_by_key(results,self.get_date)
    
    def get_date(self, result: ParsingResult) -> pd.Timestamp:
        """Extract and parse date from a result's metadata, converting 
        the day to the first day of the month."""

        return pd.to_datetime(self.extracter(result,self.date_key)).replace(day=1)
    
    def by_rank(self, rank: int = -1) -> List[ParsingResult]:
        """
        Get all results matching the date at the specified rank.
        
        Args:
            rank: -1 for most recent, 0 for earliest, -2 for 2nd most recent, etc.
        
        Returns:
            List of results with the date at that rank, or empty list if out of bounds.
        """
        if not self.results:
            return []
        
        try:
            target_date = self.dates[rank]
            return [r for r in self.results if self.get_date(r) == target_date]
        except IndexError:
            return []
    
    def by_date_range(
        self, 
        start: Optional[pd.Timestamp] = None, 
        end: Optional[pd.Timestamp] = None
    ) -> List[ParsingResult]:
        """Filter results by date range (inclusive on both ends)."""
        filtered = self.results
        if start:
            filtered = [r for r in filtered if self.get_date(r) >= start]
        if end:
            filtered = [r for r in filtered if self.get_date(r) <= end]
        return filtered


# Usage:
# manager = ParsingResultManager(results_list)
# most_recent = manager.by_rank(-1)
# earliest = manager.by_rank(0)
# second_oldest = manager.by_rank(1)
# in_range = manager.by_date_range(start_date, end_date)
# 
# Or use Python's built-in filter:
# by_cik = [r for r in manager.results if r.cik == '0001234567']



# ============================================================================
# PARSE STRATEGIES
# ============================================================================

class ParseStrategy(ABC):
    """Abstract parsing strategy defining interface for different filing types."""
    
    def __init__(self, config: ParseConfig):
        self.config = config
        self.xml = XMLParser()
    
    @abstractmethod
    def parse_metadata(self, stream: BinaryIO) -> Dict[str, str]:
        """Parse metadata from primary document stream."""
        pass
    
    @abstractmethod
    def parse_holdings(self, streams: Dict[str, BinaryIO]) -> pd.DataFrame:
        """Parse holdings data from file streams.
        
        Args:
            streams: Dict of filename -> stream for all files in filing
        """
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


class Parse13F(ParseStrategy):
    """Parsing strategy for 13-F filings (hedge fund holdings).
    
    13-F structure:
    - primary_doc.xml: Metadata (filer info, period, etc.)
    - *_holdings.xml or infotable.xml: Holdings data (one infoTable per holding)
    """
    
    def parse_metadata(self, stream: BinaryIO) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml stream."""
        root = self.xml.parse_stream(stream)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        
        # Add convenient aliases
        aliases = {alias: metadata.get(key, '') 
                  for alias, key in self.get_aliases().items()}
        
        return {**metadata, **aliases}
    
    def parse_holdings(self, streams: Dict[str, BinaryIO]) -> pd.DataFrame:
        """Parse holdings from separate holdings XML file.
        
        Searches for files with 'holding' in name or named 'infotable.xml'.
        Each <infoTable> element represents one holding.
        """
        # Find holdings file (naming varies by filer)
        holdings_stream = None
        for filename, stream in streams.items():
            if 'holding' in filename.lower() or filename == 'infotable.xml':
                holdings_stream = stream
                break
        
        if not holdings_stream:
            _log("No holdings file found", VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        root = self.xml.parse_stream(holdings_stream)
        if not root:
            return pd.DataFrame()
        
        # Extract all infoTable elements (one per holding)
        holdings = [self.xml.to_dict(table) 
                   for table in self.xml.find_all(root, 'infoTable')]
        
        if not holdings:
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['value', 'sshprnamt'])
        
        # Calculate unit value (price per share) if possible
        if 'value' in df.columns and 'shrsorprnamt_sshprnamt' in df.columns:
            with pd.option_context('mode.chained_assignment', None):
                df['unitValue'] = df['value'] / df['shrsorprnamt_sshprnamt']
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys for 13-F metadata."""
        return {
            'cik': 'headerdata_filerinfo_filer_credentials_cik',
            'periodofreport': 'headerdata_filerinfo_periodofreport',
            'name': 'formdata_coverpage_filingmanager_name'
        }


class ParseNPORT(ParseStrategy):
    """Parsing strategy for NPORT filings (mutual fund holdings).
    
    NPORT structure (single file):
    - primary_doc.xml contains both metadata and holdings
    - Metadata in <headerData> and <genInfo> sections
    - Holdings in <invstOrSecs> section (one <invstOrSec> per holding)
    """
    
    def parse_metadata(self, stream: BinaryIO) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml stream."""
        root = self.xml.parse_stream(stream)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        
        # Add convenient aliases
        aliases = {alias: metadata.get(key, '') 
                  for alias, key in self.get_aliases().items()}
        
        return {**metadata, **aliases}
    
    def parse_holdings(self, streams: Dict[str, BinaryIO]) -> pd.DataFrame:
        """Parse holdings from primary_doc.xml (single file structure).
        
        Each <invstOrSec> element represents one security holding.
        """
        primary_stream = streams.get('primary_doc.xml')
        if not primary_stream:
            return pd.DataFrame()
        
        root = self.xml.parse_stream(primary_stream)
        if not root:
            return pd.DataFrame()
        
        # Extract all invstOrSec elements
        holdings = [self.xml.to_dict(sec) 
                   for sec in self.xml.find_all(root, 'invstOrSec')]
        
        if not holdings:
            _log("No holdings found in invstOrSecs section", 
                VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['valusd', 'balance', 'pctval'])
        
        # Calculate unit value (price per unit) if possible
        if 'valusd' in df.columns and 'balance' in df.columns:
            with pd.option_context('mode.chained_assignment', None):
                df['unitValue'] = df['valusd'] / df['balance']
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys for NPORT metadata."""
        return {
            'cik': 'headerdata_filerinfo_filer_issuercredentials_cik',
            'periodofreport': 'formdata_geninfo_reppddate',
            'name': 'formdata_geninfo_seriesname',
            'seriesid': 'headerdata_filerinfo_seriesclassinfo_seriesid'
        }


# ============================================================================
# PARSER
# ============================================================================

class SECParser:
    """Parses SEC filings from streams or files.
    
    Features:
    - Works with IO streams (from downloader) or local files
    - Pluggable strategies for different filing types
    - Returns structured ParsingResult objects
    """
    
    # Registry of supported form types
    STRATEGIES = {
        '13F-HR': Parse13F,
        'NPORT-P': ParseNPORT,
        'NPORT-N': ParseNPORT,
    }
    
    def __init__(self, form_type: str, config: Optional[ParseConfig] = None):
        """Initialize parser for specified form type.
        
        Args:
            form_type: Form type (e.g., "13F-HR", "NPORT-P")
            config: Parse configuration (uses defaults if None)
        """
        if form_type not in self.STRATEGIES:
            raise ValueError(
                f"Unsupported form type: {form_type}. "
                f"Supported: {', '.join(self.STRATEGIES.keys())}"
            )
        
        self.form_type = form_type
        self.config = config or ParseConfig()
        
        # Create appropriate strategy
        strategy_cls = self.STRATEGIES[form_type]
        self.strategy = strategy_cls(self.config)
    
    def parse_streams(self, streams: Dict[str, BinaryIO]) -> ParsingResult:
        """Parse filing from streams.
        
        Args:
            streams: Dict of filename -> stream (e.g., from downloader)
            
        Returns:
            ParsingResult with metadata and holdings DataFrame
            
        Example:
            parser = SECParser("13F-HR")
            result = parser.parse_streams(filing_result.files)
            print(f"CIK: {result.cik}")
            print(f"Holdings: {len(result.holdings)}")
        """
        # Parse metadata from primary_doc.xml
        primary_stream = streams.get('primary_doc.xml')
        if not primary_stream:
            raise ParseError("primary_doc.xml not found in streams")
        
        metadata = self.strategy.parse_metadata(primary_stream)
        holdings = self.strategy.parse_holdings(streams)
        
        # Extract key fields for convenience
        cik = metadata.get('cik', 'unknown')
        period_str = metadata.get('periodofreport', 'unknown')
        
        # Parse period date
        try:
            period_date = pd.to_datetime(period_str)
        except Exception:
            _log(f"Warning: Could not parse date '{period_str}'", 
                VerbosityLevel.VERBOSE, self.config)
            period_date = pd.NaT
        
        _log(f"✓ Parsed: CIK={cik}, Period={period_date.date() if pd.notna(period_date) else 'unknown'}, "
            f"Holdings={len(holdings)}", VerbosityLevel.NORMAL, self.config)
        
        return ParsingResult(
            metadata=metadata,
            holdings=holdings,
            cik=cik,
            period_date=period_date
        )
    
    def parse_directory(self, directory: Union[str, Path],
                        globex = '*.xml') -> ParsingResult:
        """Parse filing from local directory.
        
        Args:
            directory: Path to directory containing XML files
            
        Returns:
            ParsingResult with metadata and holdings DataFrame
            
        Example:
            parser = SECParser("13F-HR")
            result = parser.parse_directory("sec_filings/0001037389_xxx")
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Load all XML files into streams
        streams = {}
        for xml_file in dir_path.glob(globex):
            with open(xml_file, 'rb') as f:
                streams[xml_file.name] = BytesIO(f.read())
        
        if not streams:
            raise ParseError(f"No XML files found in {directory}")
        
        _log(f"Parsing {directory}", VerbosityLevel.VERBOSE, self.config)
        return self.parse_streams(streams)
    
    def parse_multiple_directories(self, 
                                   directories: List[Union[str, Path]],
                                   globex = '*.xml') -> Dict[tuple, ParsingResult]:
        """Parse multiple filing directories.
        
        Args:
            directories: List of directory paths
            
        Returns:
            Dictionary keyed by (cik, period_date) containing ParsingResult objects
            
        Example:
            parser = SECParser("13F-HR")
            results = parser.parse_multiple_directories([
                "sec_filings/filing1",
                "sec_filings/filing2"
            ])
            for (cik, date), result in results.items():
                print(f"{cik} - {date}: {len(result.holdings)} holdings")
        """
        results = {}
        
        for directory in directories:
            try:
                result = self.parse_directory(directory,globex)
                key = (result.cik, result.period_date)
                results[key] = result
                
            except Exception as e:
                _log(f"✗ {directory}: {e}", VerbosityLevel.ERROR, self.config)
        
        _log(f"\n✓ Parsed {len(results)} filing(s)", VerbosityLevel.NORMAL, self.config)
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from io import BytesIO
    
    print("SEC Parser Module")
    print("=" * 60)
    print("\nThis module parses SEC filings from streams or local files.")
    print("\nExample 1: Parse from streams (as returned by downloader)")
    print("-" * 60)
    print("""
    from sec_downloader import SECDownloader
    from sec_parser import SECParser
    
    # Download filing
    downloader = SECDownloader("13F-HR")
    results = downloader.download("0001037389", num_reports=1)
    
    # Parse from streams
    parser = SECParser("13F-HR")
    parsed = parser.parse_streams(results[0].files)
    
    print(f"CIK: {parsed.cik}")
    print(f"Period: {parsed.period_date}")
    print(f"Holdings: {len(parsed.holdings)}")
    """)
    
    print("\nExample 2: Parse from local directory")
    print("-" * 60)
    print("""
    parser = SECParser("13F-HR")
    result = parser.parse_directory("sec_filings/0001037389_xxx")
    
    print(f"Metadata keys: {list(result.metadata.keys())[:5]}")
    print(f"Holdings shape: {result.holdings.shape}")
    """)
    
    print("\nExample 3: Parse multiple directories")
    print("-" * 60)
    print("""
    parser = SECParser("13F-HR")
    results = parser.parse_multiple_directories([
        "sec_filings/filing1",
        "sec_filings/filing2"
    ])
    
    for (cik, date), result in results.items():
        print(f"{cik} - {date}: {len(result.holdings)} holdings")
    """)
