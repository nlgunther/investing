"""
SEC Filing Parser Module

Parses XML streams into pandas DataFrames.
Works with both file paths and IO streams.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Tuple, Optional, List, Union, BinaryIO
import pandas as pd
from abc import ABC, abstractmethod
from enum import IntEnum
from dataclasses import dataclass


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
class ParseConfig:
    """Configuration for parsing operations."""
    verbosity: VerbosityLevel = VerbosityLevel.NORMAL


def log(message: str, level: VerbosityLevel, config: ParseConfig, prefix: str = ""):
    """Centralized logging function."""
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
    """Utility class for XML parsing operations."""
    
    @staticmethod
    def parse_stream(stream: BinaryIO) -> Optional[ET.Element]:
        """Parse XML from stream and remove namespaces.
        
        Args:
            stream: Binary stream containing XML data
            
        Returns:
            Root element with namespaces removed, or None if invalid
        """
        try:
            stream.seek(0)  # Reset stream position
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
        """Remove XML namespaces from element tree."""
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
        """
        data = {}
        def traverse(element: ET.Element, path: List[str] = []):
            for child in element:
                tag = child.tag.split('}')[-1]
                key = '_'.join(path + [tag]).lower()
                if len(child) == 0 and child.text and child.text.strip():
                    data[key] = child.text.strip()
                else:
                    traverse(child, path + [tag])
        traverse(root)
        return data
    
    @staticmethod
    def find_all(root: ET.Element, tag: str) -> List[ET.Element]:
        """Find all elements with tag (namespace-agnostic)."""
        return (root.findall(f'.//{{{root.tag.split("}")[0][1:]}}}{tag}') or
                root.findall(f'.//{tag}') or
                [e for e in root.findall('.//*') if e.tag.endswith(tag)])


# ============================================================================
# PARSING RESULT
# ============================================================================

@dataclass
class ParsingResult:
    """Result of parsing a filing."""
    metadata: Dict[str, str]
    holdings: pd.DataFrame
    cik: str
    period_date: pd.Timestamp


# ============================================================================
# PARSE STRATEGIES
# ============================================================================

class ParseStrategy(ABC):
    """Abstract parsing strategy."""
    
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
        """Get metadata field aliases."""
        pass
    
    def _to_numeric(self, df: pd.DataFrame, patterns: List[str]) -> pd.DataFrame:
        """Convert columns matching patterns to numeric types."""
        for col in df.columns:
            if any(p in col for p in patterns):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df


class Parse13F(ParseStrategy):
    """Parsing strategy for 13-F filings."""
    
    def parse_metadata(self, stream: BinaryIO) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml stream."""
        root = self.xml.parse_stream(stream)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        return {**metadata, **{alias: metadata.get(key, '') 
                              for alias, key in self.get_aliases().items()}}
    
    def parse_holdings(self, streams: Dict[str, BinaryIO]) -> pd.DataFrame:
        """Parse holdings from separate holdings XML file.
        
        Searches for files with 'holding' in name or named 'infotable.xml'.
        """
        # Find holdings file
        holdings_stream = None
        for filename, stream in streams.items():
            if 'holding' in filename.lower() or filename == 'infotable.xml':
                holdings_stream = stream
                break
        
        if not holdings_stream:
            log("No holdings file found", VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        root = self.xml.parse_stream(holdings_stream)
        if not root:
            return pd.DataFrame()
        
        holdings = [self.xml.to_dict(table) 
                   for table in self.xml.find_all(root, 'infoTable')]
        if not holdings:
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['value', 'sshprnamt'])
        
        if 'value' in df.columns and 'shrsorprnamt_sshprnamt' in df.columns:
            df = df.assign(unitValue=lambda x: x['value'] / x['shrsorprnamt_sshprnamt'])
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys."""
        return {
            'cik': 'headerdata_filerinfo_filer_credentials_cik',
            'periodofreport': 'headerdata_filerinfo_periodofreport',
            'name': 'formdata_coverpage_filingmanager_name'
        }


class ParseNPORT(ParseStrategy):
    """Parsing strategy for NPORT filings."""
    
    def parse_metadata(self, stream: BinaryIO) -> Dict[str, str]:
        """Extract metadata from primary_doc.xml stream."""
        root = self.xml.parse_stream(stream)
        if not root:
            return {}
        
        metadata = self.xml.to_dict(root)
        return {**metadata, **{alias: metadata.get(key, '') 
                              for alias, key in self.get_aliases().items()}}
    
    def parse_holdings(self, streams: Dict[str, BinaryIO]) -> pd.DataFrame:
        """Parse holdings from primary_doc.xml (single file structure)."""
        primary_stream = streams.get('primary_doc.xml')
        if not primary_stream:
            return pd.DataFrame()
        
        root = self.xml.parse_stream(primary_stream)
        if not root:
            return pd.DataFrame()
        
        holdings = [self.xml.to_dict(sec) 
                   for sec in self.xml.find_all(root, 'invstOrSec')]
        if not holdings:
            log("No holdings found in invstOrSecs section", 
                VerbosityLevel.VERBOSE, self.config, "    ")
            return pd.DataFrame()
        
        df = pd.DataFrame(holdings)
        df = self._to_numeric(df, ['valusd', 'balance', 'pctval'])
        
        if 'valusd' in df.columns and 'balance' in df.columns:
            df = df.assign(unitValue=lambda x: x['valusd'] / x['balance'])
        
        return df
    
    def get_aliases(self) -> Dict[str, str]:
        """Map short names to full hierarchical keys."""
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
    """Parses SEC filings from streams or files."""
    
    STRATEGIES = {
        '13F-HR': Parse13F,
        'NPORT-P': ParseNPORT,
        'NPORT-N': ParseNPORT,
    }
    
    def __init__(self, form_type: str, config: Optional[ParseConfig] = None):
        """Initialize parser for specified form type.
        
        Args:
            form_type: Form type (e.g., "13F-HR", "NPORT-P")
            config: Parse configuration
        """
        if form_type not in self.STRATEGIES:
            raise ValueError(f"Unsupported form type: {form_type}. "
                           f"Supported: {list(self.STRATEGIES.keys())}")
        
        self.form_type = form_type
        self.config = config or ParseConfig()
        
        strategy_cls = self.STRATEGIES[form_type]
        self.strategy = strategy_cls(self.config)
    
    def parse_streams(self, streams: Dict[str, BinaryIO]) -> ParsingResult:
        """Parse filing from streams.
        
        Args:
            streams: Dict of filename -> stream
            
        Returns:
            ParsingResult with metadata and holdings DataFrame
        """
        # Parse metadata from primary_doc.xml
        primary_stream = streams.get('primary_doc.xml')
        if not primary_stream:
            raise ParseError("primary_doc.xml not found in streams")
        
        metadata = self.strategy.parse_metadata(primary_stream)
        holdings = self.strategy.parse_holdings(streams)
        
        # Extract key fields
        cik = metadata.get('cik', 'unknown')
        period_str = metadata.get('periodofreport', 'unknown')
        
        try:
            period_date = pd.to_datetime(period_str)
        except:
            log(f"Warning: Could not parse date '{period_str}'", 
                VerbosityLevel.VERBOSE, self.config)
            period_date = pd.NaT
        
        log(f"✓ Parsed: CIK={cik}, Period={period_date.date() if pd.notna(period_date) else 'unknown'}, "
            f"Holdings={len(holdings)}", VerbosityLevel.NORMAL, self.config)
        
        return ParsingResult(
            metadata=metadata,
            holdings=holdings,
            cik=cik,
            period_date=period_date
        )
    
    def parse_directory(self, directory: Union[str, Path]) -> ParsingResult:
        """Parse filing from local directory.
        
        Args:
            directory: Path to directory containing XML files
            
        Returns:
            ParsingResult with metadata and holdings DataFrame
        """
        dir_path = Path(directory)
        if not dir_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Load all XML files into streams
        streams = {}
        for xml_file in dir_path.glob("*.xml"):
            with open(xml_file, 'rb') as f:
                from io import BytesIO
                streams[xml_file.name] = BytesIO(f.read())
        
        if not streams:
            raise ParseError(f"No XML files found in {directory}")
        
        log(f"Parsing {directory}", VerbosityLevel.VERBOSE, self.config)
        return self.parse_streams(streams)
    
    def parse_multiple_directories(self, directories: List[Union[str, Path]]) -> Dict[Tuple[str, pd.Timestamp], ParsingResult]:
        """Parse multiple filing directories.
        
        Args:
            directories: List of directory paths
            
        Returns:
            Dictionary keyed by (cik, period_date)
        """
        results = {}
        
        for directory in directories:
            try:
                result = self.parse_directory(directory)
                results[(result.cik, result.period_date)] = result
            except Exception as e:
                log(f"✗ {directory}: {e}", VerbosityLevel.ERROR, self.config)
        
        log(f"\n✓ Parsed {len(results)} filing(s)", VerbosityLevel.NORMAL, self.config)
        return results


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    from io import BytesIO
    
    # Example 1: Parse from streams (as returned by downloader)
    print("Example 1: Parse from streams")
    print("=" * 60)
    
    # Simulate streams from downloader
    example_streams = {
        'primary_doc.xml': BytesIO(b'<xml>...</xml>'),  # Would be actual XML
    }
    
    parser = SECParser("13F-HR", config=ParseConfig(verbosity=VerbosityLevel.NORMAL))
    # result = parser.parse_streams(example_streams)  # Would work with real XML
    
    # Example 2: Parse from local directory
    print("\n\nExample 2: Parse from local directory")
    print("=" * 60)
    
    parser = SECParser("13F-HR", config=ParseConfig(verbosity=VerbosityLevel.VERBOSE))
    # result = parser.parse_directory("sec_filings/0001037389_0001037389240001234")
    # print(f"\nMetadata keys: {list(result.metadata.keys())[:5]}...")
    # print(f"Holdings shape: {result.holdings.shape}")
    
    print("\nParser module ready. Use with real XML files or streams from downloader.")
