"""
Test Suite for SEC Parser Module (sec_parser.py)

Tests the SECParser class which parses SEC filings from XML streams or files
into pandas DataFrames using a strategy pattern for different filing types.

RUNNING TESTS
=============
pip install pytest pytest-cov pandas

# Run all tests
pytest test_sec_parser.py -v

# Run with coverage
pytest test_sec_parser.py --cov=sec_parser --cov-report=html

# Run specific test category
pytest test_sec_parser.py -k "XMLParser" -v

ARCHITECTURE
============
The tests follow the same structure as the parser module:
1. Shared utilities (XMLParser, logging, config)
2. Strategies (Parse13F, ParseNPORT)  
3. Parser interface (SECParser)
4. Integration scenarios

DESIGN PRINCIPLES
=================
- DRY: Reusable fixtures for common test data
- Isolation: Each test is independent with its own data
- Clarity: Descriptive test names explain what's being verified
- Coverage: Tests for success paths, error cases, and edge cases
- Maintainability: Organized by component, easy to extend
"""

import pytest
import pandas as pd
from pathlib import Path
from io import BytesIO
from unittest.mock import patch
import xml.etree.ElementTree as ET

from .sec_parser import (
    VerbosityLevel,
    ParseConfig,
    ParseError,
    XMLParser,
    Parse13F,
    ParseNPORT,
    SECParser,
    ParsingResult,
    _log
)


# =============================================================================
# TEST DATA FIXTURES
# =============================================================================

@pytest.fixture
def silent_config():
    """Configuration with no output (for most tests).
    
    Prevents test output clutter by suppressing all logging.
    """
    return ParseConfig(verbosity=VerbosityLevel.SILENT)


@pytest.fixture
def verbose_config():
    """Configuration with verbose output.
    
    Useful for debugging specific tests.
    """
    return ParseConfig(verbosity=VerbosityLevel.VERBOSE)


@pytest.fixture
def sample_13f_xml():
    """Complete 13-F filing XML data.
    
    Contains both primary document (metadata) and holdings (investment data).
    Structure matches real SEC 13-F filings:
    - primary: Filing manager info, CIK, period
    - holdings: Individual security positions with values and shares
    
    Returns:
        dict: Keys 'primary' and 'holdings' with XML strings
    """
    return {
        'primary': """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
  <headerData>
    <filerInfo>
      <filer><credentials><cik>0001037389</cik></credentials></filer>
      <periodOfReport>09-30-2024</periodOfReport>
    </filerInfo>
  </headerData>
  <formData>
    <coverPage>
      <filingManager><name>RENAISSANCE TECHNOLOGIES LLC</name></filingManager>
    </coverPage>
  </formData>
</edgarSubmission>""",
        'holdings': """<?xml version="1.0" encoding="UTF-8"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
  <infoTable>
    <nameOfIssuer>APPLE INC</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>037833100</cusip>
    <value>1000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>10000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
  <infoTable>
    <nameOfIssuer>MICROSOFT CORP</nameOfIssuer>
    <titleOfClass>COM</titleOfClass>
    <cusip>594918104</cusip>
    <value>2000000</value>
    <shrsOrPrnAmt>
      <sshPrnamt>5000</sshPrnamt>
      <sshPrnamtType>SH</sshPrnamtType>
    </shrsOrPrnAmt>
  </infoTable>
</informationTable>"""
    }


@pytest.fixture
def sample_nport_xml():
    """Complete NPORT filing XML data.
    
    NPORT filings use a single file (primary_doc.xml) containing both
    metadata and holdings. Structure includes:
    - Series information (fund identification)
    - Holdings in invstOrSecs section
    
    Returns:
        str: Complete NPORT XML document
    """
    return """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <headerData>
    <filerInfo>
      <filer><issuerCredentials><cik>0000036405</cik></issuerCredentials></filer>
      <seriesClassInfo><seriesId>S000002839</seriesId></seriesClassInfo>
    </filerInfo>
  </headerData>
  <formData>
    <genInfo>
      <seriesName>VANGUARD 500 INDEX FUND</seriesName>
      <seriesId>S000002839</seriesId>
      <repPdDate>2024-12-31</repPdDate>
    </genInfo>
    <invstOrSecs>
      <invstOrSec>
        <name>Apple Inc</name>
        <cusip>037833100</cusip>
        <balance>100000</balance>
        <valUSD>15000000</valUSD>
      </invstOrSec>
      <invstOrSec>
        <name>Microsoft Corp</name>
        <cusip>594918104</cusip>
        <balance>50000</balance>
        <valUSD>20000000</valUSD>
      </invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>"""


@pytest.fixture
def streams_13f(sample_13f_xml):
    """BytesIO streams for 13-F filing.
    
    Converts XML strings to BytesIO objects as they would be returned
    by the downloader module. This simulates the actual data flow.
    
    Args:
        sample_13f_xml: Fixture providing XML strings
        
    Returns:
        dict: Filename -> BytesIO stream mapping
    """
    return {
        'primary_doc.xml': BytesIO(sample_13f_xml['primary'].encode('utf-8')),
        'infotable.xml': BytesIO(sample_13f_xml['holdings'].encode('utf-8'))
    }


@pytest.fixture
def streams_nport(sample_nport_xml):
    """BytesIO streams for NPORT filing.
    
    NPORT uses single file, so only primary_doc.xml is needed.
    
    Args:
        sample_nport_xml: Fixture providing XML string
        
    Returns:
        dict: Filename -> BytesIO stream mapping
    """
    return {
        'primary_doc.xml': BytesIO(sample_nport_xml.encode('utf-8'))
    }


@pytest.fixture
def filing_dir_13f(tmp_path, sample_13f_xml):
    """Directory with 13-F XML files.
    
    Creates a temporary directory structure matching downloaded filings:
    filing_13f/
        primary_doc.xml
        infotable.xml
    
    Args:
        tmp_path: pytest's temporary directory fixture
        sample_13f_xml: XML data fixture
        
    Returns:
        Path: Directory containing XML files
    """
    dir_path = tmp_path / "filing_13f"
    dir_path.mkdir()
    (dir_path / "primary_doc.xml").write_text(sample_13f_xml['primary'])
    (dir_path / "infotable.xml").write_text(sample_13f_xml['holdings'])
    return dir_path


@pytest.fixture
def filing_dir_nport(tmp_path, sample_nport_xml):
    """Directory with NPORT XML file.
    
    Creates temporary directory with single primary_doc.xml file.
    
    Args:
        tmp_path: pytest's temporary directory fixture
        sample_nport_xml: XML data fixture
        
    Returns:
        Path: Directory containing XML file
    """
    dir_path = tmp_path / "filing_nport"
    dir_path.mkdir()
    (dir_path / "primary_doc.xml").write_text(sample_nport_xml)
    return dir_path


# =============================================================================
# INITIALIZATION & CONFIGURATION TESTS
# =============================================================================

class TestInitialization:
    """Test parser initialization and configuration.
    
    Verifies that:
    - All supported form types initialize correctly
    - Correct strategies are selected for each form type
    - Invalid form types are rejected with helpful errors
    - Custom configurations are properly applied
    """
    
    @pytest.mark.parametrize("form_type,strategy_class", [
        ("13F-HR", Parse13F),    # Hedge fund holdings
        ("NPORT-P", ParseNPORT),  # Mutual fund holdings (filed)
        ("NPORT-N", ParseNPORT),  # Mutual fund holdings (final)
    ])
    def test_supported_form_types(self, form_type, strategy_class):
        """Should initialize with correct strategy for each form type.
        
        Uses parametrized testing to verify all form type -> strategy mappings
        in a single test method. This ensures consistent behavior across all
        supported filing types.
        """
        parser = SECParser(form_type)
        
        assert parser.form_type == form_type
        assert isinstance(parser.strategy, strategy_class)
    
    def test_unsupported_form_type_raises_error(self):
        """Should raise ValueError with helpful message for invalid form type.
        
        Error message should:
        1. Clearly state the form type is unsupported
        2. List all valid form types for user guidance
        """
        with pytest.raises(ValueError) as exc_info:
            SECParser("INVALID-FORM")
        
        error_msg = str(exc_info.value)
        assert "Unsupported form type" in error_msg
        assert "13F-HR" in error_msg  # Should list supported types
    
    def test_custom_config(self):
        """Should accept and use custom configuration.
        
        Allows users to customize verbosity and other settings
        without modifying global defaults.
        """
        config = ParseConfig(verbosity=VerbosityLevel.DEBUG)
        parser = SECParser("13F-HR", config=config)
        
        assert parser.config is config
        assert parser.config.verbosity == VerbosityLevel.DEBUG
    
    def test_default_config(self):
        """Should create default config when none provided.
        
        Default configuration provides sensible settings for most use cases
        (NORMAL verbosity level).
        """
        parser = SECParser("13F-HR")
        
        assert parser.config is not None
        assert parser.config.verbosity == VerbosityLevel.NORMAL


class TestLogging:
    """Test logging functionality.
    
    The parser uses a verbosity-based logging system to control output.
    These tests verify the logging respects the configured verbosity level.
    """
    
    def test_log_respects_verbosity(self):
        """Should only log when message level meets config threshold.
        
        If config verbosity is NORMAL, VERBOSE messages should be suppressed.
        This prevents information overload during normal operation.
        """
        config = ParseConfig(verbosity=VerbosityLevel.NORMAL)
        
        with patch('builtins.print') as mock_print:
            _log("test", VerbosityLevel.VERBOSE, config)
            mock_print.assert_not_called()
    
    def test_log_prints_at_or_below_level(self):
        """Should log when message level is at or below config.
        
        If config verbosity is VERBOSE, both NORMAL and VERBOSE messages
        should be printed.
        """
        config = ParseConfig(verbosity=VerbosityLevel.VERBOSE)
        
        with patch('builtins.print') as mock_print:
            _log("test", VerbosityLevel.NORMAL, config)
            mock_print.assert_called_once()
    
    def test_log_includes_prefix(self):
        """Should include prefix in output.
        
        Prefixes allow indentation for hierarchical log messages,
        improving readability of parser output.
        """
        config = ParseConfig(verbosity=VerbosityLevel.NORMAL)
        
        with patch('builtins.print') as mock_print:
            _log("test", VerbosityLevel.NORMAL, config, prefix="  ")
            mock_print.assert_called_with("  test")


# =============================================================================
# XML PARSER TESTS
# =============================================================================

class TestXMLParser:
    """Test XML parsing utility methods.
    
    XMLParser is a utility class providing static methods for:
    - Parsing XML from streams and files
    - Removing namespaces
    - Converting XML to dictionaries
    - Finding elements regardless of namespace
    """
    
    class TestStreamParsing:
        """Test parsing from BytesIO streams.
        
        Stream parsing is the primary use case - the downloader provides
        streams and the parser must handle them correctly.
        """
        
        def test_parse_valid_stream(self, sample_13f_xml):
            """Should parse valid XML from stream.
            
            Basic happy path: valid XML should parse without errors
            and return the root element.
            """
            stream = BytesIO(sample_13f_xml['primary'].encode())
            root = XMLParser.parse_stream(stream)
            
            assert root is not None
            assert root.tag == 'edgarSubmission'
        
        def test_parse_invalid_stream_returns_none(self):
            """Should return None for malformed XML.
            
            Parser should handle malformed XML gracefully without crashing.
            Returning None allows calling code to handle the error appropriately.
            """
            stream = BytesIO(b"<unclosed>")
            root = XMLParser.parse_stream(stream)
            
            assert root is None
        
        def test_stream_position_reset(self, sample_13f_xml):
            """Should reset stream position before parsing.
            
            Critical for reusability: if stream has been partially read,
            parser should reset to beginning. Otherwise subsequent reads fail.
            """
            stream = BytesIO(sample_13f_xml['primary'].encode())
            stream.seek(100)  # Move away from start
            
            root = XMLParser.parse_stream(stream)
            
            assert root is not None  # Should still work
    
    class TestFileParsing:
        """Test parsing from file paths.
        
        File parsing is secondary use case for parsing previously downloaded
        filings from local storage.
        """
        
        def test_parse_valid_file(self, filing_dir_13f):
            """Should parse XML from file path.
            
            Accepts Path objects and correctly reads file contents.
            """
            xml_file = filing_dir_13f / "primary_doc.xml"
            root = XMLParser.parse_file(xml_file)
            
            assert root is not None
            assert root.tag == 'edgarSubmission'
        
        def test_missing_file_returns_none(self, tmp_path):
            """Should return None when file doesn't exist.
            
            Graceful handling of missing files prevents FileNotFoundError
            from bubbling up unexpectedly.
            """
            missing = tmp_path / "nonexistent.xml"
            root = XMLParser.parse_file(missing)
            
            assert root is None
        
        def test_invalid_file_returns_none(self, tmp_path):
            """Should return None for malformed XML file.
            
            Consistent error handling: whether stream or file, malformed
            XML returns None rather than raising exception.
            """
            bad_file = tmp_path / "bad.xml"
            bad_file.write_text("<unclosed>")
            
            root = XMLParser.parse_file(bad_file)
            
            assert root is None
    
    class TestNamespaceRemoval:
        """Test namespace handling.
        
        SEC XML files use namespaces extensively. The parser strips them
        to simplify querying and make tests more readable.
        """
        
        def test_remove_namespaces(self):
            """Should strip namespaces from all tags.
            
            Converts {http://example.com}root to just 'root'.
            This makes XPath queries simpler and more intuitive.
            """
            xml = '<root xmlns="http://example.com"><child>text</child></root>'
            root = ET.fromstring(xml)
            
            cleaned = XMLParser.remove_namespaces(root)
            
            assert cleaned.tag == 'root'
            assert cleaned[0].tag == 'child'
            assert '}' not in cleaned.tag  # Verify namespace removed
    
    class TestDictionaryConversion:
        """Test XML to dictionary conversion.
        
        The to_dict() method flattens XML hierarchies into dictionaries
        with underscore-separated keys. This makes accessing values simpler
        than navigating XML trees.
        """
        
        def test_simple_structure(self):
            """Should convert simple XML to flat dict.
            
            <root><name>John</name><age>30</age></root>
            becomes {'name': 'John', 'age': '30'}
            """
            xml = '<root><name>John</name><age>30</age></root>'
            root = ET.fromstring(xml)
            
            data = XMLParser.to_dict(root)
            
            assert data['name'] == 'John'
            assert data['age'] == '30'
        
        def test_nested_structure_creates_hierarchical_keys(self):
            """Should use underscores for nested elements.
            
            <root><person><name>John</name></person></root>
            becomes {'person_name': 'John'}
            
            This preserves the hierarchy while flattening the structure.
            """
            xml = '<root><person><name>John</name></person></root>'
            root = ET.fromstring(xml)
            
            data = XMLParser.to_dict(root)
            
            assert 'person_name' in data
            assert data['person_name'] == 'John'
        
        def test_ignores_empty_elements(self):
            """Should skip elements without text.
            
            Empty elements and branch nodes are not included in output.
            Only leaf nodes with actual text content are stored.
            """
            xml = '<root><empty></empty><filled>text</filled></root>'
            root = ET.fromstring(xml)
            
            data = XMLParser.to_dict(root)
            
            assert 'empty' not in data
            assert data['filled'] == 'text'
        
        def test_strips_whitespace(self):
            """Should trim whitespace from values.
            
            Leading and trailing whitespace is removed from all values.
            This normalizes data and prevents comparison issues.
            """
            xml = '<root><name>  John  </name></root>'
            root = ET.fromstring(xml)
            
            data = XMLParser.to_dict(root)
            
            assert data['name'] == 'John'  # No whitespace
        
        def test_converts_tags_to_lowercase(self):
            """Should convert tag names to lowercase.
            
            Normalizes all keys to lowercase for consistent access.
            <AGE> becomes 'age', matching Python naming conventions.
            """
            xml = '<root><NAME>John</NAME><AGE>30</AGE></root>'
            root = ET.fromstring(xml)
            
            data = XMLParser.to_dict(root)
            
            assert data['name'] == 'John'
            assert data['age'] == '30'
    
    class TestElementFinding:
        """Test namespace-agnostic element finding.
        
        find_all() must locate elements regardless of namespace usage.
        It tries multiple strategies to ensure robust element discovery.
        """
        
        def test_find_with_namespace(self):
            """Should find elements with namespaces.
            
            Tests that elements with namespace prefixes can be found
            using just the local tag name.
            """
            xml = '<root xmlns="http://ex.com"><item>1</item><item>2</item></root>'
            root = ET.fromstring(xml)
            
            elements = XMLParser.find_all(root, 'item')
            
            assert len(elements) == 2
            assert elements[0].text == '1'
        
        def test_find_without_namespace(self):
            """Should find elements without namespaces.
            
            Baseline test: elements without namespaces should definitely
            be findable.
            """
            xml = '<root><item>1</item><item>2</item></root>'
            root = ET.fromstring(xml)
            
            elements = XMLParser.find_all(root, 'item')
            
            assert len(elements) == 2
        
        def test_returns_empty_list_when_not_found(self):
            """Should return empty list for missing tags.
            
            Consistent return type: always returns a list, even if empty.
            Prevents None-checking in calling code.
            """
            xml = '<root><item>1</item></root>'
            root = ET.fromstring(xml)
            
            elements = XMLParser.find_all(root, 'missing')
            
            assert elements == []


# =============================================================================
# STRATEGY TESTS
# =============================================================================

class TestParse13FStrategy:
    """Test Parse13F strategy for 13-F filings.
    
    13-F filings report hedge fund holdings quarterly. The strategy must:
    - Extract metadata from primary document
    - Parse holdings from separate holdings file
    - Calculate derived fields (unit value)
    """
    
    def test_metadata_extraction(self, silent_config, streams_13f):
        """Should extract all metadata fields correctly.
        
        Verifies the hierarchical XML is correctly flattened and
        key fields are accessible via aliases.
        """
        strategy = Parse13F(silent_config)
        metadata = strategy.parse_metadata(streams_13f['primary_doc.xml'])
        
        # Check aliased fields
        assert metadata['cik'] == '0001037389'
        assert metadata['periodofreport'] == '09-30-2024'
        assert metadata['name'] == 'RENAISSANCE TECHNOLOGIES LLC'
    
    def test_holdings_parsing(self, silent_config, streams_13f):
        """Should parse all holdings into DataFrame.
        
        Each <infoTable> element becomes a row in the DataFrame.
        All child elements become columns.
        """
        strategy = Parse13F(silent_config)
        df = strategy.parse_holdings(streams_13f)
        
        assert len(df) == 2
        assert list(df['nameofissuer']) == ['APPLE INC', 'MICROSOFT CORP']
    
    def test_numeric_conversion(self, silent_config, streams_13f):
        """Should convert value fields to numeric types.
        
        String values like "1000000" must be converted to integers/floats
        for mathematical operations. The strategy automatically converts
        fields matching certain patterns.
        """
        strategy = Parse13F(silent_config)
        df = strategy.parse_holdings(streams_13f)
        
        assert pd.api.types.is_numeric_dtype(df['value'])
        assert df['value'].sum() == 3000000  # 1M + 2M
    
    def test_unit_value_calculation(self, silent_config, streams_13f):
        """Should calculate price per share correctly.
        
        unitValue = value / shares
        This derived field is useful for price analysis.
        """
        strategy = Parse13F(silent_config)
        df = strategy.parse_holdings(streams_13f)
        
        assert 'unitValue' in df.columns
        assert df['unitValue'].iloc[0] == 100.0  # 1M / 10K shares
        assert df['unitValue'].iloc[1] == 400.0  # 2M / 5K shares
    
    def test_missing_holdings_file(self, silent_config):
        """Should return empty DataFrame when holdings file missing.
        
        Graceful degradation: if holdings file is absent, returns empty
        DataFrame rather than crashing. Allows partial parsing.
        """
        strategy = Parse13F(silent_config)
        streams = {'primary_doc.xml': BytesIO(b'<root/>')}
        
        df = strategy.parse_holdings(streams)
        
        assert df.empty


class TestParseNPORTStrategy:
    """Test ParseNPORT strategy for NPORT filings.
    
    NPORT filings report mutual fund holdings monthly. Key differences
    from 13-F:
    - Single file structure (not separate holdings file)
    - Series ID for fund identification
    - Different field names and structure
    """
    
    def test_metadata_extraction(self, silent_config, streams_nport):
        """Should extract NPORT-specific metadata.
        
        NPORT has unique fields like seriesId and seriesName for
        identifying specific funds within a fund family.
        """
        strategy = ParseNPORT(silent_config)
        metadata = strategy.parse_metadata(streams_nport['primary_doc.xml'])
        
        assert metadata['cik'] == '0000036405'
        assert metadata['seriesid'] == 'S000002839'
        assert metadata['name'] == 'VANGUARD 500 INDEX FUND'
        assert metadata['periodofreport'] == '2024-12-31'
    
    def test_holdings_from_primary_doc(self, silent_config, streams_nport):
        """Should parse holdings from single primary_doc.xml.
        
        Unlike 13-F, NPORT stores holdings in the same file as metadata.
        Holdings are in <invstOrSecs> section.
        """
        strategy = ParseNPORT(silent_config)
        df = strategy.parse_holdings(streams_nport)
        
        assert len(df) == 2
        assert list(df['name']) == ['Apple Inc', 'Microsoft Corp']
    
    def test_unit_value_calculation(self, silent_config, streams_nport):
        """Should calculate price per unit correctly.
        
        unitValue = valUSD / balance
        Similar to 13-F but uses NPORT field names.
        """
        strategy = ParseNPORT(silent_config)
        df = strategy.parse_holdings(streams_nport)
        
        assert df['unitValue'].iloc[0] == 150.0  # 15M / 100K
        assert df['unitValue'].iloc[1] == 400.0  # 20M / 50K


# =============================================================================
# PARSER INTERFACE TESTS
# =============================================================================

class TestParseStreams:
    """Test parsing from streams (primary interface).
    
    parse_streams() is the main entry point for parsing. It coordinates
    metadata and holdings parsing, then packages results into ParsingResult.
    """
    
    def test_13f_success(self, streams_13f):
        """Should parse 13-F from streams successfully.
        
        End-to-end test of complete 13-F parsing workflow from streams.
        Verifies both metadata and holdings are extracted correctly.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert isinstance(result, ParsingResult)
        assert result.cik == '0001037389'
        assert result.period_date == pd.Timestamp('2024-09-30')
        assert len(result.holdings) == 2
    
    def test_nport_success(self, streams_nport):
        """Should parse NPORT from streams successfully.
        
        Verifies NPORT parsing works with single-file structure.
        """
        parser = SECParser("NPORT-P")
        result = parser.parse_streams(streams_nport)
        
        assert result.cik == '0000036405'
        assert result.period_date == pd.Timestamp('2024-12-31')
    
    def test_missing_primary_doc_raises_error(self):
        """Should raise ParseError when primary_doc.xml missing.
        
        primary_doc.xml is required for all filing types. Its absence
        is an error condition that should be reported clearly.
        """
        parser = SECParser("13F-HR")
        
        with pytest.raises(ParseError) as exc_info:
            parser.parse_streams({'other.xml': BytesIO(b'<root/>')})
        
        assert "primary_doc.xml not found" in str(exc_info.value)
    
    def test_invalid_date_handling(self, streams_13f):
        """Should handle unparseable dates gracefully.
        
        If date string can't be parsed, sets period_date to NaT (Not a Time)
        rather than crashing. Allows rest of parsing to continue.
        """
        # Corrupt the date in the XML
        bad_xml = streams_13f['primary_doc.xml'].getvalue().decode()
        bad_xml = bad_xml.replace('09-30-2024', 'bad-date')
        streams_13f['primary_doc.xml'] = BytesIO(bad_xml.encode())
        
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert pd.isna(result.period_date)  # Should be NaT


class TestParseDirectory:
    """Test parsing from directories (secondary interface).
    
    parse_directory() is convenience method for parsing previously
    downloaded filings from local storage.
    """
    
    def test_13f_directory(self, filing_dir_13f):
        """Should parse 13-F from directory.
        
        Loads XML files from directory, converts to streams internally,
        then parses normally.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_directory(filing_dir_13f)
        
        assert result.cik == '0001037389'
        assert len(result.holdings) == 2
    
    def test_nport_directory(self, filing_dir_nport):
        """Should parse NPORT from directory.
        
        Verifies directory parsing works for single-file NPORT structure.
        """
        parser = SECParser("NPORT-P")
        result = parser.parse_directory(filing_dir_nport)
        
        assert result.cik == '0000036405'
    
    def test_missing_directory_raises_error(self):
        """Should raise FileNotFoundError for missing directory.
        
        Clear error when directory doesn't exist prevents silent failures.
        """
        parser = SECParser("13F-HR")
        
        with pytest.raises(FileNotFoundError):
            parser.parse_directory("/nonexistent")
    
    def test_empty_directory_raises_error(self, tmp_path):
        """Should raise ParseError for directory without XML files.
        
        Directory exists but contains no XML files - this is an error
        condition that should be reported.
        """
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        
        parser = SECParser("13F-HR")
        
        with pytest.raises(ParseError):
            parser.parse_directory(empty_dir)


class TestParseMultipleDirectories:
    """Test batch parsing of multiple directories.
    
    parse_multiple_directories() enables efficient processing of many
    filings at once, with error resilience for production use.
    """
    
    def test_parse_multiple_success(self, filing_dir_13f, filing_dir_nport):
        """Should parse all directories and return keyed dict.
        
        Results are returned as dictionary with (cik, period_date) keys.
        This allows easy lookup and prevents duplicates.
        """
        parser_13f = SECParser("13F-HR")
        results = parser_13f.parse_multiple_directories([filing_dir_13f])
        
        assert len(results) == 1
        key = list(results.keys())[0]
        assert isinstance(key, tuple)
        assert len(key) == 2  # (cik, date)
    
    def test_error_resilience(self, tmp_path, filing_dir_13f):
        """Should continue after individual directory errors.
        
        Critical for batch processing: one bad directory shouldn't stop
        the entire batch. Errors are logged but processing continues.
        """
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()  # Empty directory will cause error
        
        parser = SECParser("13F-HR")
        results = parser.parse_multiple_directories([filing_dir_13f, bad_dir])
        
        assert len(results) == 1  # Only the good one


# =============================================================================
# PARSING RESULT TESTS
# =============================================================================

class TestParsingResult:
    """Test ParsingResult dataclass.
    
    ParsingResult is the return type from parsing operations. It packages
    metadata and holdings together with convenient extracted fields.
    """
    
    def test_parsing_result_structure(self, streams_13f):
        """ParsingResult should have expected attributes.
        
        Verifies the data structure contract for parsing results.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert hasattr(result, 'metadata')
        assert hasattr(result, 'holdings')
        assert hasattr(result, 'cik')
        assert hasattr(result, 'period_date')
    
    def test_parsing_result_metadata_is_dict(self, streams_13f):
        """Metadata should be dictionary.
        
        Dictionary format allows flexible access to all parsed fields.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert isinstance(result.metadata, dict)
    
    def test_parsing_result_holdings_is_dataframe(self, streams_13f):
        """Holdings should be pandas DataFrame.
        
        DataFrame format enables analysis, filtering, and manipulation
        using pandas operations.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert isinstance(result.holdings, pd.DataFrame)
    
    def test_parsing_result_cik_extracted(self, streams_13f):
        """CIK should be extracted for convenience.
        
        CIK is stored in both metadata dict and as top-level attribute
        for easy access without navigating the dictionary.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert result.cik == '0001037389'
        assert result.cik == result.metadata['cik']
    
    def test_parsing_result_period_date_parsed(self, streams_13f):
        """Period date should be parsed as Timestamp.
        
        Date is converted from string to pandas Timestamp for date
        operations and comparisons.
        """
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams_13f)
        
        assert isinstance(result.period_date, pd.Timestamp)
        assert result.period_date == pd.Timestamp('2024-09-30')


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """End-to-end integration tests.
    
    These tests verify complete workflows from start to finish,
    exercising multiple components together.
    """
    
    def test_full_13f_workflow(self, filing_dir_13f):
        """Test complete 13-F parsing workflow.
        
        Simulates real-world usage: parse a filing directory and verify
        all data is correctly extracted and structured.
        """
        # Initialize parser with silent config (no output during test)
        parser = SECParser("13F-HR", config=ParseConfig(verbosity=VerbosityLevel.SILENT))
        
        # Parse directory
        result = parser.parse_directory(filing_dir_13f)
        
        # Verify result structure
        assert isinstance(result, ParsingResult)
        
        # Verify metadata extracted correctly
        assert result.cik == '0001037389'
        assert result.period_date == pd.Timestamp('2024-09-30')
        assert result.metadata['name'] == 'RENAISSANCE TECHNOLOGIES LLC'
        
        # Verify holdings parsed correctly
        assert len(result.holdings) == 2
        assert result.holdings['value'].sum() == 3000000
    
    def test_full_nport_workflow(self, filing_dir_nport):
        """Test complete NPORT parsing workflow.
        
        Verifies NPORT single-file structure parses correctly end-to-end.
        """
        parser = SECParser("NPORT-P", config=ParseConfig(verbosity=VerbosityLevel.SILENT))
        
        result = parser.parse_directory(filing_dir_nport)
        
        # Verify NPORT-specific fields
        assert result.cik == '0000036405'
        assert result.period_date == pd.Timestamp('2024-12-31')
        assert result.metadata['name'] == 'VANGUARD 500 INDEX FUND'
        assert result.metadata['seriesid'] == 'S000002839'
        
        # Verify holdings
        assert len(result.holdings) == 2
        assert result.holdings['valusd'].sum() == 35000000
    
    def test_batch_processing_workflow(self, tmp_path, sample_13f_xml):
        """Test batch processing of multiple filings.
        
        Real-world scenario: process multiple filings at once, such as
        processing all quarterly filings for a year.
        """
        # Create three filings with different dates
        dirs = []
        for i, date in enumerate(['2024-09-30', '2024-06-30', '2024-03-31']):
            # Modify XML to have different date
            xml = sample_13f_xml['primary'].replace('09-30-2024', date)
            d = tmp_path / f"filing{i}"
            d.mkdir()
            (d / "primary_doc.xml").write_text(xml)
            (d / "infotable.xml").write_text(sample_13f_xml['holdings'])
            dirs.append(d)
        
        # Parse all at once
        parser = SECParser("13F-HR", config=ParseConfig(verbosity=VerbosityLevel.SILENT))
        results = parser.parse_multiple_directories(dirs)
        
        # Verify all parsed
        assert len(results) == 3
        
        # Verify they're keyed correctly by date
        dates = [key[1] for key in results.keys()]
        assert pd.Timestamp('2024-09-30') in dates
        assert pd.Timestamp('2024-06-30') in dates
        assert pd.Timestamp('2024-03-31') in dates
    
    def test_mixed_form_types_separate_parsers(self, filing_dir_13f, filing_dir_nport):
        """Different parsers for different form types should work independently.
        
        Verifies that parsing different filing types doesn't cause
        interference or state issues.
        """
        parser_13f = SECParser("13F-HR")
        parser_nport = SECParser("NPORT-P")
        
        result_13f = parser_13f.parse_directory(filing_dir_13f)
        result_nport = parser_nport.parse_directory(filing_dir_nport)
        
        # Both should succeed
        assert result_13f.cik == '0001037389'
        assert result_nport.cik == '0000036405'
        
        # Holdings structures should differ
        assert 'nameofissuer' in result_13f.holdings.columns
        assert 'name' in result_nport.holdings.columns  # Different column name


# =============================================================================
# EDGE CASES & ERROR HANDLING
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling.
    
    These tests verify the parser handles unusual or error conditions
    gracefully without crashing.
    """
    
    def test_empty_holdings_dataframe(self):
        """Should handle filings with no holdings gracefully.
        
        Some filings might have metadata but no holdings data.
        Parser should return empty DataFrame rather than crashing.
        """
        xml = """<?xml version="1.0"?>
        <root xmlns="http://example.com">
            <headerData><filerInfo><filer><credentials>
                <cik>123</cik>
            </credentials></filer><periodOfReport>2024-01-01</periodOfReport></filerInfo></headerData>
        </root>"""
        
        streams = {
            'primary_doc.xml': BytesIO(xml.encode()),
            'infotable.xml': BytesIO(b'<informationTable></informationTable>')
        }
        
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams)
        
        assert result.holdings.empty
        assert isinstance(result.holdings, pd.DataFrame)
    
    def xtest_malformed_xml_graceful_failure(self, tmp_path):
        """Should handle malformed XML without crashing.
        
        Malformed XML should raise ParseError rather than letting
        XML parsing exceptions bubble up uncaught.
        """
        bad_dir = tmp_path / "bad"
        bad_dir.mkdir()
        (bad_dir / "primary_doc.xml").write_text("<not>closed")
        
        parser = SECParser("13F-HR")
        
        # Should raise our custom exception, not raw XML error
        with pytest.raises(ParseError):
            parser.parse_directory(bad_dir)
    
    def test_missing_required_fields(self):
        """Should handle missing required metadata fields.
        
        If certain expected fields are missing from XML, parser should
        continue rather than crashing. Missing fields appear as empty strings.
        """
        xml = """<?xml version="1.0"?>
        <root xmlns="http://example.com">
            <headerData><filerInfo><filer><credentials>
                <cik>123</cik>
            </credentials></filer></filerInfo></headerData>
        </root>"""
        
        streams = {'primary_doc.xml': BytesIO(xml.encode())}
        
        parser = SECParser("13F-HR")
        result = parser.parse_streams(streams)
        
        # Should have CIK but not period
        assert result.cik == '123'
        assert result.metadata.get('periodofreport', '') == ''
    
    def test_very_large_holdings_file(self, tmp_path, sample_13f_xml):
        """Should handle large holdings files efficiently.
        
        Real hedge funds can have thousands of holdings. Parser should
        handle large files without performance issues or memory problems.
        """
        # Create holdings XML with 1000 entries
        holdings_xml = '<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">'
        for i in range(1000):
            holdings_xml += f'''
            <infoTable>
                <nameOfIssuer>COMPANY {i}</nameOfIssuer>
                <value>{i * 1000}</value>
                <shrsOrPrnAmt><sshPrnamt>{i * 10}</sshPrnamt></shrsOrPrnAmt>
            </infoTable>'''
        holdings_xml += '</informationTable>'
        
        dir_path = tmp_path / "large"
        dir_path.mkdir()
        (dir_path / "primary_doc.xml").write_text(sample_13f_xml['primary'])
        (dir_path / "infotable.xml").write_text(holdings_xml)
        
        parser = SECParser("13F-HR")
        result = parser.parse_directory(dir_path)
        
        assert len(result.holdings) == 1000
    
    def test_unicode_handling(self, tmp_path, sample_13f_xml):
        """Should handle unicode characters in company names.
        
        International companies may have non-ASCII characters.
        Parser should handle UTF-8 encoding correctly.
        """
        xml = """<?xml version="1.0" encoding="UTF-8"?>
        <edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
          <headerData>
            <filerInfo>
              <filer><credentials><cik>123</cik></credentials></filer>
              <periodOfReport>2024-01-01</periodOfReport>
            </filerInfo>
          </headerData>
          <formData>
            <coverPage>
              <filingManager><name>Société Générale</name></filingManager>
            </coverPage>
          </formData>
        </edgarSubmission>"""
        
        dir_path = tmp_path / "unicode"
        dir_path.mkdir()
        (dir_path / "primary_doc.xml").write_text(xml, encoding='utf-8')
        (dir_path / "infotable.xml").write_text(sample_13f_xml['holdings'])
        
        parser = SECParser("13F-HR")
        result = parser.parse_directory(dir_path)
        
        assert result.metadata['name'] == 'Société Générale'


# =============================================================================
# PARAMETRIZED TESTS
# =============================================================================

@pytest.mark.parametrize("form_type,strategy_class", [
    ("13F-HR", Parse13F),
    ("NPORT-P", ParseNPORT),
    ("NPORT-N", ParseNPORT),
])
def test_form_type_strategy_mapping(form_type, strategy_class):
    """Each form type should map to correct strategy class.
    
    Parametrized test ensures all form types in STRATEGIES registry
    are correctly mapped. Makes it easy to add new types.
    """
    parser = SECParser(form_type)
    assert isinstance(parser.strategy, strategy_class)


@pytest.mark.parametrize("verbosity", [
    VerbosityLevel.SILENT,
    VerbosityLevel.ERROR,
    VerbosityLevel.NORMAL,
    VerbosityLevel.VERBOSE,
    VerbosityLevel.DEBUG,
])
def test_verbosity_levels(verbosity):
    """Parser should accept all verbosity levels.
    
    Ensures all defined verbosity levels work with the parser.
    """
    config = ParseConfig(verbosity=verbosity)
    parser = SECParser("13F-HR", config=config)
    assert parser.config.verbosity == verbosity


# =============================================================================
# RUN CONFIGURATION
# =============================================================================

if __name__ == "__main__":
    # Run tests with verbose output and short tracebacks
    pytest.main([__file__, "-v", "--tb=short"])