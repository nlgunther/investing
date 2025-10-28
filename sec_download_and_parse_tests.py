"""
Improved Test Suite for SEC Filing Module

Comprehensive tests with proper isolation, clear naming, and complete fixtures.

Install:
    pip install pytest pytest-mock pandas

Run:
    pytest test_sec_filing.py -v
    pytest test_sec_filing.py -v --cov
    pytest test_sec_filing.py -v -k "13f"  # Only 13-F tests
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import xml.etree.ElementTree as ET


# Try to import module
try:
    from sec_file_manager import (
        SECConfig,
        FilingNotFoundError,
        ParsingError,
        XMLParser,
        FilingFileDiscoverer,
        FilingMetadata,
        Filing13FDownloaderStrategy,
        FilingNPORTDownloaderStrategy,
        Filing13FParserStrategy,
        FilingNPORTParserStrategy,
        SECFilingDownloader,
        SECFilingParser,
        SECFilingManager,
        get_filing_by_date
    )
except ImportError:
    pytest.skip("Module not found", allow_module_level=True)


# ============================================================================
# CONFIGURATION FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Test configuration with faster delays."""
    return SECConfig(
        user_agent="Test Agent test@example.com",
        rate_limit_delay=0.0,  # No delays in tests
        request_delay=0.0
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP client that doesn't make real requests."""
    client = Mock()
    client.get = Mock()
    return client


# ============================================================================
# XML FIXTURES
# ============================================================================

@pytest.fixture
def sample_13f_primary_xml():
    """Complete 13-F primary_doc.xml sample."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/thirteenffiler">
  <headerData>
    <submissionType>13F-HR</submissionType>
    <filerInfo>
      <filer>
        <credentials>
          <cik>0001037389</cik>
        </credentials>
      </filer>
      <periodOfReport>09-30-2024</periodOfReport>
    </filerInfo>
  </headerData>
  <formData>
    <coverPage>
      <reportCalendarOrQuarter>09-30-2024</reportCalendarOrQuarter>
      <filingManager>
        <name>RENAISSANCE TECHNOLOGIES LLC</name>
      </filingManager>
    </coverPage>
  </formData>
</edgarSubmission>"""


@pytest.fixture
def sample_13f_holdings_xml():
    """Complete 13-F holdings XML sample."""
    return """<?xml version="1.0" encoding="UTF-8"?>
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


@pytest.fixture
def sample_nport_primary_xml():
    """Complete NPORT-P primary_doc.xml with holdings."""
    return """<?xml version="1.0" encoding="UTF-8"?>
<edgarSubmission xmlns="http://www.sec.gov/edgar/nport">
  <headerData>
    <filerInfo>
      <filer>
        <issuerCredentials>
          <cik>0000036405</cik>
        </issuerCredentials>
      </filer>
      <seriesClassInfo>
        <seriesId>S000002839</seriesId>
      </seriesClassInfo>
    </filerInfo>
  </headerData>
  <formData>
    <genInfo>
      <regName>VANGUARD INDEX FUNDS</regName>
      <seriesName>VANGUARD 500 INDEX FUND</seriesName>
      <seriesId>S000002839</seriesId>
      <repPdEnd>2024-12-31</repPdEnd>
    </genInfo>
    <invstOrSecs>
      <invstOrSec>
        <n>Apple Inc</n>
        <cusip>037833100</cusip>
        <balance>100000</balance>
        <valUSD>15000000</valUSD>
      </invstOrSec>
      <invstOrSec>
        <n>Microsoft Corp</n>
        <cusip>594918104</cusip>
        <balance>50000</balance>
        <valUSD>20000000</valUSD>
      </invstOrSec>
    </invstOrSecs>
  </formData>
</edgarSubmission>"""


# ============================================================================
# TEMP DIRECTORY FIXTURES
# ============================================================================

@pytest.fixture
def temp_filing_dir(tmp_path):
    """Temporary directory for filing files."""
    filing_dir = tmp_path / "test_filing"
    filing_dir.mkdir()
    return filing_dir


@pytest.fixture
def filing_with_13f_files(temp_filing_dir, sample_13f_primary_xml, sample_13f_holdings_xml):
    """Filing directory with 13-F files already written."""
    (temp_filing_dir / "primary_doc.xml").write_text(sample_13f_primary_xml)
    (temp_filing_dir / "infotable.xml").write_text(sample_13f_holdings_xml)
    return temp_filing_dir


@pytest.fixture
def filing_with_nport_files(temp_filing_dir, sample_nport_primary_xml):
    """Filing directory with NPORT file already written."""
    (temp_filing_dir / "primary_doc.xml").write_text(sample_nport_primary_xml)
    return temp_filing_dir


# ============================================================================
# XMLPARSER TESTS
# ============================================================================

class TestXMLParser:
    """Tests for XMLParser utility class."""
    
    def test_remove_namespaces_strips_namespace_prefixes(self):
        """XMLParser should remove namespace prefixes from tags."""
        xml = '<root xmlns="http://example.com"><child>text</child></root>'
        root = ET.fromstring(xml)
        parser = XMLParser()
        
        cleaned = parser.remove_namespaces(root)
        
        assert cleaned.tag == 'root'
        assert cleaned[0].tag == 'child'
    
    def test_extract_hierarchical_data_creates_nested_keys(self):
        """Should create underscore-separated keys for nested elements."""
        xml = '<root><level1><level2>value</level2></level1></root>'
        root = ET.fromstring(xml)
        parser = XMLParser()
        
        data = parser.extract_hierarchical_data(root)
        
        assert 'level1_level2' in data
        assert data['level1_level2'] == 'value'
    
    def test_extract_hierarchical_data_handles_multiple_children(self):
        """Should extract all leaf nodes from complex structure."""
        xml = '<root><a>1</a><b><c>2</c><d>3</d></b></root>'
        root = ET.fromstring(xml)
        parser = XMLParser()
        
        data = parser.extract_hierarchical_data(root)
        
        assert data['a'] == '1'
        assert data['b_c'] == '2'
        assert data['b_d'] == '3'
    
    def test_find_elements_locates_elements_with_namespace(self):
        """Should find elements even with XML namespaces."""
        xml = '<root xmlns="http://ex.com"><target>1</target><target>2</target></root>'
        root = ET.fromstring(xml)
        parser = XMLParser()
        
        elements = parser.find_elements(root, 'target')
        
        assert len(elements) == 2


# ============================================================================
# FILING METADATA TESTS
# ============================================================================

class TestFilingMetadata:
    """Tests for FilingMetadata dataclass."""
    
    def test_accession_no_dash_removes_dashes(self):
        """Should remove dashes from accession number."""
        metadata = FilingMetadata(
            cik="0001037389",
            accession_number="0001037389-24-000123",
            filing_date="2024-09-30",
            form_type="13F-HR"
        )
        
        assert metadata.accession_no_dash == "000103738924000123"
    
    def test_archive_url_constructs_correct_url(self):
        """Should construct proper SEC archive URL."""
        metadata = FilingMetadata(
            cik="0001037389",
            accession_number="0001037389-24-000123",
            filing_date="2024-09-30",
            form_type="13F-HR"
        )
        
        url = metadata.archive_url("https://www.sec.gov")
        
        assert url == "https://www.sec.gov/Archives/edgar/data/1037389/000103738924000123"


# ============================================================================
# 13-F DOWNLOADER STRATEGY TESTS
# ============================================================================

class Test13FDownloaderStrategy:
    """Tests for 13-F downloader strategy."""
    
    def test_should_download_always_returns_true(self, mock_http_client, config):
        """13-F strategy downloads all filings without filtering."""
        discoverer = Mock()
        strategy = Filing13FDownloaderStrategy(mock_http_client, config, discoverer)
        filing = FilingMetadata("123", "456", "2024-01-01", "13F-HR")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_get_required_files_uses_discoverer(self, mock_http_client, config):
        """Should delegate file discovery to FilingFileDiscoverer."""
        discoverer = Mock()
        discoverer.discover_xml_files.return_value = ["file1.xml", "file2.xml"]
        strategy = Filing13FDownloaderStrategy(mock_http_client, config, discoverer)
        
        files = strategy.get_required_files("http://example.com")
        
        assert files == ["file1.xml", "file2.xml"]
        discoverer.discover_xml_files.assert_called_once_with("http://example.com")


# ============================================================================
# NPORT DOWNLOADER STRATEGY TESTS
# ============================================================================

class TestNPORTDownloaderStrategy:
    """Tests for NPORT downloader strategy."""
    
    def test_should_download_with_no_filter_returns_true(self, mock_http_client, config):
        """When no series_id specified, should download all filings."""
        strategy = FilingNPORTDownloaderStrategy(mock_http_client, config, series_id=None)
        filing = FilingMetadata("123", "456", "2024-01-01", "NPORT-P")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_should_download_with_matching_series_returns_true(
        self, mock_http_client, config, sample_nport_primary_xml
    ):
        """Should return True when series ID matches."""
        mock_response = Mock()
        mock_response.content = sample_nport_primary_xml.encode()
        mock_http_client.get.return_value = mock_response
        
        strategy = FilingNPORTDownloaderStrategy(mock_http_client, config, series_id="S000002839")
        filing = FilingMetadata("36405", "456", "2024-01-01", "NPORT-P")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_should_download_with_non_matching_series_returns_false(
        self, mock_http_client, config, sample_nport_primary_xml
    ):
        """Should return False when series ID doesn't match."""
        mock_response = Mock()
        mock_response.content = sample_nport_primary_xml.encode()
        mock_http_client.get.return_value = mock_response
        
        strategy = FilingNPORTDownloaderStrategy(mock_http_client, config, series_id="S999999999")
        filing = FilingMetadata("36405", "456", "2024-01-01", "NPORT-P")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is False
    
    def test_get_required_files_returns_primary_doc_only(self, mock_http_client, config):
        """NPORT should only require primary_doc.xml."""
        strategy = FilingNPORTDownloaderStrategy(mock_http_client, config)
        
        files = strategy.get_required_files("http://example.com")
        
        assert files == ["primary_doc.xml"]


# ============================================================================
# 13-F PARSER STRATEGY TESTS
# ============================================================================

class Test13FParserStrategy:
    """Tests for 13-F parser strategy."""
    
    def test_parse_metadata_extracts_cik(self, filing_with_13f_files):
        """Should extract CIK from 13-F primary document."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        assert metadata['cik'] == '0001037389'
    
    def test_parse_metadata_extracts_period(self, filing_with_13f_files):
        """Should extract period of report."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        assert metadata['periodofreport'] == '09-30-2024'
    
    def test_parse_metadata_extracts_name(self, filing_with_13f_files):
        """Should extract filing manager name."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        assert metadata['name'] == 'RENAISSANCE TECHNOLOGIES LLC'
    
    def test_parse_metadata_creates_aliases(self, filing_with_13f_files):
        """Should create convenient aliases for common fields."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        # Should have both hierarchical key and alias
        assert 'cik' in metadata
        assert 'headerdata_filerinfo_filer_credentials_cik' in metadata
    
    def test_parse_holdings_creates_dataframe_with_correct_rows(self, filing_with_13f_files):
        """Should create DataFrame with one row per holding."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert len(df) == 2
    
    def test_parse_holdings_extracts_issuer_names(self, filing_with_13f_files):
        """Should extract security names."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert 'nameofissuer' in df.columns
        assert 'APPLE INC' in df['nameofissuer'].values
        assert 'MICROSOFT CORP' in df['nameofissuer'].values
    
    def test_parse_holdings_flattens_nested_structure(self, filing_with_13f_files):
        """Should flatten nested XML elements like shrsOrPrnAmt."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert 'shrsorprnamt_sshprnamt' in df.columns
        assert 'shrsorprnamt_sshprnamttype' in df.columns
    
    def test_parse_holdings_converts_numeric_columns(self, filing_with_13f_files):
        """Should convert value and share columns to numeric types."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert pd.api.types.is_numeric_dtype(df['value'])
        assert pd.api.types.is_numeric_dtype(df['shrsorprnamt_sshprnamt'])
    
    def test_parse_holdings_calculates_unit_value(self, filing_with_13f_files):
        """Should calculate unitValue as value divided by shares."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert 'unitValue' in df.columns
        # First holding: value=1000000, shares=10000 -> unitValue=100
        assert df['unitValue'].iloc[0] == 100.0
        # Second holding: value=2000000, shares=5000 -> unitValue=400
        assert df['unitValue'].iloc[1] == 400.0
    
    def test_parse_holdings_returns_empty_dataframe_when_no_file(self, temp_filing_dir):
        """Should return empty DataFrame when holdings file missing."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(temp_filing_dir)
        
        assert df.empty


# ============================================================================
# NPORT PARSER STRATEGY TESTS
# ============================================================================

class TestNPORTParserStrategy:
    """Tests for NPORT parser strategy."""
    
    def test_parse_metadata_extracts_cik(self, filing_with_nport_files):
        """Should extract CIK from NPORT document."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_nport_files / "primary_doc.xml")
        
        assert metadata['cik'] == '0000036405'
    
    def test_parse_metadata_extracts_series_name(self, filing_with_nport_files):
        """Should extract series name."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_nport_files / "primary_doc.xml")
        
        assert metadata['name'] == 'VANGUARD 500 INDEX FUND'
    
    def test_parse_metadata_extracts_series_id(self, filing_with_nport_files):
        """Should extract series ID."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_nport_files / "primary_doc.xml")
        
        assert metadata['seriesid'] == 'S000002839'
    
    def test_parse_metadata_extracts_period(self, filing_with_nport_files):
        """Should extract reporting period end date."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(filing_with_nport_files / "primary_doc.xml")
        
        assert metadata['periodofreport'] == '2024-12-31'
    
    def test_parse_holdings_from_single_file(self, filing_with_nport_files):
        """Should parse holdings from primary_doc.xml (single file structure)."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert len(df) == 2
    
    def test_parse_holdings_extracts_security_names(self, filing_with_nport_files):
        """Should extract security names from n tag."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert 'n' in df.columns
        assert 'Apple Inc' in df['n'].values
    
    def test_parse_holdings_extracts_values_and_balances(self, filing_with_nport_files):
        """Should extract valUSD and balance columns."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert 'valusd' in df.columns
        assert 'balance' in df.columns
    
    def test_parse_holdings_calculates_unit_value(self, filing_with_nport_files):
        """Should calculate unitValue as valUSD / balance."""
        xml_parser = XMLParser()
        strategy = FilingNPORTParserStrategy(xml_parser)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert 'unitValue' in df.columns
        # First: valUSD=15000000, balance=100000 -> unitValue=150
        assert df['unitValue'].iloc[0] == 150.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSECFilingManager:
    """Integration tests for manager facade."""
    
    def test_manager_initializes_13f_strategies(self):
        """Should create appropriate strategies for 13F-HR form."""
        manager = SECFilingManager(form_type="13F-HR")
        
        assert isinstance(manager.downloader.strategy, Filing13FDownloaderStrategy)
        assert isinstance(manager.parser.strategy, Filing13FParserStrategy)
    
    def test_manager_initializes_nport_strategies(self):
        """Should create appropriate strategies for NPORT-P form."""
        manager = SECFilingManager(form_type="NPORT-P", series_id="S000002839")
        
        assert isinstance(manager.downloader.strategy, FilingNPORTDownloaderStrategy)
        assert isinstance(manager.parser.strategy, FilingNPORTParserStrategy)
    
    def test_manager_passes_series_id_to_strategy(self):
        """Should pass series_id to NPORT downloader strategy."""
        manager = SECFilingManager(form_type="NPORT-P", series_id="S000002839")
        
        assert manager.downloader.strategy.series_id == "S000002839"
    
    def test_manager_raises_error_for_unsupported_form(self):
        """Should raise ValueError for unsupported form types."""
        with pytest.raises(ValueError, match="Unsupported form type"):
            SECFilingManager(form_type="INVALID-FORM")


# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestGetFilingByDate:
    """Tests for get_filing_by_date utility."""
    
    def test_finds_filing_with_exact_date_match(self):
        """Should return filing when date matches exactly."""
        filings = {
            ('123', pd.Timestamp('2024-09-30')): ({'name': 'Fund A'}, pd.DataFrame()),
            ('123', pd.Timestamp('2024-12-31')): ({'name': 'Fund B'}, pd.DataFrame()),
        }
        
        key, (metadata, df) = get_filing_by_date(filings, '2024-09-30')
        
        assert key[1] == pd.Timestamp('2024-09-30')
        assert metadata['name'] == 'Fund A'
    
    def test_raises_keyerror_when_date_not_found(self):
        """Should raise KeyError when no matching date exists."""
        filings = {
            ('123', pd.Timestamp('2024-09-30')): ({'name': 'Fund A'}, pd.DataFrame()),
        }
        
        with pytest.raises(KeyError, match="No filing found for date"):
            get_filing_by_date(filings, '2024-12-31')
    
    def test_handles_date_string_formats(self):
        """Should handle various date string formats."""
        filings = {
            ('123', pd.Timestamp('2024-09-30')): ({'name': 'Fund A'}, pd.DataFrame()),
        }
        
        # Different format should still match
        key, _ = get_filing_by_date(filings, '09-30-2024')
        
        assert key[1] == pd.Timestamp('2024-09-30')


# ============================================================================
# ERROR HANDLING TESTS
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and edge cases."""
    
    def test_parse_metadata_with_missing_file_returns_empty_dict(self, temp_filing_dir):
        """Should return empty dict when XML file doesn't exist."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        metadata = strategy.parse_metadata(temp_filing_dir / "nonexistent.xml")
        
        assert metadata == {}
    
    def test_parse_holdings_with_empty_xml_returns_empty_dataframe(self, temp_filing_dir):
        """Should return empty DataFrame for XML with no holdings."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        empty_xml = """<?xml version="1.0"?>
<informationTable xmlns="http://www.sec.gov/edgar/document/thirteenf/informationtable">
</informationTable>"""
        (temp_filing_dir / "infotable.xml").write_text(empty_xml)
        
        df = strategy.parse_holdings(temp_filing_dir)
        
        assert df.empty
    
    def test_parse_metadata_with_malformed_xml_raises_parsing_error(self, temp_filing_dir):
        """Should raise ParsingError for malformed XML."""
        xml_parser = XMLParser()
        strategy = Filing13FParserStrategy(xml_parser)
        
        malformed = "<invalid>xml<without</closing>"
        (temp_filing_dir / "primary_doc.xml").write_text(malformed)
        
        with pytest.raises(ParsingError):
            strategy.parse_metadata(temp_filing_dir / "primary_doc.xml")


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("cik,expected", [
    ("1037389", "0001037389"),
    ("0001037389", "0001037389"),
    ("  1037389  ", "0001037389"),
])
def test_cik_normalization(cik, expected):
    """CIK should be normalized to 10 digits with leading zeros."""
    normalized = cik.strip().replace('-', '').zfill(10)
    assert normalized == expected


@pytest.mark.parametrize("form_type,strategy_type", [
    ("13F-HR", Filing13FDownloaderStrategy),
    ("NPORT-P", FilingNPORTDownloaderStrategy),
    ("NPORT-N", FilingNPORTDownloaderStrategy),
])
def test_manager_creates_correct_strategy_for_form_type(form_type, strategy_type):
    """Manager should create appropriate strategy for each form type."""
    manager = SECFilingManager(form_type=form_type)
    assert isinstance(manager.downloader.strategy, strategy_type)


# ============================================================================
# RUN CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])