"""
Test Suite for SEC Filing Module (Version 41)

Updated to match the refactored module with:
- VerbosityLevel enum
- Refactored strategy classes
- New method signatures

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

print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX - Starting SEC Filing Module tests...")
# Import module
try:
    from sec_file_manager import (
        VerbosityLevel,
        SECConfig,
        SECFilingError,
        FilingNotFoundError,
        ParsingError,
        DefaultHTTPClient,
        XMLParser,
        FilingFileDiscoverer,
        FilingMetadata,
        Filing13FDownloader,
        FilingNPORTDownloader,
        Filing13FParser,
        FilingNPORTParser,
        SECFilingDownloader,
        SECFilingParser,
        SECFilingManager,
        get_filing_by_date,
        log
    )
    print("successfully imported sec_file_manager module for testing.")
except ImportError:
    print("Could not import sec_file_manager module. Skipping tests.")
    pytest.skip("Module not found", allow_module_level=True)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def config():
    """Test configuration with silent verbosity."""
    return SECConfig(
        user_agent="Test Agent test@example.com",
        rate_limit_delay=0.0,
        request_delay=0.0,
        verbosity=VerbosityLevel.SILENT  # No output during tests
    )


@pytest.fixture
def mock_http_client():
    """Mock HTTP client that doesn't make real requests."""
    client = Mock()
    client.get = Mock()
    return client


@pytest.fixture
def xml_parser():
    """XML parser utility."""
    return XMLParser()


@pytest.fixture
def sample_filing_metadata():
    """Sample filing metadata."""
    return FilingMetadata(
        cik="0001037389",
        accession_number="0001037389-24-000123",
        filing_date="2024-09-30",
        form_type="13F-HR"
    )


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
        <n>RENAISSANCE TECHNOLOGIES LLC</n>
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
      <repPdDate>2024-12-31</repPdDate>
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
# CONFIGURATION TESTS
# ============================================================================

class TestVerbosityLevel:
    """Tests for VerbosityLevel enum."""
    
    def test_verbosity_levels_ordered(self):
        """Verbosity levels should be in ascending order."""
        assert VerbosityLevel.SILENT < VerbosityLevel.ERROR
        assert VerbosityLevel.ERROR < VerbosityLevel.NORMAL
        assert VerbosityLevel.NORMAL < VerbosityLevel.VERBOSE
        assert VerbosityLevel.VERBOSE < VerbosityLevel.DEBUG
    
    def test_log_function_respects_verbosity(self, config):
        """Log function should only print if level >= config verbosity."""
        config.verbosity = VerbosityLevel.NORMAL
        
        # Should not print (VERBOSE > NORMAL)
        with patch('builtins.print') as mock_print:
            log("test", VerbosityLevel.VERBOSE, config)
            mock_print.assert_not_called()
        
        # Should print (NORMAL >= NORMAL)
        with patch('builtins.print') as mock_print:
            log("test", VerbosityLevel.NORMAL, config)
            mock_print.assert_called_once()


# ============================================================================
# XMLPARSER TESTS
# ============================================================================

class TestXMLParser:
    """Tests for XMLParser utility class."""
    
    def test_remove_namespaces_strips_prefixes(self):
        """Should remove namespace prefixes from tags."""
        xml = '<root xmlns="http://example.com"><child>text</child></root>'
        root = ET.fromstring(xml)
        
        cleaned = XMLParser.remove_namespaces(root)
        
        assert cleaned.tag == 'root'
        assert cleaned[0].tag == 'child'
    
    def test_to_dict_creates_hierarchical_keys(self):
        """Should create underscore-separated keys for nested elements."""
        xml = '<root><level1><level2>value</level2></level1></root>'
        root = ET.fromstring(xml)
        
        data = XMLParser.to_dict(root)
        
        assert 'level1_level2' in data
        assert data['level1_level2'] == 'value'
    
    def test_parse_file_returns_none_for_missing_file(self, tmp_path):
        """Should return None when file doesn't exist."""
        result = XMLParser.parse_file(tmp_path / "nonexistent.xml")
        assert result is None
    
    def test_find_all_locates_elements(self):
        """Should find elements regardless of namespace."""
        xml = '<root xmlns="http://ex.com"><target>1</target><target>2</target></root>'
        root = ET.fromstring(xml)
        
        elements = XMLParser.find_all(root, 'target')
        
        assert len(elements) == 2


# ============================================================================
# FILING METADATA TESTS
# ============================================================================

class TestFilingMetadata:
    """Tests for FilingMetadata dataclass."""
    
    def test_accession_no_dash_removes_dashes(self, sample_filing_metadata):
        """Should remove dashes from accession number."""
        assert sample_filing_metadata.accession_no_dash == "000103738924000123"
    
    def test_archive_url_constructs_correctly(self, sample_filing_metadata):
        """Should construct proper SEC archive URL."""
        url = sample_filing_metadata.archive_url("https://www.sec.gov")
        assert url == "https://www.sec.gov/Archives/edgar/data/1037389/000103738924000123"


# ============================================================================
# 13-F DOWNLOADER STRATEGY TESTS
# ============================================================================

class Test13FDownloaderStrategy:
    """Tests for 13-F downloader strategy."""
    
    def test_should_download_always_returns_true(self, mock_http_client, config):
        """13-F strategy downloads all filings without filtering."""
        strategy = Filing13FDownloader(mock_http_client, config)
        filing = FilingMetadata("123", "456", "2024-01-01", "13F-HR")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_get_files_uses_discoverer(self, mock_http_client, config):
        """Should use FilingFileDiscoverer to find files."""
        strategy = Filing13FDownloader(mock_http_client, config)
        
        # Mock the discoverer
        with patch.object(strategy.discoverer, 'discover', return_value=['file1.xml', 'file2.xml']):
            files = strategy.get_files("http://example.com")
        
        assert files == ['file1.xml', 'file2.xml']


# ============================================================================
# NPORT DOWNLOADER STRATEGY TESTS
# ============================================================================

class TestNPORTDownloaderStrategy:
    """Tests for NPORT downloader strategy."""
    
    def test_should_download_with_no_filter_returns_true(self, mock_http_client, config):
        """When no series_id specified, should download all filings."""
        strategy = FilingNPORTDownloader(mock_http_client, config, series_id=None)
        filing = FilingMetadata("123", "456", "2024-01-01", "NPORT-P")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_should_download_with_matching_series(self, mock_http_client, config, sample_nport_primary_xml):
        """Should return True when series ID matches."""
        mock_response = Mock()
        mock_response.content = sample_nport_primary_xml.encode()
        mock_http_client.get.return_value = mock_response
        
        strategy = FilingNPORTDownloader(mock_http_client, config, series_id="S000002839")
        filing = FilingMetadata("36405", "456", "2024-01-01", "NPORT-P")
        
        result = strategy.should_download(filing, "http://example.com")
        
        assert result is True
    
    def test_get_files_returns_primary_doc_only(self, mock_http_client, config):
        """NPORT should only require primary_doc.xml."""
        strategy = FilingNPORTDownloader(mock_http_client, config)
        
        files = strategy.get_files("http://example.com")
        
        assert files == ["primary_doc.xml"]


# ============================================================================
# 13-F PARSER STRATEGY TESTS
# ============================================================================

class Test13FParserStrategy:
    """Tests for 13-F parser strategy."""
    
    def test_parse_metadata_extracts_cik(self, mock_http_client, config, filing_with_13f_files):
        """Should extract CIK from 13-F primary document."""
        strategy = Filing13FParser(mock_http_client, config)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        assert metadata['cik'] == '0001037389'
    
    def test_parse_metadata_extracts_period(self, mock_http_client, config, filing_with_13f_files):
        """Should extract period of report."""
        strategy = Filing13FParser(mock_http_client, config)
        
        metadata = strategy.parse_metadata(filing_with_13f_files / "primary_doc.xml")
        
        assert metadata['periodofreport'] == '09-30-2024'
    
    def test_parse_holdings_creates_dataframe(self, mock_http_client, config, filing_with_13f_files):
        """Should parse 13-F holdings into DataFrame."""
        strategy = Filing13FParser(mock_http_client, config)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert len(df) == 2
        assert 'nameofissuer' in df.columns
        assert 'value' in df.columns
    
    def test_parse_holdings_calculates_unit_value(self, mock_http_client, config, filing_with_13f_files):
        """Should calculate unitValue correctly."""
        strategy = Filing13FParser(mock_http_client, config)
        
        df = strategy.parse_holdings(filing_with_13f_files)
        
        assert 'unitValue' in df.columns
        # value=1000000, shares=10000 -> unitValue=100
        assert df['unitValue'].iloc[0] == 100.0
    
    def test_get_aliases_returns_correct_mapping(self, mock_http_client, config):
        """Should return correct alias mappings."""
        strategy = Filing13FParser(mock_http_client, config)
        
        aliases = strategy.get_aliases()
        
        assert 'cik' in aliases
        assert 'periodofreport' in aliases
        assert 'name' in aliases


# ============================================================================
# NPORT PARSER STRATEGY TESTS
# ============================================================================

class TestNPORTParserStrategy:
    """Tests for NPORT parser strategy."""
    
    def test_parse_metadata_extracts_series_info(self, mock_http_client, config, filing_with_nport_files):
        """Should extract series information from NPORT."""
        strategy = FilingNPORTParser(mock_http_client, config)
        
        metadata = strategy.parse_metadata(filing_with_nport_files / "primary_doc.xml")
        
        assert metadata['cik'] == '0000036405'
        assert metadata['name'] == 'VANGUARD 500 INDEX FUND'
        assert metadata['seriesid'] == 'S000002839'
    
    def test_parse_holdings_from_single_file(self, mock_http_client, config, filing_with_nport_files):
        """Should parse holdings from primary_doc.xml."""
        strategy = FilingNPORTParser(mock_http_client, config)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert len(df) == 2
        assert 'n' in df.columns
        assert 'valusd' in df.columns
    
    def test_parse_holdings_calculates_unit_value(self, mock_http_client, config, filing_with_nport_files):
        """Should calculate unitValue for NPORT holdings."""
        strategy = FilingNPORTParser(mock_http_client, config)
        
        df = strategy.parse_holdings(filing_with_nport_files)
        
        assert 'unitValue' in df.columns
        # valUSD=15000000, balance=100000 -> unitValue=150
        assert df['unitValue'].iloc[0] == 150.0


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestSECFilingManager:
    """Integration tests for manager facade."""
    
    def test_manager_initializes_13f_strategies(self):
        """Should create appropriate strategies for 13F-HR form."""
        manager = SECFilingManager(form_type="13F-HR", verbosity=VerbosityLevel.SILENT)
        
        assert isinstance(manager.downloader.strategy, Filing13FDownloader)
        assert isinstance(manager.parser.strategy, Filing13FParser)
    
    def test_manager_initializes_nport_strategies(self):
        """Should create appropriate strategies for NPORT-P form."""
        manager = SECFilingManager(form_type="NPORT-P", series_id="S000002839", 
                                   verbosity=VerbosityLevel.SILENT)
        
        assert isinstance(manager.downloader.strategy, FilingNPORTDownloader)
        assert isinstance(manager.parser.strategy, FilingNPORTParser)
    
    def test_manager_passes_series_id_to_strategy(self):
        """Should pass series_id to NPORT downloader strategy."""
        manager = SECFilingManager(form_type="NPORT-P", series_id="S000002839",
                                   verbosity=VerbosityLevel.SILENT)
        
        assert manager.downloader.strategy.series_id == "S000002839"
    
    def test_manager_raises_error_for_unsupported_form(self):
        """Should raise ValueError for unsupported form types."""
        with pytest.raises(ValueError, match="Unsupported form type"):
            SECFilingManager(form_type="INVALID-FORM")
    
    def test_verbosity_can_be_set_at_init(self):
        """Verbosity should be configurable at initialization."""
        manager = SECFilingManager(form_type="13F-HR", verbosity=VerbosityLevel.DEBUG)
        
        assert manager.config.verbosity == VerbosityLevel.DEBUG


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
        
        with pytest.raises(KeyError, match="No filing found"):
            get_filing_by_date(filings, '2024-12-31')


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

@pytest.mark.parametrize("form_type,expected_downloader,expected_parser", [
    ("13F-HR", Filing13FDownloader, Filing13FParser),
    ("NPORT-P", FilingNPORTDownloader, FilingNPORTParser),
    ("NPORT-N", FilingNPORTDownloader, FilingNPORTParser),
])
def test_manager_creates_correct_strategies(form_type, expected_downloader, expected_parser):
    """Manager should create appropriate strategies for each form type."""
    manager = SECFilingManager(form_type=form_type, verbosity=VerbosityLevel.SILENT)
    
    assert isinstance(manager.downloader.strategy, expected_downloader)
    assert isinstance(manager.parser.strategy, expected_parser)


# ============================================================================
# RUN CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])