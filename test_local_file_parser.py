"""
Test suite for LocalFileParser class from .sec_file_manager.py

This module tests the LocalFileParser class, which parses SEC filings from local XML files
without downloading them. It's useful for analyzing previously downloaded filings, testing
with sample files, and batch processing existing archives.

RUNNING TESTS
=============

From Command Line:
------------------
IMPORTANT: Use 'pytest' command, NOT 'python'!

# First, install pytest if you haven't already
pip install pytest pytest-cov

# Run all tests in this file
pytest test_local_file_parser.py -v

# Run specific test class
pytest test_local_file_parser.py::TestParseFile -v

# Run specific test
pytest test_local_file_parser.py::TestParseFile::test_success -v

# Run with coverage report
pytest test_local_file_parser.py --cov=.sec_file_manager --cov-report=html

# Run with output (see print statements)
pytest test_local_file_parser.py -v -s

# Run tests matching pattern
pytest test_local_file_parser.py -k "error" -v

# Quick run (less verbose)
pytest test_local_file_parser.py

# Stop at first failure
pytest test_local_file_parser.py -x


From VSCode:
------------
1. Install Python Test Explorer extension
2. Open Testing panel (beaker icon in sidebar)
3. Click "Configure Python Tests" → select "pytest"
4. Tests will appear in Testing panel
5. Click play button next to any test to run it
6. Set breakpoints and click debug icon to debug tests

Alternative VSCode method:
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P)
2. Type "Python: Run All Tests" or "Python: Debug All Tests"


EXAMPLES
========

Basic Usage:
------------
# The actual code being tested would be used like:
from .sec_file_manager import LocalFileParser

parser = LocalFileParser(form_type="13F-HR")

# Parse single filing directory
metadata, holdings = parser.parse_file("path/to/filing_dir")
print(f"CIK: {metadata['cik']}, Holdings: {len(holdings)}")

# Parse multiple directories
results = parser.parse_directories(["archive_2023", "archive_2024"])
for (cik, period_date), (metadata, holdings) in results.items():
    print(f"{cik} - {period_date}: {len(holdings)} holdings")


What These Tests Verify:
-------------------------
1. Initialization: Valid/invalid form types, custom configs
2. File parsing: Success cases, missing files, error handling
3. Batch operations: Multiple files, error resilience
4. Directory scanning: Pattern matching, subdirectory discovery
5. Date handling: Valid dates, invalid dates, NaT handling
"""

import pytest
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from .sec_file_manager import LocalFileParser, SECConfig, VerbosityLevel, get_filing_by_date


# =============================================================================
# FIXTURES - Reusable test components
# =============================================================================

@pytest.fixture
def mock_strategy():
    """
    Mock parsing strategy with default behavior.
    
    The strategy is responsible for parsing metadata and holdings from XML files.
    This fixture provides a mock that returns standard test data without needing
    actual XML files.
    
    Returns:
        Mock: Strategy object with parse_metadata and parse_holdings methods
        
    Example usage in test:
        def test_something(mock_strategy):
            # mock_strategy already configured with default returns
            metadata = mock_strategy.parse_metadata("path")
            assert metadata['cik'] == '123'
    """
    strategy = Mock()
    strategy.parse_metadata.return_value = {
        'cik': '123',
        'periodofreport': '2024-12-31'
    }
    strategy.parse_holdings.return_value = pd.DataFrame({
        'holding': ['AAPL', 'GOOGL']
    })
    return strategy


@pytest.fixture
def mock_strategies():
    """Mock the SECFilingManager.STRATEGIES dictionary."""
    mock_parser_cls = Mock()
    return {'13F-HR': (Mock, mock_parser_cls), 'NPORT-P': (Mock, mock_parser_cls)}


@pytest.fixture
def parser(mock_strategy, mock_strategies):
    """
    Create LocalFileParser instance with mocked dependencies.
    
    This fixture:
    - Patches the SECFilingManager.STRATEGIES to avoid real strategy initialization
    - Patches the log function to avoid logging during tests
    - Injects the mock_strategy to control parsing behavior
    
    Returns:
        LocalFileParser: Ready-to-test parser instance
        
    Example usage:
        def test_parse_operation(parser):
            # parser is ready to use with all dependencies mocked
            results = parser.parse_file("/some/path")
    """
    with patch('.sec_file_manager.SECFilingManager.STRATEGIES', mock_strategies), \
         patch('.sec_file_manager.log'):
        p = LocalFileParser(form_type="13F-HR")
        p.strategy = mock_strategy
        return p


# =============================================================================
# INITIALIZATION TESTS
# =============================================================================

class TestInit:
    """
    Test LocalFileParser initialization and configuration.
    
    These tests verify that:
    - Parser accepts valid form types (13F-HR, NPORT-P, etc.)
    - Parser rejects invalid form types
    - Custom configuration can be provided
    """
    
    def test_valid_form_type(self, mock_strategies):
        """
        Parser should initialize successfully with supported form type.
        
        Example form types: "13F-HR", "NPORT-P"
        """
        with patch('.sec_file_manager.SECFilingManager.STRATEGIES', mock_strategies):
            parser = LocalFileParser(form_type="13F-HR")
            assert parser.form_type == "13F-HR"
            assert parser.config is not None
            assert isinstance(parser.config, SECConfig)
    
    def test_invalid_form_type(self):
        """
        Parser should raise ValueError for unsupported form types.
        
        This prevents accidentally using the wrong parser for a form type.
        """
        with patch('.sec_file_manager.SECFilingManager.STRATEGIES', {}):
            with pytest.raises(ValueError, match="Unsupported form type: INVALID"):
                LocalFileParser(form_type="INVALID")
    
    def test_custom_config(self, mock_strategies):
        """
        Parser should accept and use custom SECConfig.
        
        Allows customizing verbosity, timeouts, and other settings:
            config = SECConfig(verbosity=VerbosityLevel.QUIET)
            parser = LocalFileParser(form_type="13F-HR", config=config)
        """
        mock_config = SECConfig(verbosity=VerbosityLevel.SILENT)
        with patch('.sec_file_manager.SECFilingManager.STRATEGIES', mock_strategies):
            parser = LocalFileParser(form_type="13F-HR", config=mock_config)
            assert parser.config is mock_config
            assert parser.config.verbosity == VerbosityLevel.SILENT


# =============================================================================
# SINGLE FILE PARSING TESTS
# =============================================================================

class TestParseFile:
    """
    Test parsing individual filing directories.
    
    Each filing directory contains:
    - primary_doc.xml: Main filing document with metadata
    - infotable.xml: Holdings information (for 13F-HR)
    - Other supporting files
    
    The parse_file method returns:
        - metadata: Dict with CIK, period of report, etc.
        - holdings: DataFrame with investment holdings
    """
    
    def test_success(self, parser, mock_strategy):
        """
        Should successfully parse valid filing directory.
        
        Verifies:
        - Directory existence check passes
        - Metadata parsing is called
        - Holdings parsing is called
        - Correct data is returned
        """
        with patch('.sec_file_manager.Path') as mock_path_cls:
            # Mock filesystem to simulate existing directory
            mock_path = MagicMock()
            mock_path.exists.return_value = True
            mock_path.__truediv__ = Mock(return_value=mock_path)  # For / operator
            mock_path_cls.return_value = mock_path
            
            metadata, holdings = parser.parse_file("/path/to/filing")
            
            assert metadata['cik'] == '123'
            assert len(holdings) == 2
            mock_strategy.parse_metadata.assert_called_once()
            mock_strategy.parse_holdings.assert_called_once()
    
    def test_directory_not_found(self, parser):
        """
        Should raise FileNotFoundError for missing directories.
        
        This prevents silent failures when paths are incorrect.
        Example error case:
            parser.parse_file("/wrong/path/to/filing")
            # Raises: FileNotFoundError("Directory not found: /wrong/path/to/filing")
        """
        with patch('.sec_file_manager.Path') as mock_path_cls:
            mock_path = MagicMock()
            mock_path.exists.return_value = False
            mock_path_cls.return_value = mock_path
            
            with pytest.raises(FileNotFoundError, match="Directory not found"):
                parser.parse_file("/nonexistent")


# =============================================================================
# BATCH FILE PARSING TESTS
# =============================================================================

class TestParseFiles:
    """
    Test parsing multiple filing directories in one call.
    
    parse_files() processes a list of paths and returns a dictionary keyed by
    (cik, period_date) tuples. This is useful for batch processing.
    
    Example usage:
        results = parser.parse_files([
            "/filings/0001234_2024Q4",
            "/filings/0001234_2024Q3",
            "/filings/0005678_2024Q4"
        ])
        
        # Returns:
        # {
        #   ('1234', Timestamp('2024-12-31')): (metadata, holdings_df),
        #   ('1234', Timestamp('2024-09-30')): (metadata, holdings_df),
        #   ('5678', Timestamp('2024-12-31')): (metadata, holdings_df)
        # }
    """
    
    def test_multiple_files(self, parser):
        """
        Should parse all files and return keyed dictionary.
        
        The (cik, period_date) key allows easy lookup of specific filings:
            filing = results[('1234', pd.Timestamp('2024-12-31'))]
        """
        parser.parse_file = Mock(side_effect=[
            ({'cik': '123', 'periodofreport': '2024-12-31'}, pd.DataFrame()),
            ({'cik': '456', 'periodofreport': '2024-09-30'}, pd.DataFrame())
        ])
        
        results = parser.parse_files(["/path1", "/path2"])
        
        assert len(results) == 2
        assert ('123', pd.Timestamp('2024-12-31')) in results
        assert ('456', pd.Timestamp('2024-09-30')) in results
    
    def test_error_handling(self, parser):
        """
        Should continue processing after individual file errors.
        
        Critical for batch operations - one bad file shouldn't stop the entire batch.
        Errors are logged but don't halt processing:
            paths = ["/good1", "/bad", "/good2"]  # bad file has parse error
            results = parser.parse_files(paths)
            # Returns 2 results (from good1 and good2), logs error for bad
        """
        parser.parse_file = Mock(side_effect=[
            ({'cik': '123', 'periodofreport': '2024-12-31'}, pd.DataFrame()),
            Exception("Parse error")
        ])
        
        results = parser.parse_files(["/path1", "/path2"])
        
        assert len(results) == 1  # Only successful parse included
        assert ('123', pd.Timestamp('2024-12-31')) in results
    
    def test_invalid_date_handling(self, parser):
        """
        Should handle malformed dates gracefully.
        
        If a filing has an invalid date string, it's stored as NaT (Not a Time)
        rather than causing a crash. This allows the filing to still be processed:
            metadata = {'periodofreport': 'invalid-date-string'}
            # Results in key: ('123', NaT) instead of crashing
        """
        parser.parse_file = Mock(return_value=(
            {'cik': '123', 'periodofreport': 'invalid-date'}, 
            pd.DataFrame()
        ))
        
        results = parser.parse_files(["/path1"])
        
        key = list(results.keys())[0]
        assert key[0] == '123'
        assert pd.isna(key[1])  # Date should be NaT


# =============================================================================
# DIRECTORY SCANNING TESTS
# =============================================================================

class TestParseDirectory:
    """
    Test scanning directories for filing subdirectories.
    
    parse_directory() finds all subdirectories matching a pattern and parses them.
    This is the most common way to process downloaded filings organized by directory.
    
    Example directory structure:
        sec_filings/
        ├── 0001234_20241231/
        │   ├── primary_doc.xml
        │   └── infotable.xml
        ├── 0001234_20240930/
        │   ├── primary_doc.xml
        │   └── infotable.xml
        └── 0005678_20241231/
            ├── primary_doc.xml
            └── infotable.xml
    
    Usage:
        # Parse all filings
        results = parser.parse_directory("sec_filings")
        
        # Parse specific CIK only
        results = parser.parse_directory("sec_filings", "0001234_*")
    """
    
    def test_finds_and_parses_subdirectories(self, parser):
        """
        Should discover subdirectories and parse them.
        
        Uses glob patterns to find matching directories, then calls parse_files().
        """
        with patch('.sec_file_manager.Path') as mock_path_cls:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            
            # Create mock subdirectories
            mock_subdir1 = MagicMock()
            mock_subdir1.is_dir.return_value = True
            mock_subdir1.__str__ = Mock(return_value="/parent/subdir1")
            
            mock_subdir2 = MagicMock()
            mock_subdir2.is_dir.return_value = True
            mock_subdir2.__str__ = Mock(return_value="/parent/subdir2")
            
            mock_dir.glob.return_value = [mock_subdir1, mock_subdir2]
            mock_path_cls.return_value = mock_dir
            
            parser.parse_files = Mock(return_value={'key': 'value'})
            results = parser.parse_directory("/parent")
            
            parser.parse_files.assert_called_once()
            # Check that parse_files was called with list of 2 paths
            call_args = parser.parse_files.call_args[0][0]
            assert len(call_args) == 2
            assert results == {'key': 'value'}
    
    def test_directory_not_found(self, parser):
        """
        Should raise FileNotFoundError for missing parent directory.
        
        Example:
            parser.parse_directory("/nonexistent/path")
            # Raises: FileNotFoundError
        """
        with patch('.sec_file_manager.Path') as mock_path_cls:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = False
            mock_path_cls.return_value = mock_dir
            
            with pytest.raises(FileNotFoundError):
                parser.parse_directory("/nonexistent")
    
    def test_no_matching_subdirectories(self, parser):
        """
        Should return empty dict when no subdirectories match pattern.
        
        Example scenarios:
        - Directory is empty
        - Pattern doesn't match any subdirectories
        - Directory contains only files (no subdirectories)
        
        This is not an error - just returns {} and logs a message.
        """
        with patch('.sec_file_manager.Path') as mock_path_cls:
            mock_dir = MagicMock()
            mock_dir.exists.return_value = True
            mock_dir.glob.return_value = []
            mock_path_cls.return_value = mock_dir
            
            results = parser.parse_directory("/parent", "pattern*")
            
            assert results == {}


# =============================================================================
# MULTIPLE DIRECTORY TESTS
# =============================================================================

class TestParseDirectories:
    """
    Test parsing filings from multiple parent directories.
    
    Useful for processing filings organized by year or other criteria:
    
    Structure:
        archives/
        ├── 2023/
        │   ├── filing1/
        │   └── filing2/
        ├── 2024/
        │   ├── filing3/
        │   └── filing4/
        └── 2025/
            └── filing5/
    
    Usage:
        results = parser.parse_directories([
            "archives/2023",
            "archives/2024",
            "archives/2025"
        ])
        # Returns combined dictionary from all three directories
    """
    
    def test_combines_results(self, parser):
        """
        Should merge results from all directories into single dictionary.
        
        Results are combined using dictionary update, so later directories
        can overwrite earlier ones if they have the same (cik, period_date) key.
        """
        parser.parse_directory = Mock(side_effect=[
            {('123', pd.Timestamp('2024-12-31')): ({}, pd.DataFrame())},
            {('456', pd.Timestamp('2024-09-30')): ({}, pd.DataFrame())}
        ])
        
        results = parser.parse_directories(["/dir1", "/dir2"])
        
        assert len(results) == 2
        assert ('123', pd.Timestamp('2024-12-31')) in results
        assert ('456', pd.Timestamp('2024-09-30')) in results
    
    def test_error_resilience(self, parser):
        """
        Should continue processing remaining directories after errors.
        
        If one directory fails (doesn't exist, permission error, etc.),
        the parser logs the error and continues with remaining directories.
        
        Example:
            dirs = ["/good1", "/missing", "/good2"]
            results = parser.parse_directories(dirs)
            # Returns results from good1 and good2, logs error for missing
        """
        parser.parse_directory = Mock(side_effect=[
            {('123', pd.Timestamp('2024-12-31')): ({}, pd.DataFrame())},
            Exception("Directory error")
        ])
        
        results = parser.parse_directories(["/dir1", "/dir2"])
        
        assert len(results) == 1  # Only successful directory included


# =============================================================================
# UTILITY METHOD TESTS
# =============================================================================

class TestGetFilingByDate:
    """
    Test convenience method for retrieving filings by date.
    
    This is a simple wrapper around the module-level get_filing_by_date() function
    for easier access within the class.
    
    Usage:
        results = parser.parse_directory("filings")
        filing = parser.get_filing_by_date(results, "2024-12-31")
        # Returns: ((cik, timestamp), (metadata, holdings_df))
    """
    
    def test_delegates_to_module_function(self, parser):
        """
        Should call module-level function with correct arguments.
        
        The method is just a convenience wrapper - all logic is in the
        module-level function to allow usage without a parser instance.
        """
        mock_filings = {('123', pd.Timestamp('2024-12-31')): ({}, pd.DataFrame())}
        
        with patch('.sec_file_manager.get_filing_by_date') as mock_get:
            mock_get.return_value = "result"
            result = parser.get_filing_by_date(mock_filings, "2024-12-31")
            
            mock_get.assert_called_once_with(mock_filings, "2024-12-31")
            assert result == "result"
    
    def test_actual_function_finds_filing(self):
        """
        Test the actual get_filing_by_date function works correctly.
        
        This tests the real implementation to ensure date matching works.
        """
        mock_metadata = {'cik': '123', 'periodofreport': '2024-12-31'}
        mock_holdings = pd.DataFrame({'holding': ['AAPL']})
        filings = {
            ('123', pd.Timestamp('2024-12-31')): (mock_metadata, mock_holdings),
            ('456', pd.Timestamp('2024-09-30')): ({}, pd.DataFrame())
        }
        
        result = get_filing_by_date(filings, "2024-12-31")
        
        assert result[0] == ('123', pd.Timestamp('2024-12-31'))
        assert result[1][0] == mock_metadata
        assert len(result[1][1]) == 1
    
    def test_actual_function_raises_on_not_found(self):
        """
        Test that get_filing_by_date raises KeyError when date not found.
        """
        filings = {
            ('123', pd.Timestamp('2024-12-31')): ({}, pd.DataFrame())
        }
        
        with pytest.raises(KeyError, match="No filing found for date"):
            get_filing_by_date(filings, "2024-09-30")