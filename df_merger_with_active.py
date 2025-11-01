import pandas as pd
import unittest
from typing import List, Tuple, Optional, Dict, Union


class DataFrameMerger:
    """
    Efficiently merges multiple DataFrames with date indices.
    
    This class handles the common task of combining multiple time-series DataFrames,
    keeping only rows and columns that exist in all DataFrames. Optionally computes
    active (benchmark-relative) performance for a specified column.
    
    Features:
    - Automatic name generation for unnamed DataFrames
    - Lazy evaluation and caching of expensive computations
    - Memory-efficient operations using views where possible
    - Support for active return calculations (returns relative to benchmark)
    
    Example:
        >>> merger = DataFrameMerger([df1, df2, df3])  # Auto-named as 0, 1, 2
        >>> result = merger.merge()
        >>> 
        >>> # Or with explicit names and active returns
        >>> merger = DataFrameMerger([(df1, 'A'), (df2, 'B'), (df3, 'C')])
        >>> result = merger.merge(benchmark_config=('returns', 'A'))
    """
    
    # Class constant for active column naming convention
    ACTIVE_COL_PREFIX = 'active'
    
    def __init__(
        self, 
        dataframes: Union[List[pd.DataFrame], List[Tuple[pd.DataFrame, Union[str, int]]]]
    ):
        """
        Initialize merger with DataFrames and optional names.
        
        Args:
            dataframes: Either a list of DataFrames (names auto-generated as indices)
                       or list of (DataFrame, name) tuples
        
        Raises:
            ValueError: If no DataFrames provided or names are not unique
        """
        if not dataframes:
            raise ValueError("At least one dataframe must be provided")
        
        # Detect input format and extract DataFrames and names
        if isinstance(dataframes[0], tuple):
            self.dfs, self.names = zip(*dataframes)
            self.dfs, self.names = list(self.dfs), list(self.names)
        else:
            self.dfs = list(dataframes)
            self.names = list(range(len(dataframes)))
        
        # Validate unique names
        if len(self.names) != len(set(self.names)):
            raise ValueError("DataFrame names must be unique")
        
        # Create O(1) lookup dictionary for DataFrames by name
        self._df_dict: Dict[Union[str, int], pd.DataFrame] = dict(zip(self.names, self.dfs))
        
        # Cache for expensive computations (lazy evaluation)
        self._common_cols: Optional[List[str]] = None
        self._common_index: Optional[pd.Index] = None
    
    @property
    def common_columns(self) -> List[str]:
        """
        Get columns present in all DataFrames (computed once, then cached).
        
        Returns:
            Sorted list of common column names
            
        Raises:
            ValueError: If no common columns found
        """
        if self._common_cols is None:
            # Efficient set intersection across all DataFrames
            self._common_cols = sorted(
                set.intersection(*[set(df.columns) for df in self.dfs])
            )
            if not self._common_cols:
                raise ValueError("No common columns found across all DataFrames")
        return self._common_cols
    
    @property
    def common_index(self) -> pd.Index:
        """
        Get index values present in all DataFrames (computed once, then cached).
        
        Returns:
            Index with common values across all DataFrames
            
        Raises:
            ValueError: If no common index values found
        """
        if self._common_index is None:
            # Chain intersection for memory efficiency (vs creating intermediate sets)
            self._common_index = self.dfs[0].index
            for df in self.dfs[1:]:
                self._common_index = self._common_index.intersection(df.index)
            
            if len(self._common_index) == 0:
                raise ValueError("No common index values found across all DataFrames")
        return self._common_index
    
    @staticmethod
    def get_active_column_name(benchmark_name: Union[str, int]) -> str:
        """
        Generate standardized active column name.
        
        Args:
            benchmark_name: Name of benchmark DataFrame
            
        Returns:
            Active column name in format 'active_{benchmark_name}'
        """
        return f'{DataFrameMerger.ACTIVE_COL_PREFIX}_{benchmark_name}'
    
    def merge(
        self, 
        benchmark_config: Optional[Tuple[str, Union[str, int]]] = None,
        sort: bool = False
    ) -> pd.DataFrame:
        """
        Merge DataFrames with hierarchical column structure.
        
        Args:
            benchmark_config: Optional (column_name, benchmark_name) tuple.
                            Adds active column: df[col] - benchmark[col]
                            Named as 'active_{benchmark_name}'
            sort: Whether to sort result by index (default False for efficiency)
        
        Returns:
            DataFrame with MultiIndex columns (name, column)
            
        Raises:
            ValueError: If benchmark column not in all DataFrames
            ValueError: If benchmark name doesn't exist
            
        Example:
            Result structure with benchmark:
                          A                              B                              C
                      price volume active_A         price volume active_A         price volume active_A
            2024-01-02   101   1100      0.0            50   2000    -51.0           201    550    100.0
        """
        # Prepare DataFrames efficiently (select common rows/cols, add active if needed)
        prepared_dfs = self._prepare_dataframes(benchmark_config)
        
        # Concatenate with hierarchical columns (copy=False avoids unnecessary copies)
        return pd.concat(
            prepared_dfs, 
            axis=1, 
            keys=self.names,
            copy=False,
            sort=sort
        )
    
    def get_dataframe(self, name: Union[str, int]) -> pd.DataFrame:
        """
        Retrieve a specific DataFrame by name.
        
        Args:
            name: DataFrame name (string or integer index)
            
        Returns:
            The requested DataFrame
            
        Raises:
            KeyError: If name not found
        """
        if name not in self._df_dict:
            raise KeyError(f"DataFrame '{name}' not found. Available: {self.names}")
        return self._df_dict[name]
    
    def _prepare_dataframes(
        self, 
        benchmark_config: Optional[Tuple[str, Union[str, int]]]
    ) -> List[pd.DataFrame]:
        """
        Prepare DataFrames for merging: select common data and add active columns.
        
        This method:
        1. Selects only common rows and columns (using view for efficiency)
        2. If benchmark specified, validates and adds active performance column
        
        Args:
            benchmark_config: Optional (column_name, benchmark_name) tuple
            
        Returns:
            List of prepared DataFrames ready for concatenation
        """
        # Select common index and columns in single operation (creates view, not copy)
        prepared_dfs = [
            df.loc[self.common_index, self.common_columns] 
            for df in self.dfs
        ]
        
        # Add active (benchmark-relative) columns if requested
        if benchmark_config is not None:
            col_name, benchmark_name = benchmark_config
            self._validate_benchmark_config(col_name, benchmark_name)
            
            # Extract benchmark series once (avoid repeated lookups)
            benchmark_series = self._df_dict[benchmark_name].loc[
                self.common_index, col_name
            ]
            
            # Copy DataFrames before modification (avoids SettingWithCopyWarning)
            # Only copy when necessary (i.e., when adding new column)
            prepared_dfs = [df.copy() for df in prepared_dfs]
            
            # Vectorized subtraction for each DataFrame
            # Active return = portfolio return - benchmark return
            active_col_name = self.get_active_column_name(benchmark_name)
            for df in prepared_dfs:
                df[active_col_name] = df[col_name] - benchmark_series
        
        return prepared_dfs
    
    def _validate_benchmark_config(
        self, 
        col_name: str, 
        benchmark_name: Union[str, int]
    ) -> None:
        """
        Validate benchmark configuration parameters.
        
        Args:
            col_name: Column name to use for benchmark comparison
            benchmark_name: Name of benchmark DataFrame
            
        Raises:
            ValueError: If column not in all DataFrames or benchmark name not found
        """
        if col_name not in self.common_columns:
            raise ValueError(
                f"Column '{col_name}' not found in all DataFrames. "
                f"Common columns: {self.common_columns}"
            )
        
        if benchmark_name not in self.names:
            raise ValueError(
                f"Benchmark DataFrame '{benchmark_name}' not found. "
                f"Available names: {self.names}"
            )


class TestDataFrameMerger(unittest.TestCase):
    """Comprehensive unit tests for DataFrameMerger class."""
    
    def setUp(self):
        """Create sample DataFrames for testing."""
        # Create DataFrames with overlapping indices
        self.dates1 = pd.date_range('2024-01-01', periods=5, freq='D')
        self.dates2 = pd.date_range('2024-01-02', periods=5, freq='D')
        self.dates3 = pd.date_range('2024-01-01', periods=6, freq='D')
        
        self.df1 = pd.DataFrame({
            'price': [100, 101, 102, 103, 104],
            'volume': [1000, 1100, 1200, 1300, 1400],
            'returns': [0.01, 0.02, 0.015, 0.01, 0.02]
        }, index=self.dates1)
        
        self.df2 = pd.DataFrame({
            'price': [50, 51, 52, 53, 54],
            'volume': [2000, 2100, 2200, 2300, 2400],
            'returns': [0.02, 0.01, 0.015, 0.02, 0.01]
        }, index=self.dates2)
        
        self.df3 = pd.DataFrame({
            'price': [200, 201, 202, 203, 204, 205],
            'volume': [500, 550, 600, 650, 700, 750],
            'returns': [0.005, 0.01, 0.012, 0.008, 0.015, 0.02]
        }, index=self.dates3)
    
    def test_auto_generated_names(self):
        """Test that auto-generated names work correctly."""
        merger = DataFrameMerger([self.df1, self.df2, self.df3])
        self.assertEqual(merger.names, [0, 1, 2])
    
    def test_explicit_names(self):
        """Test that explicit names are properly stored."""
        merger = DataFrameMerger([
            (self.df1, 'A'), 
            (self.df2, 'B'), 
            (self.df3, 'C')
        ])
        self.assertEqual(merger.names, ['A', 'B', 'C'])
    
    def test_empty_dataframes_raises_error(self):
        """Test that empty list of DataFrames raises ValueError."""
        with self.assertRaises(ValueError) as context:
            DataFrameMerger([])
        self.assertIn("At least one dataframe", str(context.exception))
    
    def test_duplicate_names_raises_error(self):
        """Test that duplicate names raise ValueError."""
        with self.assertRaises(ValueError) as context:
            DataFrameMerger([
                (self.df1, 'A'), 
                (self.df2, 'A')
            ])
        self.assertIn("unique", str(context.exception))
    
    def test_common_columns(self):
        """Test that common columns are correctly identified."""
        merger = DataFrameMerger([self.df1, self.df2, self.df3])
        expected = ['price', 'returns', 'volume']
        self.assertEqual(merger.common_columns, expected)
    
    def test_common_index(self):
        """Test that common index is correctly computed."""
        merger = DataFrameMerger([self.df1, self.df2, self.df3])
        # Common dates are 2024-01-02 through 2024-01-05 (4 dates)
        self.assertEqual(len(merger.common_index), 4)
        self.assertTrue(pd.Timestamp('2024-01-02') in merger.common_index)
        self.assertTrue(pd.Timestamp('2024-01-05') in merger.common_index)
    
    def test_no_common_columns_raises_error(self):
        """Test that DataFrames with no common columns raise ValueError."""
        df_no_common = pd.DataFrame({'unique': [1, 2, 3]}, index=self.dates1[:3])
        merger = DataFrameMerger([self.df1, df_no_common])
        with self.assertRaises(ValueError) as context:
            _ = merger.common_columns
        self.assertIn("No common columns", str(context.exception))
    
    def test_no_common_index_raises_error(self):
        """Test that DataFrames with no common index raise ValueError."""
        dates_no_overlap = pd.date_range('2025-01-01', periods=3, freq='D')
        df_no_overlap = pd.DataFrame({'price': [1, 2, 3]}, index=dates_no_overlap)
        merger = DataFrameMerger([self.df1, df_no_overlap])
        with self.assertRaises(ValueError) as context:
            _ = merger.common_index
        self.assertIn("No common index", str(context.exception))
    
    def test_basic_merge(self):
        """Test basic merge without benchmark."""
        merger = DataFrameMerger([
            (self.df1, 'A'), 
            (self.df2, 'B'), 
            (self.df3, 'C')
        ])
        result = merger.merge()
        
        # Check structure
        self.assertIsInstance(result.columns, pd.MultiIndex)
        self.assertEqual(result.columns.levels[0].tolist(), ['A', 'B', 'C'])
        self.assertEqual(len(result), 4)  # 4 common dates
    
    def test_merge_with_benchmark_integer_name(self):
        """Test merge with benchmark using integer name."""
        merger = DataFrameMerger([self.df1, self.df2, self.df3])
        result = merger.merge(benchmark_config=('returns', 0))
        
        # Check that active column exists
        active_col = DataFrameMerger.get_active_column_name(0)
        self.assertIn(active_col, result[0].columns)
        self.assertIn(active_col, result[1].columns)
        self.assertIn(active_col, result[2].columns)
        
        # Check that benchmark's active column is zero
        self.assertTrue((result[0][active_col] == 0).all())
    
    def test_merge_with_benchmark_string_name(self):
        """Test merge with benchmark using string name."""
        merger = DataFrameMerger([
            (self.df1, 'SPX'), 
            (self.df2, 'Portfolio'), 
            (self.df3, 'Fund')
        ])
        result = merger.merge(benchmark_config=('returns', 'SPX'))
        
        # Check that active column exists with correct name
        active_col = 'active_SPX'
        self.assertIn(active_col, result['SPX'].columns)
        self.assertIn(active_col, result['Portfolio'].columns)
        self.assertIn(active_col, result['Fund'].columns)
        
        # Check that active returns are correctly calculated
        # Portfolio active return = Portfolio return - SPX return
        common_idx = merger.common_index
        expected_active = (
            self.df2.loc[common_idx, 'returns'] - 
            self.df1.loc[common_idx, 'returns']
        )
        pd.testing.assert_series_equal(
            result['Portfolio'][active_col], 
            expected_active,
            check_names=False
        )
    
    def test_benchmark_column_not_in_all_dfs_raises_error(self):
        """Test that invalid benchmark column raises ValueError."""
        merger = DataFrameMerger([self.df1, self.df2])
        with self.assertRaises(ValueError) as context:
            merger.merge(benchmark_config=('nonexistent', 0))
        self.assertIn("not found in all DataFrames", str(context.exception))
    
    def test_benchmark_name_not_found_raises_error(self):
        """Test that invalid benchmark name raises ValueError."""
        merger = DataFrameMerger([
            (self.df1, 'A'), 
            (self.df2, 'B')
        ])
        with self.assertRaises(ValueError) as context:
            merger.merge(benchmark_config=('returns', 'Z'))
        self.assertIn("not found", str(context.exception))
    
    def test_get_dataframe(self):
        """Test retrieving DataFrames by name."""
        merger = DataFrameMerger([
            (self.df1, 'A'), 
            (self.df2, 'B')
        ])
        retrieved = merger.get_dataframe('A')
        pd.testing.assert_frame_equal(retrieved, self.df1)
    
    def test_get_dataframe_not_found_raises_error(self):
        """Test that retrieving non-existent DataFrame raises KeyError."""
        merger = DataFrameMerger([self.df1, self.df2])
        with self.assertRaises(KeyError) as context:
            merger.get_dataframe('Z')
        self.assertIn("not found", str(context.exception))
    
    def test_caching_behavior(self):
        """Test that common_columns and common_index are cached."""
        merger = DataFrameMerger([self.df1, self.df2])
        
        # Access properties
        cols1 = merger.common_columns
        cols2 = merger.common_columns
        self.assertIs(cols1, cols2)  # Same object (cached)
        
        idx1 = merger.common_index
        idx2 = merger.common_index
        self.assertIs(idx1, idx2)  # Same object (cached)
    
    def test_active_column_name_generation(self):
        """Test active column naming convention."""
        self.assertEqual(
            DataFrameMerger.get_active_column_name('SPX'), 
            'active_SPX'
        )
        self.assertEqual(
            DataFrameMerger.get_active_column_name(0), 
            'active_0'
        )


# Example usage demonstrating all features
if __name__ == "__main__":
    # Run unit tests
    print("Running unit tests...\n")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    print("\n" + "="*80)
    print("EXAMPLE USAGE")
    print("="*80 + "\n")
    
    # Create sample DataFrames
    dates1 = pd.date_range('2024-01-01', periods=5, freq='D')
    dates2 = pd.date_range('2024-01-02', periods=5, freq='D')
    dates3 = pd.date_range('2024-01-01', periods=6, freq='D')
    
    df1 = pd.DataFrame({
        'price': [100, 101, 102, 103, 104],
        'volume': [1000, 1100, 1200, 1300, 1400],
        'returns': [0.00, 0.01, 0.0099, 0.0098, 0.0097]
    }, index=dates1)
    
    df2 = pd.DataFrame({
        'price': [50, 51, 52, 53, 54],
        'volume': [2000, 2100, 2200, 2300, 2400],
        'returns': [0.02, 0.0196, 0.0192, 0.0189, 0.0185]
    }, index=dates2)
    
    df3 = pd.DataFrame({
        'price': [200, 201, 202, 203, 204, 205],
        'volume': [500, 550, 600, 650, 700, 750],
        'returns': [0.005, 0.00497, 0.00493, 0.00492, 0.00490, 0.00487]
    }, index=dates3)
    
    print("Example 1: Merge with benchmark (SPX) - Active returns")
    print("="*80)
    merger = DataFrameMerger([
        (df1, 'SPX'),       # S&P 500 index as benchmark
        (df2, 'Portfolio'), # Portfolio returns
        (df3, 'Fund')       # Mutual fund returns
    ])
    result = merger.merge(benchmark_config=('returns', 'SPX'))
    print(result)
    print("\nNote: active_SPX shows returns relative to the S&P 500 benchmark")
