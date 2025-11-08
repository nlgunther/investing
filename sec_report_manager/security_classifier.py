"""
Security Classification and Portfolio Analysis System

A comprehensive framework for classifying securities and analyzing portfolios.

Key Features:
- Multi-dimensional classification (type, asset class, geography, exposure, leverage)
- Automatic ticker discovery and persistence
- DataFrame-native operations for batch processing
- Polymorphic visualization (Plotly, Seaborn, Matplotlib)
- Portfolio comparison and effective exposure calculation

Example:
    classifier = SecurityClassifier()
    result = classifier.classify("AAON INC", "COM PAR $0.004")
    
    df_classified = classifier.classify_dataframe(df)
    exposure_df = classifier.calculate_effective_exposure(df_classified)
"""

import re
import json
import os
from typing import Literal, Optional, Dict, List, Tuple, Set, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


# ==================== TYPE DEFINITIONS ====================

SecurityType = Literal["leveraged_etf", "etf", "derivative", "physical"]
AssetClass = Literal["stock", "debt", "index", "commodity"]
Geography = Literal["us", "foreign", "emerging", "china", "europe", "asia", "latam", "global"]
Exposure = Literal["long", "short"]


# ==================== CONFIGURATION ====================

class Config:
    """Central configuration for the classification system."""
    
    # File persistence
    TICKER_CACHE_FILE = 'learned_tickers.json'
    
    # Ticker discovery parameters
    MIN_TICKER_LENGTH = 2
    MAX_TICKER_LENGTH = 5
    
    # Default values
    DEFAULT_LEVERAGE = 1.0
    DEFAULT_EXPOSURE_COL = 'effective_exposure'
    
    # Column name mappings
    COLUMN_ALIASES = {
        'issuer': ['nameofissuer', 'issuer', 'issuer_name'],
        'title': ['titleofclass', 'title', 'security_title'],
        'value': ['value', 'market_value', 'amount']
    }


# ==================== DATA CLASSES ====================

@dataclass
class SecurityClassification:
    """Multi-dimensional security classification."""
    type: SecurityType
    asset_class: Optional[AssetClass] = None
    geography: Optional[Geography] = None
    exposure: Optional[Exposure] = None
    leverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def with_leverage(self, default_leverage: float) -> 'SecurityClassification':
        """Return copy with default leverage if not set."""
        if self.leverage is not None:
            return self
        return SecurityClassification(
            type=self.type,
            asset_class=self.asset_class,
            geography=self.geography,
            exposure=self.exposure,
            leverage=default_leverage
        )


# ==================== PATTERN DEFINITIONS ====================

class Patterns:
    """
    Centralized pattern and keyword repository.
    
    Design: All patterns defined as class attributes for easy discovery and modification.
    """
    
    # Regex patterns
    LEVERAGE = r'(-?\d+\.?\d*)\s*X'
    OPTION_PUT = r'\bPUT\b'
    OPTION_CALL = r'\bCALL\b'
    TICKER = r'\b[A-Z]{{{min},{max}}}\b'  # Format with min/max later
    
    # Pattern lists
    DAILY = [r'\bDLY\b', r'\bDAILY\b', r'\bDEF\b']
    PHYSICAL = [
        r'\bCOM\b', r'\bCL\s*[A-Z]\b', r'\bNOTE\b', r'\bDEBT\b',
        r'\bPAR\s*\$', r'\d+\.?\d*%', r'\bBOND\b', r'\bREG\b'
    ]
    
    # Keyword sets
    CORPORATE_SUFFIXES = {'INC', 'CORP', 'LTD', 'LP', 'LLC', 'MLP', 'CO', 'SA', 'AG', 'PLC'}
    ETF_KEYWORDS = {'ETF', 'TRUST', 'FUND', 'FDS'}
    LONG_TERMS = {'LONG', 'BULL', 'LNG', 'BL'}
    SHORT_TERMS = {'SHORT', 'BEAR', 'INVERSE', 'SHT', 'SHR', 'INV', 'BR'}
    
    # Hierarchical keyword dictionaries
    ASSET_KEYWORDS = {
        'debt': {'NOTE', 'BOND', 'DEBT', 'MORTGAGE', 'DEBENTURE', 'MTNF', 'CONVERT', 'SENIOR'},
        'stock': {'COM', 'COMMON', 'PREFERRED', 'EQUITY', 'SHARES', 'SHS', 'CL A', 'CL B', 'CL C'},
        'commodity': {'GOLD', 'SILVER', 'OIL', 'COPPER', 'GAS', 'BITCOIN', 'ETHER', 'SOLANA'},
        'index': {
            'S&P', 'RUSSELL', 'NASDAQ', 'MSCI', 'FTSE', 'CSI', 'NIKKEI',
            'MIDCAP', 'SMCAP', 'SMALLCAP', 'LARGECAP',
            'TECH', 'FINANCIAL', 'ENERGY', 'HEALTHCARE', 'UTILITIES', 'INDUSTRIAL',
            'CONSUMER', 'MATERIALS', 'AEROSPACE', 'TRANSPORTATION', 'RETAIL'
        }
    }
    
    GEO_KEYWORDS = {
        'china': {'CHINA', 'CHINESE', 'SHANGHAI', 'SHENZHEN', 'CSI', 'BABA', 'HONG KONG'},
        'latam': {'LATIN', 'LATAM', 'BRAZIL', 'BRAZILIAN', 'BRZ', 'MEXICO', 'ARGENTINA',
                  'PERU', 'COLOMBIA', 'CHILE', 'VENEZUELA'},
        'europe': {'EUROPE', 'EUROPEAN', 'UK', 'BRITAIN', 'GERMANY', 'FRANCE', 'ITALY',
                   'SPAIN', 'NETHERLANDS', 'SWITZERLAND', 'SWEDEN', 'AEGON', 'NICE LTD'},
        'asia': {'ASIA', 'ASIAN', 'PACIFIC', 'JAPAN', 'KOREA', 'SINGAPORE', 'INDIA',
                 'THAILAND', 'VIETNAM', 'INDONESIA', 'MALAYSIA', 'PHILIPPINES'},
        'emerging': {'EMERGING', 'EMG', 'FRONTIER', 'DEVELOPING'},
        'global': {'GLOBAL', 'WORLD', 'INTERNATIONAL', 'INTL', 'WORLDWIDE'},
        'foreign': {'CAYMAN', 'BERMUDA', 'LUXEMBOURG', 'IRELAND', 'JERSEY', 'BAHAMAS'}
    }
    
    # Seed tickers
    SEED_TICKERS = {
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'COIN',
        'MSTR', 'BABA', 'INTC', 'UBER', 'PLTR', 'CRWD', 'SMCI', 'MARA', 'RDDT'
    }
    
    # Ticker exclusions
    TICKER_EXCLUSIONS = {
        'COM', 'NOTE', 'BOND', 'DEBT', 'CALL', 'PUT', 'ETF', 'FUND',
        'LONG', 'SHORT', 'BULL', 'BEAR', 'INVERSE',
        'INC', 'CORP', 'LTD', 'LLC', 'LP', 'CO', 'SA', 'AG', 'TR',
        'US', 'UK', 'EU', 'ASIA', 'NEW', 'AND', 'THE', 'FOR', 'DLY', 'DAILY', 'SHS', 'CL'
    }


# ==================== PATTERN MATCHER ====================

class PatternMatcher:
    """
    Pattern matching and feature extraction engine.
    
    Design: Static methods for stateless operations, class-level cache for learned tickers.
    """
    
    _learned_tickers: Optional[Set[str]] = None
    
    # ===== Core Extraction Methods =====
    
    @staticmethod
    def extract_leverage(text: str) -> Optional[float]:
        """Extract leverage multiplier (e.g., '2X' -> 2.0)."""
        match = re.search(Patterns.LEVERAGE, text.upper())
        return float(match.group(1)) if match else None
    
    @staticmethod
    def extract_option_leverage(text: str) -> int:
        """Return -1 for PUT options, +1 for CALL/other."""
        return -1 if re.search(Patterns.OPTION_PUT, text.upper()) else 1
    
    @staticmethod
    def detect_exposure(text: str) -> Optional[Exposure]:
        """Detect directional exposure from text."""
        text_upper = text.upper()
        if any(term in text_upper for term in Patterns.SHORT_TERMS):
            return "short"
        if any(term in text_upper for term in Patterns.LONG_TERMS):
            return "long"
        return None
    
    # ===== Generic Matching =====
    
    @staticmethod
    def match_any(text: str, keywords: Set[str]) -> bool:
        """Check if any keyword appears in text."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in keywords)
    
    @staticmethod
    def match_dict(text: str, keyword_dict: Dict[str, Set[str]]) -> Optional[str]:
        """Return first matching category from keyword dictionary."""
        text_upper = text.upper()
        for category, keywords in keyword_dict.items():
            if any(kw in text_upper for kw in keywords):
                return category
        return None
    
    @staticmethod
    def match_patterns(text: str, patterns: List[str]) -> bool:
        """Check if any regex pattern matches text."""
        text_upper = text.upper()
        return any(re.search(pattern, text_upper) for pattern in patterns)
    
    # ===== Ticker Discovery =====
    
    @classmethod
    def discover_tickers(cls, text: str) -> List[str]:
        """Discover potential ticker symbols in text."""
        ticker_pattern = Patterns.TICKER.format(
            min=Config.MIN_TICKER_LENGTH,
            max=Config.MAX_TICKER_LENGTH
        )
        words = re.findall(ticker_pattern, text.upper())
        
        # Filter out exclusions and known keywords
        all_keywords = (
            Patterns.TICKER_EXCLUSIONS |
            Patterns.CORPORATE_SUFFIXES |
            Patterns.ETF_KEYWORDS |
            Patterns.LONG_TERMS |
            Patterns.SHORT_TERMS
        )
        for keyword_set in Patterns.ASSET_KEYWORDS.values():
            all_keywords |= keyword_set
        for keyword_set in Patterns.GEO_KEYWORDS.values():
            all_keywords |= keyword_set
        
        candidates = [w for w in words if w not in all_keywords]
        
        # Check for ticker-like contexts
        text_upper = text.upper()
        contexts = ['DLY', 'DAILY', 'BEAR', 'BULL', 'LONG', 'SHORT', '2X', '3X']
        
        return [
            ticker for ticker in candidates
            if any(f'{ctx} {ticker}' in text_upper or f'{ticker} {ctx}' in text_upper
                   for ctx in contexts)
        ]
    
    @classmethod
    def learn_tickers(cls, issuer: str, title: str) -> List[str]:
        """Discover and persist new tickers."""
        candidates = cls.discover_tickers(f"{issuer} {title}")
        if not candidates:
            return []
        
        learned = cls.get_learned_tickers()
        all_known = Patterns.SEED_TICKERS | learned
        new_tickers = [t for t in candidates if t not in all_known]
        
        if new_tickers:
            learned.update(new_tickers)
            cls._save_tickers(learned)
        
        return new_tickers
    
    @classmethod
    def get_learned_tickers(cls) -> Set[str]:
        """Get learned tickers (cached)."""
        if cls._learned_tickers is None:
            cls._learned_tickers = cls._load_tickers()
        return cls._learned_tickers
    
    @classmethod
    def get_all_tickers(cls) -> Set[str]:
        """Get all tickers (seed + learned)."""
        return Patterns.SEED_TICKERS | cls.get_learned_tickers()
    
    @classmethod
    def _load_tickers(cls) -> Set[str]:
        """Load tickers from file."""
        try:
            if os.path.exists(Config.TICKER_CACHE_FILE):
                with open(Config.TICKER_CACHE_FILE, 'r') as f:
                    return set(json.load(f).get('tickers', []))
        except Exception:
            pass
        return set()
    
    @classmethod
    def _save_tickers(cls, tickers: Set[str]):
        """Save tickers to file."""
        try:
            with open(Config.TICKER_CACHE_FILE, 'w') as f:
                json.dump({
                    'tickers': sorted(tickers),
                    'count': len(tickers),
                    'last_updated': pd.Timestamp.now().isoformat()
                }, f, indent=2)
        except Exception:
            pass


# ==================== CLASSIFIERS ====================

class AssetClassifier:
    """Detects asset class from text (priority: debt > stock > commodity > index)."""
    
    @staticmethod
    def classify(issuer: str, title: str) -> Optional[AssetClass]:
        combined = f"{issuer} {title}"
        
        # Interest rate is strong debt signal
        if re.search(r'\d+\.?\d*%', title):
            return "debt"
        
        # Check in priority order
        for asset_class in ['debt', 'stock', 'commodity', 'index']:
            if PatternMatcher.match_any(combined, Patterns.ASSET_KEYWORDS[asset_class]):
                return asset_class
        
        return None


class GeographyClassifier:
    """Detects geography (priority: specific regions > general > US default)."""
    
    @staticmethod
    def classify(issuer: str, title: str) -> Geography:
        combined = f"{issuer} {title}"
        
        # Try specific regions
        for region in ['china', 'latam', 'europe', 'asia', 'emerging', 'global']:
            if PatternMatcher.match_any(combined, Patterns.GEO_KEYWORDS[region]):
                return region
        
        # Check offshore domicile
        if PatternMatcher.match_any(issuer, Patterns.GEO_KEYWORDS['foreign']):
            return "foreign"
        
        return "us"


# ==================== MAIN CLASSIFIER ====================

class SecurityClassifier:
    """
    Main security classifier with DataFrame operations.
    
    Design: Combines feature detection with high-level batch operations.
    """
    
    def __init__(self, default_leverage: Optional[float] = Config.DEFAULT_LEVERAGE):
        """
        Initialize classifier.
        
        Args:
            default_leverage: Default leverage for unleveraged securities (None to skip)
        """
        self.default_leverage = default_leverage
    
    def classify(self, issuer: str, title: str) -> SecurityClassification:
        """
        Classify a single security.
        
        Args:
            issuer: Issuer name
            title: Security title
        
        Returns:
            SecurityClassification with all dimensions
        """
        # Extract features
        leverage = PatternMatcher.extract_leverage(title)
        exposure = PatternMatcher.detect_exposure(title)
        
        # Determine type
        sec_type = self._determine_type(issuer, title, leverage, exposure)
        
        # Build classification
        classification = SecurityClassification(
            type=sec_type,
            asset_class=AssetClassifier.classify(issuer, title),
            geography=GeographyClassifier.classify(issuer, title),
            exposure=exposure,
            leverage=abs(leverage) if leverage else None
        )
        
        # Apply default leverage
        if self.default_leverage is not None:
            classification = classification.with_leverage(self.default_leverage)
        
        return classification
    
    def _determine_type(self, issuer: str, title: str,
                       leverage: Optional[float], exposure: Optional[Exposure]) -> SecurityType:
        """Determine security type using priority rules."""
        issuer_upper, title_upper = issuer.upper(), title.upper()
        
        # Check features
        is_corporate = any(
            s in issuer_upper.replace('.', '').split()[-2:]
            for s in Patterns.CORPORATE_SUFFIXES
        )
        is_etf = PatternMatcher.match_any(issuer, Patterns.ETF_KEYWORDS)
        has_leverage_exposure = leverage is not None and exposure is not None
        has_physical_pattern = PatternMatcher.match_patterns(title, Patterns.PHYSICAL)
        
        # Apply priority rules
        if is_corporate:
            return "physical"
        if has_leverage_exposure:
            return "leveraged_etf" if is_etf else "derivative"
        if has_physical_pattern:
            return "physical"
        if is_etf:
            return "etf"
        return "physical"
    
    # ===== DataFrame Operations =====
    
    def classify_dataframe(self, df: pd.DataFrame,
                          issuer_col: str = None,
                          title_col: str = None,
                          merge: bool = False) -> pd.DataFrame:
        """
        Classify all securities in DataFrame.
        
        Args:
            df: DataFrame with securities
            issuer_col: Issuer column name (auto-detected if None)
            title_col: Title column name (auto-detected if None)
            merge: If True, merge with original DataFrame
        
        Returns:
            Classification DataFrame (or merged with original if merge=True)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Auto-detect columns
        issuer_col = issuer_col or self._find_column(df, 'issuer')
        title_col = title_col or self._find_column(df, 'title')
        
        # Classify each row
        classifications = [
            self.classify(row[issuer_col], row[title_col]).to_dict()
            for _, row in df.iterrows()
        ]
        
        result = pd.DataFrame(classifications)
        return pd.concat([df.reset_index(drop=True), result], axis=1) if merge else result
    
    def calculate_effective_exposure(self, df: pd.DataFrame,
                                    value_col: str = None,
                                    leverage_col: str = 'leverage',
                                    default_leverage: float = Config.DEFAULT_LEVERAGE,
                                    result_col: str = Config.DEFAULT_EXPOSURE_COL,
                                    merge: bool = False) -> pd.DataFrame:
        """
        Calculate effective exposure (value × leverage).
        
        Args:
            df: DataFrame with value and leverage columns
            value_col: Value column name (auto-detected if None)
            leverage_col: Leverage column name
            default_leverage: Default for missing leverage
            result_col: Name for result column
            merge: If True, merge with original DataFrame
        
        Returns:
            Exposure DataFrame (or merged with original if merge=True)
        """
        if df.empty:
            return pd.DataFrame()
        
        # Auto-detect value column
        value_col = value_col or self._find_column(df, 'value')
        
        # Calculate exposure
        leverage = df[leverage_col].fillna(default_leverage)
        result = pd.DataFrame({result_col: df[value_col] * leverage})
        
        return pd.concat([df.reset_index(drop=True), result], axis=1) if merge else result
    
    @staticmethod
    def _find_column(df: pd.DataFrame, col_type: str) -> str:
        """Find column by type using aliases."""
        for col in df.columns:
            if col.lower() in Config.COLUMN_ALIASES.get(col_type, []):
                return col
        raise ValueError(f"No {col_type} column found. Expected one of: {Config.COLUMN_ALIASES[col_type]}")


# ==================== PORTFOLIO ANALYSIS ====================

class PortfolioAnalyzer:
    """Portfolio analysis and comparison."""
    
    def __init__(self, classifier: SecurityClassifier = None):
        self.classifier = classifier or SecurityClassifier()
    
    def analyze(self, holdings) -> pd.DataFrame:
        """Analyze portfolio holdings (handles DataFrame or list of tuples)."""
        # Convert to DataFrame if needed
        if isinstance(holdings, pd.DataFrame):
            df = holdings.copy()
        else:
            df = pd.DataFrame(holdings, columns=['nameofissuer', 'titleofclass', 'value'])
        
        # Classify
        classified = self.classifier.classify_dataframe(df, merge=True)
        
        # Add weights
        total_value = classified['value'].sum()
        classified['weight'] = (classified['value'] / total_value * 100) if total_value > 0 else 0
        
        return classified
    
    def compare(self, holdings1, holdings2,
               names: Tuple[str, str] = ('Portfolio 1', 'Portfolio 2')) -> Dict[str, Any]:
        """Compare two portfolios across all categories."""
        p1 = self.analyze(holdings1)
        p2 = self.analyze(holdings2)
        
        categories = ['type', 'asset_class', 'geography', 'exposure']
        comparisons = {}
        
        for cat in categories:
            if cat in p1.columns and cat in p2.columns:
                comp = self._compare_category(p1, p2, cat, names)
                if not comp.empty:
                    comparisons[cat] = comp
        
        return {
            'portfolio1': p1,
            'portfolio2': p2,
            'comparisons': comparisons,
            'summary': self._summarize(comparisons),
            'portfolio_names': names
        }
    
    def _compare_category(self, p1: pd.DataFrame, p2: pd.DataFrame,
                         category: str, names: Tuple[str, str]) -> pd.DataFrame:
        """Compare single category."""
        # Aggregate weights
        agg1 = p1.groupby(category, as_index=False)['weight'].sum()
        agg2 = p2.groupby(category, as_index=False)['weight'].sum()
        
        # Merge and calculate differences
        comp = agg1.merge(agg2, on=category, how='outer', suffixes=('_1', '_2')).fillna(0)
        comp.columns = [category, names[0], names[1]]
        comp['difference'] = comp[names[1]] - comp[names[0]]
        comp['abs_difference'] = comp['difference'].abs()
        
        return comp.sort_values('abs_difference', ascending=False)
    
    @staticmethod
    def _summarize(comparisons: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Generate summary statistics."""
        all_diffs = []
        max_diff = 0
        max_cat = None
        
        for cat, df in comparisons.items():
            if df.empty:
                continue
            diffs = df['abs_difference'].tolist()
            all_diffs.extend(diffs)
            
            cat_max = df['abs_difference'].max()
            if cat_max > max_diff:
                max_diff = cat_max
                max_cat = f"{cat}: {df.loc[df['abs_difference'].idxmax(), cat]}"
        
        return {
            'total_categories': len(comparisons),
            'max_difference': max_diff,
            'max_difference_category': max_cat,
            'mean_absolute_difference': np.mean(all_diffs) if all_diffs else 0
        }


# ==================== VISUALIZATION INTERFACE ====================

class VisualizationBackend(ABC):
    """Abstract visualization interface."""
    
    @abstractmethod
    def plot_dashboard(self, comparisons: Dict[str, pd.DataFrame],
                      names: Tuple[str, str]) -> Any:
        """Create comparison dashboard."""
        pass
    
    @abstractmethod
    def plot_differences(self, comparisons: Dict[str, pd.DataFrame],
                        names: Tuple[str, str], top_n: int = 10) -> Any:
        """Create difference chart."""
        pass


class PlotlyBackend(VisualizationBackend):
    """Plotly visualization backend."""
    
    def __init__(self):
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots
            self.go = go
            self.make_subplots = make_subplots
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")
    
    def plot_dashboard(self, comparisons, names):
        """Create 2x2 dashboard."""
        cats = [c for c in ['type', 'asset_class', 'geography', 'exposure']
                if c in comparisons and not comparisons[c].empty]
        
        if not cats:
            return None
        
        rows, cols = (len(cats) + 1) // 2, 2
        fig = self.make_subplots(rows=rows, cols=cols,
                                subplot_titles=[c.replace('_', ' ').title() for c in cats])
        
        for idx, cat in enumerate(cats):
            row, col = (idx // cols) + 1, (idx % cols) + 1
            df = comparisons[cat]
            
            for i, name in enumerate(names):
                fig.add_trace(
                    self.go.Bar(name=name, x=df[cat], y=df[name],
                              marker_color=['#6495ED', '#FF6B6B'][i],
                              showlegend=(idx == 0)),
                    row=row, col=col
                )
        
        fig.update_layout(height=400*rows, title="Portfolio Comparison",
                         barmode='group', template='plotly_white')
        fig.update_yaxes(title_text="Weight (%)")
        return fig
    
    def plot_differences(self, comparisons, names, top_n=10):
        """Create difference chart."""
        all_diffs = []
        for cat, df in comparisons.items():
            for _, row in df.iterrows():
                all_diffs.append({
                    'label': f"{cat}: {row[cat]}",
                    'diff': row['difference'],
                    'abs_diff': row['abs_difference']
                })
        
        if not all_diffs:
            return None
        
        df_all = pd.DataFrame(all_diffs).nlargest(top_n, 'abs_diff')
        colors = ['#28a745' if x > 0 else '#dc3545' for x in df_all['diff']]
        
        fig = self.go.Figure()
        fig.add_trace(self.go.Bar(y=df_all['label'], x=df_all['diff'],
                                  orientation='h', marker_color=colors))
        fig.update_layout(title=f"Top {top_n} Differences", 
                         xaxis_title="Difference (%)", height=max(400, top_n*40))
        return fig


class VisualizationFactory:
    """Factory for creating visualization backends."""
    
    _backends = {'plotly': PlotlyBackend}
    
    @classmethod
    def create(cls, backend: str = 'plotly') -> VisualizationBackend:
        """Create visualization backend."""
        if backend not in cls._backends:
            raise ValueError(f"Unknown backend: {backend}")
        return cls._backends[backend]()


# ==================== HIGH-LEVEL API ====================

def compare_portfolios(holdings1, holdings2,
                      portfolio_names: Tuple[str, str] = ('Portfolio 1', 'Portfolio 2'),
                      backend: str = 'plotly',
                      show_plots: bool = True) -> Dict[str, Any]:
    """
    Compare two portfolios with automatic visualization.
    
    Args:
        holdings1, holdings2: Portfolios as DataFrame or list of tuples
        portfolio_names: Display names
        backend: Visualization backend ('plotly')
        show_plots: Whether to display plots
    
    Returns:
        Dict with portfolio data, comparisons, and figures
    """
    analyzer = PortfolioAnalyzer()
    results = analyzer.compare(holdings1, holdings2, portfolio_names)
    
    # Print summary
    print("\n" + "="*80)
    print(f"PORTFOLIO COMPARISON: {portfolio_names[0]} vs {portfolio_names[1]}")
    print("="*80)
    
    for cat, df in results['comparisons'].items():
        print(f"\n{cat.upper()}:")
        print(df[[cat, *portfolio_names, 'difference']].to_string(index=False))
    
    print(f"\nMax Difference: {results['summary']['max_difference']:.1f}% in {results['summary']['max_difference_category']}")
    
    # Create visualizations
    viz = VisualizationFactory.create(backend)
    results['figures'] = {
        'dashboard': viz.plot_dashboard(results['comparisons'], portfolio_names),
        'differences': viz.plot_differences(results['comparisons'], portfolio_names)
    }
    
    if show_plots:
        for fig in results['figures'].values():
            if fig:
                fig.show()
    
    return results


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Security Classification System - Demo\n")
    
    # Test basic classification
    classifier = SecurityClassifier()
    result = classifier.classify("DIREXION SHS ETF TR", "DLY AAPL BEAR 2X")
    print(f"Sample Classification: {result}\n")
    
    # Test DataFrame operations
    df = pd.DataFrame({
        'nameofissuer': ['AAON INC', 'DIREXION SHS ETF TR'],
        'titleofclass': ['COM', 'DLY AAPL BEAR 2X'],
        'value': [10000, 5000]
    })
    
    classified = classifier.classify_dataframe(df, merge=True)
    exposure = classifier.calculate_effective_exposure(classified, merge=True)
    
    print("Classified Holdings:")
    print(exposure[['nameofissuer', 'type', 'leverage', 'value', 'effective_exposure']])
    
    print(f"\nTotal Exposure: ${exposure['effective_exposure'].sum():,.0f}")
    print(f"Portfolio Leverage: {exposure['effective_exposure'].sum() / classified['value'].sum():.2f}X")
    
    # Test portfolio comparison
    print("\n" + "="*80)
    portfolio1 = [
        ("AAON INC", "COM", 15000),
        ("AIRBNB INC", "NOTE 3/1", 25000),
        ("ISHARES TR", "MSCI PERU AND GL", 10000)
    ]
    
    portfolio2 = [
        ("DIREXION SHS ETF TR", "DLY AAPL BEAR 2X", 15000),
        ("GRANITESHARES ETF TR", "2X LONG NVDA DAI", 25000),
        ("VOLATILITY SHS TR", "2X BITCOIN STRAT", 10000)
    ]
    
    results = compare_portfolios(
        portfolio1, portfolio2,
        portfolio_names=('Conservative', 'Aggressive'),
        show_plots=False
    )
    
    print("\n" + "="*80)
    print("Demo Complete!")