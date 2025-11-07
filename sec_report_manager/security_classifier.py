"""
Security Classification and Portfolio Analysis System

A comprehensive framework for:
1. Classifying securities across multiple dimensions (type, asset class, geography, exposure)
2. Analyzing portfolio compositions
3. Comparing portfolios with flexible visualization backends

Key Features:
- Rule-based classification with extensible pattern matching
- Multi-dimensional aggregation (asset class, geography, exposure, leverage)
- Polymorphic visualization supporting Plotly, Seaborn, and Matplotlib
- Clean separation of concerns with clear interfaces

Example Usage:
    # Classify a single security
    classifier = SecurityClassifier()
    result = classifier.classify("AAON INC", "COM PAR $0.004")
    # -> SecurityClassification(type="physical", asset_class="stock", geography="us")
    
    # Compare two portfolios
    analyzer = PortfolioAnalyzer()
    results = analyzer.compare_portfolios(portfolio1, portfolio2)
    
    # Visualize with any backend
    viz = VisualizationFactory.create('plotly')
    viz.plot_comparison(results['comparisons'], portfolio_names=('P1', 'P2'))
"""

import re
from typing import Literal, Optional, Dict, List, Tuple, Protocol, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np


# ==================== TYPE DEFINITIONS ====================

SecurityType = Literal["leveraged_etf", "etf", "derivative", "physical"]
AssetClass = Literal["stock", "debt", "index", "commodity"]
Geography = Literal["us", "foreign", "emerging", "china", "europe", "asia", "latam", "global"]
Exposure = Literal["long", "short"]


# ==================== DATA CLASSES ====================

@dataclass
class SecurityClassification:
    """
    Multi-dimensional classification of a security.
    
    Attributes:
        type: Product structure (leveraged_etf, etf, derivative, physical)
        asset_class: Underlying asset type (stock, debt, index, commodity)
        geography: Geographic focus/domicile (us, china, europe, etc.)
        exposure: Directional bias (long, short, or None for non-leveraged)
        leverage: Leverage multiplier (2.0 for 2X, None for unleveraged)
    
    Examples:
        Physical US stock:
            SecurityClassification(type="physical", asset_class="stock", geography="us")
        
        Leveraged China index ETF:
            SecurityClassification(type="leveraged_etf", asset_class="index", 
                                 geography="china", exposure="long", leverage=3.0)
    """
    type: SecurityType
    asset_class: Optional[AssetClass] = None
    geography: Optional[Geography] = None
    exposure: Optional[Exposure] = None
    leverage: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        return {k: v for k, v in asdict(self).items() if v is not None}


# ==================== PATTERN MATCHING ====================

class PatternMatcher:
    """
    Centralized repository of patterns and keywords for security classification.
    
    This class consolidates all regex patterns and keyword sets used across
    the classification process, making it easy to extend or modify detection rules.
    
    To add a new category:
    1. Add keywords to appropriate dictionary (KEYWORDS, GEO_KEYWORDS, etc.)
    2. No code changes needed in classifier - it automatically picks up new keywords
    """
    
    # ===== Leverage and Exposure Patterns =====
    LEVERAGE_PATTERN = r'(-?\d+\.?\d*)\s*X'  # Matches: 2X, -1X, 1.25X
    LONG_TERMS = {'LONG', 'BULL', 'LNG', 'BL'}
    SHORT_TERMS = {'SHORT', 'BEAR', 'INVERSE', 'SHT', 'SHR', 'INV', 'BR'}
    
    # ===== Entity Type Patterns =====
    CORPORATE_SUFFIXES = {'INC', 'CORP', 'LTD', 'LP', 'LLC', 'MLP', 'CO', 'SA', 'AG', 'PLC'}
    ETF_KEYWORDS = {'ETF', 'TRUST', 'FUND', 'FDS'}
    DAILY_PATTERNS = [r'\bDLY\b', r'\bDAILY\b', r'\bDEF\b']
    PHYSICAL_PATTERNS = [r'\bCOM\b', r'\bCL\s*[A-Z]\b', r'\bNOTE\b', r'\bDEBT\b', 
                        r'\bPAR\s*\$', r'\d+\.?\d*%', r'\bBOND\b', r'\bREG\b']
    
    # ===== Asset Class Keywords (Priority: debt > stock > commodity > index) =====
    KEYWORDS = {
        'debt': {'NOTE', 'BOND', 'DEBT', 'MORTGAGE', 'DEBENTURE', 'MTNF', 'CONVERT', 'SENIOR'},
        'stock': {'COM', 'COMMON', 'PREFERRED', 'EQUITY', 'SHARES', 'SHS', 'CL A', 'CL B', 'CL C'},
        'commodity': {'GOLD', 'SILVER', 'OIL', 'COPPER', 'GAS', 'BITCOIN', 'ETHER', 'SOLANA'},
        'index': {'S&P', 'RUSSELL', 'NASDAQ', 'MSCI', 'FTSE', 'CSI', 'NIKKEI',
                 'MIDCAP', 'SMCAP', 'SMALLCAP', 'LARGECAP',
                 'TECH', 'FINANCIAL', 'ENERGY', 'HEALTHCARE', 'UTILITIES', 'INDUSTRIAL',
                 'CONSUMER', 'MATERIALS', 'AEROSPACE', 'TRANSPORTATION', 'RETAIL'}
    }
    
    # ===== Geographic Keywords (Priority: specific regions > general > default) =====
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
    
    # ===== Known Ticker Symbols =====
    TICKERS = {'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'COIN',
               'MSTR', 'BABA', 'INTC', 'UBER', 'PLTR', 'CRWD', 'SMCI', 'MARA', 'RDDT'}
    
    @classmethod
    def extract_leverage(cls, text: str) -> Optional[float]:
        """Extract numeric leverage multiplier from text (e.g., '2X' -> 2.0)."""
        match = re.search(cls.LEVERAGE_PATTERN, text.upper())
        return float(match.group(1)) if match else None
    
    @classmethod
    def detect_exposure(cls, text: str) -> Optional[Exposure]:
        """Detect directional exposure: short terms override long terms."""
        text_upper = text.upper()
        # Check short first (more specific than long)
        if any(term in text_upper for term in cls.SHORT_TERMS):
            return "short"
        if any(term in text_upper for term in cls.LONG_TERMS):
            return "long"
        return None
    
    @classmethod
    def match_keywords(cls, text: str, keywords: set) -> bool:
        """Check if any keyword from set appears in text."""
        text_upper = text.upper()
        return any(kw in text_upper for kw in keywords)
    
    @classmethod
    def detect_from_dict(cls, text: str, keyword_dict: Dict[str, set]) -> Optional[str]:
        """Generic category detection from keyword dictionary. Returns first match."""
        text_upper = text.upper()
        for category, keywords in keyword_dict.items():
            if any(kw in text_upper for kw in keywords):
                return category
        return None


# ==================== CLASSIFIER ====================

class SecurityClassifier:
    """
    Rule-based security classifier using priority-ordered feature detection.
    
    Simplified from complex rule hierarchy to streamlined feature detection.
    Classification logic is concentrated in _determine_type() for easy modification.
    
    Classification Priority:
        1. Corporate issuers (INC, CORP, LTD) -> physical (highest confidence)
        2. Leveraged products (has leverage + exposure) -> leveraged_etf/derivative
        3. Physical patterns (COM, NOTE, BOND) -> physical
        4. ETF issuers without leverage -> etf
        5. Default -> physical
    
    Usage:
        classifier = SecurityClassifier()
        result = classifier.classify("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X")
        # Returns: SecurityClassification with all dimensions populated
    """
    
    def __init__(self):
        self.pm = PatternMatcher
    
    def classify(self, issuer: str, title: str) -> SecurityClassification:
        """
        Classify security across all dimensions.
        
        Args:
            issuer: Issuer name (e.g., "AAON INC", "DIREXION SHS ETF TR")
            title: Security title (e.g., "COM", "DLY AAPL BEAR 1X")
        
        Returns:
            Complete SecurityClassification with all applicable dimensions
        """
        # Extract all features in parallel
        leverage = self.pm.extract_leverage(title)
        exposure = self.pm.detect_exposure(title)
        
        # Determine classification dimensions
        return SecurityClassification(
            type=self._determine_type(issuer, title, leverage, exposure),
            asset_class=self._detect_asset_class(issuer, title),
            geography=self._detect_geography(issuer, title),
            exposure=exposure,
            leverage=abs(leverage) if leverage else None
        )
    
    def _determine_type(self, issuer: str, title: str, 
                       leverage: Optional[float], exposure: Optional[Exposure]) -> SecurityType:
        """
        Determine security type using priority rules.
        
        Returns the first matching type in priority order.
        """
        issuer_upper, title_upper = issuer.upper(), title.upper()
        
        # Check key features once
        is_corporate = any(s in issuer_upper.replace('.', '').split()[-2:] 
                          for s in self.pm.CORPORATE_SUFFIXES)
        is_etf = any(kw in issuer_upper for kw in self.pm.ETF_KEYWORDS)
        has_leverage_exposure = leverage is not None and exposure is not None
        has_physical = any(re.search(p, title_upper) for p in self.pm.PHYSICAL_PATTERNS)
        
        # Priority 1: Corporate issuers
        if is_corporate:
            return "physical"
        
        # Priority 2: Leveraged products
        if has_leverage_exposure:
            return "leveraged_etf" if is_etf else "derivative"
        
        # Priority 3: Physical security indicators
        if has_physical:
            return "physical"
        
        # Priority 4: Plain ETFs
        if is_etf:
            return "etf"
        
        return "physical"
    
    def _detect_asset_class(self, issuer: str, title: str) -> Optional[AssetClass]:
        """
        Detect asset class with priority ordering.
        
        Priority: debt > stock > commodity > index
        (Most specific to most general)
        """
        combined = f"{issuer} {title}"
        
        # Special case: interest rate is strong debt signal
        if re.search(r'\d+\.?\d*%', title):
            return "debt"
        
        # Check categories in priority order
        for asset_class in ['debt', 'stock', 'commodity', 'index']:
            if self.pm.match_keywords(combined, self.pm.KEYWORDS[asset_class]):
                return asset_class
        
        return None
    
    def _detect_geography(self, issuer: str, title: str) -> Geography:
        """
        Detect geography with regional priority.
        
        Priority: specific regions > general categories > default US
        """
        combined = f"{issuer} {title}"
        
        # Try specific regions first
        for region in ['china', 'latam', 'europe', 'asia']:
            if self.pm.match_keywords(combined, self.pm.GEO_KEYWORDS[region]):
                return region
        
        # Then general categories
        for category in ['emerging', 'global']:
            if self.pm.match_keywords(combined, self.pm.GEO_KEYWORDS[category]):
                return category
        
        # Check offshore domicile in issuer only
        if self.pm.match_keywords(issuer, self.pm.GEO_KEYWORDS['foreign']):
            return "foreign"
        
        # Default to US
        return "us"


# ==================== PORTFOLIO ANALYSIS ====================

class PortfolioAnalyzer:
    """
    Analyzes and compares portfolios based on security classifications.
    
    Core responsibilities:
    1. Classify all holdings in a portfolio
    2. Aggregate weights by classification categories
    3. Compare two portfolios across categories
    
    Example:
        analyzer = PortfolioAnalyzer()
        
        # Analyze single portfolio
        portfolio_df = analyzer.analyze_portfolio([
            ("AAON INC", "COM", 10000),
            ("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X", 5000)
        ])
        
        # Compare two portfolios
        results = analyzer.compare_portfolios(portfolio1, portfolio2)
    """
    
    def __init__(self):
        self.classifier = SecurityClassifier()
    
    def analyze_portfolio(self, holdings) -> pd.DataFrame:
        """
        Analyze portfolio holdings with classification and weights.
        
        Args:
            holdings: Either:
                - List of (issuer, title, value) tuples
                - DataFrame with columns: 'nameofissuer', 'titleofclass', 'value'
                - DataFrame with columns: 'issuer', 'title', 'value'
        
        Returns:
            DataFrame with columns: issuer, title, value, weight, type, asset_class, 
            geography, exposure, leverage
        
        Examples:
            # From list of tuples
            holdings = [("AAON INC", "COM", 10000), ...]
            df = analyzer.analyze_portfolio(holdings)
            
            # From DataFrame
            df = pd.DataFrame({
                'nameofissuer': ['AAON INC', ...],
                'titleofclass': ['COM', ...],
                'value': [10000, ...]
            })
            result = analyzer.analyze_portfolio(df)
        """
        # Convert DataFrame to list of tuples if needed
        if isinstance(holdings, pd.DataFrame):
            # Handle different column naming conventions
            if 'nameofissuer' in holdings.columns:
                issuer_col = 'nameofissuer'
                title_col = 'titleofclass'
            elif 'issuer' in holdings.columns:
                issuer_col = 'issuer'
                title_col = 'title'
            else:
                raise ValueError("DataFrame must have either ('nameofissuer', 'titleofclass') or ('issuer', 'title') columns")
            
            if 'value' not in holdings.columns:
                raise ValueError("DataFrame must have 'value' column")
            
            # Convert to list of tuples
            holdings = list(holdings[[issuer_col, title_col, 'value']].itertuples(index=False, name=None))
        
        # Now handle as list
        if not holdings or len(holdings) == 0:
            return pd.DataFrame()
        
        # Classify all holdings
        total_value = sum(value for _, _, value in holdings)
        
        rows = []
        for issuer, title, value in holdings:
            classification = self.classifier.classify(issuer, title)
            row = {
                'issuer': issuer,
                'title': title,
                'value': value,
                'weight': (value / total_value * 100) if total_value > 0 else 0,
                **classification.to_dict()
            }
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def aggregate_weights(self, portfolio_df: pd.DataFrame, category: str) -> pd.DataFrame:
        """
        Aggregate portfolio weights by classification category.
        
        Args:
            portfolio_df: Portfolio DataFrame from analyze_portfolio()
            category: Category to aggregate by (e.g., 'asset_class', 'geography')
        
        Returns:
            DataFrame with category values and aggregated weights
        """
        if category not in portfolio_df.columns or portfolio_df.empty:
            return pd.DataFrame(columns=[category, 'weight'])
        
        # Handle None values as 'unclassified'
        df = portfolio_df.copy()
        df[category] = df[category].fillna('unclassified')
        
        return df.groupby(category, as_index=False)['weight'].sum()
    
    def compare_portfolios(self, 
                          holdings1,
                          holdings2,
                          portfolio_names: Tuple[str, str] = ('Portfolio 1', 'Portfolio 2'),
                          categories: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive comparison of two portfolios across all categories.
        
        Args:
            holdings1: First portfolio as either:
                - List of (issuer, title, value) tuples
                - DataFrame with columns: 'nameofissuer'/'issuer', 'titleofclass'/'title', 'value'
            holdings2: Second portfolio (same format options as holdings1)
            portfolio_names: Names for the portfolios
            categories: Categories to compare (default: all)
        
        Returns:
            Dictionary containing:
                - 'portfolio1': Analyzed portfolio 1 DataFrame
                - 'portfolio2': Analyzed portfolio 2 DataFrame
                - 'comparisons': Dict of comparison DataFrames by category
                - 'summary': Overall statistics
        
        Examples:
            # From lists
            p1 = [("AAON INC", "COM", 10000), ...]
            p2 = [("AAR CORP", "COM", 15000), ...]
            results = analyzer.compare_portfolios(p1, p2)
            
            # From DataFrames
            df1 = pd.DataFrame({'nameofissuer': [...], 'titleofclass': [...], 'value': [...]})
            df2 = pd.DataFrame({'nameofissuer': [...], 'titleofclass': [...], 'value': [...]})
            results = analyzer.compare_portfolios(df1, df2)
        """
        # Analyze both portfolios
        p1_df = self.analyze_portfolio(holdings1)
        p2_df = self.analyze_portfolio(holdings2)
        
        # Default to all available categories
        if categories is None:
            categories = ['type', 'asset_class', 'geography', 'exposure']
        
        # Compare each category
        comparisons = {}
        for category in categories:
            comp_df = self._compare_category(p1_df, p2_df, category, portfolio_names)
            if not comp_df.empty:
                comparisons[category] = comp_df
        
        # Generate summary statistics
        summary = self._generate_summary(comparisons, portfolio_names)
        
        return {
            'portfolio1': p1_df,
            'portfolio2': p2_df,
            'comparisons': comparisons,
            'summary': summary,
            'portfolio_names': portfolio_names
        }
    
    def _compare_category(self, 
                         p1_df: pd.DataFrame, 
                         p2_df: pd.DataFrame,
                         category: str,
                         portfolio_names: Tuple[str, str]) -> pd.DataFrame:
        """Compare two portfolios for a specific category."""
        # Aggregate both portfolios
        agg1 = self.aggregate_weights(p1_df, category)
        agg2 = self.aggregate_weights(p2_df, category)
        
        if agg1.empty or agg2.empty:
            return pd.DataFrame()
        
        # Merge and calculate differences
        comp = agg1.merge(agg2, on=category, how='outer', suffixes=('_1', '_2')).fillna(0)
        comp.columns = [category, portfolio_names[0], portfolio_names[1]]
        comp['difference'] = comp[portfolio_names[1]] - comp[portfolio_names[0]]
        comp['abs_difference'] = comp['difference'].abs()
        
        return comp.sort_values('abs_difference', ascending=False)
    
    def _generate_summary(self, comparisons: Dict[str, pd.DataFrame], 
                         portfolio_names: Tuple[str, str]) -> Dict[str, Any]:
        """Generate summary statistics from comparison results."""
        summary = {
            'total_categories': len(comparisons),
            'max_difference': 0,
            'max_difference_category': None,
            'mean_absolute_difference': 0
        }
        
        all_diffs = []
        for category, df in comparisons.items():
            if df.empty:
                continue
            
            max_diff = df['abs_difference'].max()
            all_diffs.extend(df['abs_difference'].tolist())
            
            if max_diff > summary['max_difference']:
                summary['max_difference'] = max_diff
                summary['max_difference_category'] = f"{category}: {df.loc[df['abs_difference'].idxmax(), category]}"
        
        if all_diffs:
            summary['mean_absolute_difference'] = np.mean(all_diffs)
        
        return summary


# ==================== VISUALIZATION INTERFACE ====================

class VisualizationBackend(ABC):
    """
    Abstract base class defining the visualization interface.
    
    All visualization backends (Plotly, Seaborn, Matplotlib) must implement
    these methods with consistent signatures and behavior.
    
    Design Philosophy:
    - Unified interface across backends
    - Returns figure objects that can be saved or displayed
    - Consistent parameter naming and behavior
    """
    
    @abstractmethod
    def plot_comparison_dashboard(self, 
                                  comparison_data: Dict[str, pd.DataFrame],
                                  portfolio_names: Tuple[str, str]) -> Any:
        """
        Create multi-panel comparison dashboard.
        
        Args:
            comparison_data: Dict mapping category -> comparison DataFrame
            portfolio_names: Names of the two portfolios
        
        Returns:
            Backend-specific figure object
        """
        pass
    
    @abstractmethod
    def plot_difference_chart(self,
                             comparison_data: Dict[str, pd.DataFrame],
                             portfolio_names: Tuple[str, str],
                             top_n: int = 10) -> Any:
        """
        Create chart showing top differences between portfolios.
        
        Args:
            comparison_data: Dict mapping category -> comparison DataFrame
            portfolio_names: Names of the two portfolios
            top_n: Number of top differences to show
        
        Returns:
            Backend-specific figure object
        """
        pass
    
    @abstractmethod
    def plot_category_breakdown(self,
                                category: str,
                                comparison_df: pd.DataFrame,
                                portfolio_names: Tuple[str, str]) -> Any:
        """
        Create detailed breakdown for a single category.
        
        Args:
            category: Category name (e.g., 'asset_class')
            comparison_df: Comparison DataFrame for this category
            portfolio_names: Names of the two portfolios
        
        Returns:
            Backend-specific figure object
        """
        pass


class PlotlyBackend(VisualizationBackend):
    """
    Plotly implementation of visualization interface.
    
    Produces interactive HTML visualizations with hover tooltips,
    zoom capabilities, and modern aesthetics.
    """
    
    def __init__(self):
        """Import plotly on instantiation to avoid unnecessary dependencies."""
        try:
            import plotly.graph_objects as go
            import plotly.express as px
            from plotly.subplots import make_subplots
            self.go = go
            self.px = px
            self.make_subplots = make_subplots
        except ImportError:
            raise ImportError("Plotly not installed. Run: pip install plotly")
    
    def plot_comparison_dashboard(self, 
                                  comparison_data: Dict[str, pd.DataFrame],
                                  portfolio_names: Tuple[str, str]) -> Any:
        """Create 2x2 dashboard with grouped bar charts for each category."""
        # Determine layout based on available categories
        categories = [c for c in ['type', 'asset_class', 'geography', 'exposure'] 
                     if c in comparison_data and not comparison_data[c].empty]
        
        if not categories:
            return None
        
        n_cats = len(categories)
        rows = (n_cats + 1) // 2
        cols = 2 if n_cats > 1 else 1
        
        fig = self.make_subplots(
            rows=rows, cols=cols,
            subplot_titles=[c.replace('_', ' ').title() for c in categories],
            specs=[[{'type': 'bar'} for _ in range(cols)] for _ in range(rows)]
        )
        
        # Add bars for each category
        for idx, category in enumerate(categories):
            row = (idx // cols) + 1
            col = (idx % cols) + 1
            df = comparison_data[category]
            
            # Portfolio 1 bars
            fig.add_trace(
                self.go.Bar(
                    name=portfolio_names[0],
                    x=df[category],
                    y=df[portfolio_names[0]],
                    marker_color='#6495ED',  # Cornflower blue
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
            
            # Portfolio 2 bars
            fig.add_trace(
                self.go.Bar(
                    name=portfolio_names[1],
                    x=df[category],
                    y=df[portfolio_names[1]],
                    marker_color='#FF6B6B',  # Coral red
                    showlegend=(idx == 0)
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            height=400 * rows,
            title_text="Portfolio Comparison Dashboard",
            barmode='group',
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(tickangle=45)
        fig.update_yaxes(title_text="Weight (%)")
        
        return fig
    
    def plot_difference_chart(self,
                             comparison_data: Dict[str, pd.DataFrame],
                             portfolio_names: Tuple[str, str],
                             top_n: int = 10) -> Any:
        """Create horizontal bar chart of top differences."""
        # Collect all differences across categories
        all_diffs = []
        for category, df in comparison_data.items():
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_diffs.append({
                    'label': f"{category}: {row[category]}",
                    'difference': row['difference'],
                    'abs_diff': row['abs_difference'],
                    portfolio_names[0]: row[portfolio_names[0]],
                    portfolio_names[1]: row[portfolio_names[1]]
                })
        
        if not all_diffs:
            return None
        
        # Get top N by absolute difference
        df_all = pd.DataFrame(all_diffs).nlargest(top_n, 'abs_diff')
        
        # Color code: green if P2 > P1, red if P1 > P2
        colors = ['#28a745' if x > 0 else '#dc3545' for x in df_all['difference']]
        
        fig = self.go.Figure()
        fig.add_trace(self.go.Bar(
            y=df_all['label'],
            x=df_all['difference'],
            orientation='h',
            marker_color=colors,
            text=[f"{x:+.1f}%" for x in df_all['difference']],
            textposition='outside',
            hovertemplate=(
                '<b>%{y}</b><br>' +
                f'{portfolio_names[0]}: %{{customdata[0]:.1f}}%<br>' +
                f'{portfolio_names[1]}: %{{customdata[1]:.1f}}%<br>' +
                'Difference: %{x:+.1f}%<extra></extra>'
            ),
            customdata=df_all[[portfolio_names[0], portfolio_names[1]]].values
        ))
        
        fig.update_layout(
            title=f"Top {top_n} Differences: {portfolio_names[1]} - {portfolio_names[0]}",
            xaxis_title="Difference in Weight (%)",
            yaxis_title="",
            height=max(400, top_n * 40),
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def plot_category_breakdown(self,
                                category: str,
                                comparison_df: pd.DataFrame,
                                portfolio_names: Tuple[str, str]) -> Any:
        """Create side-by-side bar chart for single category."""
        fig = self.go.Figure()
        
        fig.add_trace(self.go.Bar(
            name=portfolio_names[0],
            x=comparison_df[category],
            y=comparison_df[portfolio_names[0]],
            marker_color='#6495ED'
        ))
        
        fig.add_trace(self.go.Bar(
            name=portfolio_names[1],
            x=comparison_df[category],
            y=comparison_df[portfolio_names[1]],
            marker_color='#FF6B6B'
        ))
        
        fig.update_layout(
            title=f"{category.replace('_', ' ').title()} Breakdown",
            xaxis_title=category.replace('_', ' ').title(),
            yaxis_title="Weight (%)",
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig


class SeabornBackend(VisualizationBackend):
    """
    Seaborn implementation of visualization interface.
    
    Produces publication-quality static visualizations with
    excellent statistical graphics and color palettes.
    """
    
    def __init__(self):
        """Import seaborn and matplotlib on instantiation."""
        try:
            import seaborn as sns
            import matplotlib.pyplot as plt
            self.sns = sns
            self.plt = plt
            # Set style
            sns.set_style("whitegrid")
            sns.set_palette("husl")
        except ImportError:
            raise ImportError("Seaborn/Matplotlib not installed. Run: pip install seaborn matplotlib")
    
    def plot_comparison_dashboard(self, 
                                  comparison_data: Dict[str, pd.DataFrame],
                                  portfolio_names: Tuple[str, str]) -> Any:
        """Create 2x2 dashboard with grouped bar plots."""
        categories = [c for c in ['type', 'asset_class', 'geography', 'exposure'] 
                     if c in comparison_data and not comparison_data[c].empty]
        
        if not categories:
            return None
        
        n_cats = len(categories)
        rows = (n_cats + 1) // 2
        cols = 2 if n_cats > 1 else 1
        
        fig, axes = self.plt.subplots(rows, cols, figsize=(12, 5*rows))
        if n_cats == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, category in enumerate(categories):
            df = comparison_data[category]
            
            # Reshape for seaborn
            df_plot = df.melt(
                id_vars=[category],
                value_vars=portfolio_names,
                var_name='Portfolio',
                value_name='Weight'
            )
            
            self.sns.barplot(
                data=df_plot,
                x=category,
                y='Weight',
                hue='Portfolio',
                ax=axes[idx]
            )
            
            axes[idx].set_title(category.replace('_', ' ').title())
            axes[idx].set_ylabel('Weight (%)')
            axes[idx].set_xlabel('')
            axes[idx].tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for idx in range(n_cats, len(axes)):
            axes[idx].set_visible(False)
        
        self.plt.tight_layout()
        return fig
    
    def plot_difference_chart(self,
                             comparison_data: Dict[str, pd.DataFrame],
                             portfolio_names: Tuple[str, str],
                             top_n: int = 10) -> Any:
        """Create horizontal bar chart of top differences."""
        # Collect all differences
        all_diffs = []
        for category, df in comparison_data.items():
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_diffs.append({
                    'label': f"{category}: {row[category]}",
                    'difference': row['difference'],
                    'abs_diff': row['abs_difference']
                })
        
        if not all_diffs:
            return None
        
        df_all = pd.DataFrame(all_diffs).nlargest(top_n, 'abs_diff')
        
        # Create figure
        fig, ax = self.plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        # Color code bars
        colors = ['#28a745' if x > 0 else '#dc3545' for x in df_all['difference']]
        
        ax.barh(df_all['label'], df_all['difference'], color=colors)
        ax.set_xlabel('Difference in Weight (%)')
        ax.set_title(f'Top {top_n} Differences: {portfolio_names[1]} - {portfolio_names[0]}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        
        self.plt.tight_layout()
        return fig
    
    def plot_category_breakdown(self,
                                category: str,
                                comparison_df: pd.DataFrame,
                                portfolio_names: Tuple[str, str]) -> Any:
        """Create grouped bar chart for single category."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Reshape for seaborn
        df_plot = comparison_df.melt(
            id_vars=[category],
            value_vars=portfolio_names,
            var_name='Portfolio',
            value_name='Weight'
        )
        
        self.sns.barplot(
            data=df_plot,
            x=category,
            y='Weight',
            hue='Portfolio',
            ax=ax
        )
        
        ax.set_title(f"{category.replace('_', ' ').title()} Breakdown")
        ax.set_ylabel('Weight (%)')
        ax.set_xlabel('')
        ax.tick_params(axis='x', rotation=45)
        
        self.plt.tight_layout()
        return fig


class MatplotlibBackend(VisualizationBackend):
    """
    Pure Matplotlib implementation of visualization interface.
    
    Produces highly customizable static visualizations with
    fine-grained control over all visual elements.
    """
    
    def __init__(self):
        """Import matplotlib on instantiation."""
        try:
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
            self.plt = plt
            self.Rectangle = Rectangle
        except ImportError:
            raise ImportError("Matplotlib not installed. Run: pip install matplotlib")
    
    def plot_comparison_dashboard(self, 
                                  comparison_data: Dict[str, pd.DataFrame],
                                  portfolio_names: Tuple[str, str]) -> Any:
        """Create 2x2 dashboard with grouped bar charts."""
        categories = [c for c in ['type', 'asset_class', 'geography', 'exposure'] 
                     if c in comparison_data and not comparison_data[c].empty]
        
        if not categories:
            return None
        
        n_cats = len(categories)
        rows = (n_cats + 1) // 2
        cols = 2 if n_cats > 1 else 1
        
        fig, axes = self.plt.subplots(rows, cols, figsize=(12, 5*rows))
        if n_cats == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        # Colors for portfolios
        colors = ['#6495ED', '#FF6B6B']
        
        for idx, category in enumerate(categories):
            df = comparison_data[category]
            ax = axes[idx]
            
            # Prepare data
            categories_list = df[category].tolist()
            p1_values = df[portfolio_names[0]].tolist()
            p2_values = df[portfolio_names[1]].tolist()
            
            # Create bars
            x = np.arange(len(categories_list))
            width = 0.35
            
            ax.bar(x - width/2, p1_values, width, label=portfolio_names[0], color=colors[0])
            ax.bar(x + width/2, p2_values, width, label=portfolio_names[1], color=colors[1])
            
            ax.set_ylabel('Weight (%)')
            ax.set_title(category.replace('_', ' ').title())
            ax.set_xticks(x)
            ax.set_xticklabels(categories_list, rotation=45, ha='right')
            
            if idx == 0:
                ax.legend()
            
            ax.grid(axis='y', alpha=0.3)
        
        # Hide unused subplots
        for idx in range(n_cats, len(axes)):
            axes[idx].set_visible(False)
        
        self.plt.tight_layout()
        return fig
    
    def plot_difference_chart(self,
                             comparison_data: Dict[str, pd.DataFrame],
                             portfolio_names: Tuple[str, str],
                             top_n: int = 10) -> Any:
        """Create horizontal bar chart of top differences."""
        # Collect all differences
        all_diffs = []
        for category, df in comparison_data.items():
            if df.empty:
                continue
            for _, row in df.iterrows():
                all_diffs.append({
                    'label': f"{category}: {row[category]}",
                    'difference': row['difference'],
                    'abs_diff': row['abs_difference']
                })
        
        if not all_diffs:
            return None
        
        df_all = pd.DataFrame(all_diffs).nlargest(top_n, 'abs_diff')
        
        # Create figure
        fig, ax = self.plt.subplots(figsize=(10, max(6, top_n * 0.4)))
        
        # Color code bars
        colors = ['#28a745' if x > 0 else '#dc3545' for x in df_all['difference']]
        
        y_pos = np.arange(len(df_all))
        ax.barh(y_pos, df_all['difference'], color=colors)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_all['label'])
        ax.set_xlabel('Difference in Weight (%)')
        ax.set_title(f'Top {top_n} Differences: {portfolio_names[1]} - {portfolio_names[0]}')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.grid(axis='x', alpha=0.3)
        
        self.plt.tight_layout()
        return fig
    
    def plot_category_breakdown(self,
                                category: str,
                                comparison_df: pd.DataFrame,
                                portfolio_names: Tuple[str, str]) -> Any:
        """Create grouped bar chart for single category."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Prepare data
        categories_list = comparison_df[category].tolist()
        p1_values = comparison_df[portfolio_names[0]].tolist()
        p2_values = comparison_df[portfolio_names[1]].tolist()
        
        # Create bars
        x = np.arange(len(categories_list))
        width = 0.35
        colors = ['#6495ED', '#FF6B6B']
        
        ax.bar(x - width/2, p1_values, width, label=portfolio_names[0], color=colors[0])
        ax.bar(x + width/2, p2_values, width, label=portfolio_names[1], color=colors[1])
        
        ax.set_ylabel('Weight (%)')
        ax.set_title(f"{category.replace('_', ' ').title()} Breakdown")
        ax.set_xticks(x)
        ax.set_xticklabels(categories_list, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        self.plt.tight_layout()
        return fig


class VisualizationFactory:
    """
    Factory for creating visualization backend instances.
    
    Provides a clean interface to instantiate the appropriate visualization
    backend without exposing implementation details.
    
    Usage:
        # Create Plotly backend
        viz = VisualizationFactory.create('plotly')
        
        # Create Seaborn backend
        viz = VisualizationFactory.create('seaborn')
        
        # Use with comparison results
        fig = viz.plot_comparison_dashboard(comparisons, ('P1', 'P2'))
        fig.show()  # For Plotly
        # or
        fig.savefig('comparison.png')  # For Matplotlib/Seaborn
    """
    
    _backends = {
        'plotly': PlotlyBackend,
        'seaborn': SeabornBackend,
        'matplotlib': MatplotlibBackend
    }
    
    @classmethod
    def create(cls, backend: str = 'plotly') -> VisualizationBackend:
        """
        Create visualization backend instance.
        
        Args:
            backend: Backend name ('plotly', 'seaborn', or 'matplotlib')
        
        Returns:
            Instantiated visualization backend
        
        Raises:
            ValueError: If backend name is invalid
            ImportError: If required packages not installed
        """
        backend_lower = backend.lower()
        if backend_lower not in cls._backends:
            raise ValueError(f"Unknown backend '{backend}'. Choose from: {list(cls._backends.keys())}")
        
        return cls._backends[backend_lower]()
    
    @classmethod
    def available_backends(cls) -> List[str]:
        """Return list of available backend names."""
        return list(cls._backends.keys())


# ==================== HIGH-LEVEL API ====================

def compare_portfolios(
    holdings1,
    holdings2,
    portfolio_names: Tuple[str, str] = ('Portfolio 1', 'Portfolio 2'),
    backend: str = 'plotly',
    show_plots: bool = True
) -> Dict[str, Any]:
    """
    Complete portfolio comparison with automatic visualization.
    
    This is the main entry point for portfolio analysis. It handles:
    1. Classification of all holdings
    2. Weight aggregation by categories
    3. Comparison calculations
    4. Automatic visualization generation
    
    Args:
        holdings1: First portfolio as either:
            - List of (issuer, title, value) tuples: [("AAON INC", "COM", 10000), ...]
            - DataFrame with columns: 'nameofissuer'/'issuer', 'titleofclass'/'title', 'value'
        holdings2: Second portfolio (same format options as holdings1)
        portfolio_names: Display names for portfolios
        backend: Visualization backend ('plotly', 'seaborn', 'matplotlib')
        show_plots: Whether to display plots immediately
    
    Returns:
        Dictionary containing:
            - 'portfolio1': DataFrame with classified holdings
            - 'portfolio2': DataFrame with classified holdings
            - 'comparisons': Dict of comparison DataFrames by category
            - 'summary': Statistical summary
            - 'figures': Dict of generated visualizations
    
    Example:
        # From lists of tuples
        portfolio_a = [
            ("AAON INC", "COM", 10000),
            ("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X", 5000),
        ]
        
        portfolio_b = [
            ("ZTO EXPRESS CAYMAN INC", "COM", 12000),
            ("GRANITESHARES ETF TR", "2X LONG NVDA DAI", 8000),
        ]
        
        results = compare_portfolios(portfolio_a, portfolio_b)
        
        # From DataFrames
        df1 = pd.DataFrame({
            'nameofissuer': ['AAON INC', 'DIREXION SHS ETF TR'],
            'titleofclass': ['COM', 'DLY AAPL BEAR 1X'],
            'value': [10000, 5000]
        })
        
        df2 = pd.DataFrame({
            'nameofissuer': ['ZTO EXPRESS CAYMAN INC', 'GRANITESHARES ETF TR'],
            'titleofclass': ['COM', '2X LONG NVDA DAI'],
            'value': [12000, 8000]
        })
        
        results = compare_portfolios(df1, df2, backend='seaborn')
        
        # Access results
        print(results['summary'])
        results['figures']['dashboard'].show()  # For Plotly
    """
    # Analyze portfolios
    analyzer = PortfolioAnalyzer()
    results = analyzer.compare_portfolios(holdings1, holdings2, portfolio_names)
    
    # Print summary
    print("\n" + "="*80)
    print(f"PORTFOLIO COMPARISON: {portfolio_names[0]} vs {portfolio_names[1]}")
    print("="*80)
    
    for category, df in results['comparisons'].items():
        print(f"\n{category.upper()} Distribution:")
        print(df[[category, portfolio_names[0], portfolio_names[1], 'difference']].to_string(index=False))
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS:")
    print("="*80)
    for key, value in results['summary'].items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    
    # Generate visualizations
    viz = VisualizationFactory.create(backend)
    
    figures = {
        'dashboard': viz.plot_comparison_dashboard(results['comparisons'], portfolio_names),
        'differences': viz.plot_difference_chart(results['comparisons'], portfolio_names, top_n=10)
    }
    
    # Add individual category breakdowns
    for category, df in results['comparisons'].items():
        figures[f'{category}_detail'] = viz.plot_category_breakdown(category, df, portfolio_names)
    
    results['figures'] = figures
    
    # Display plots if requested
    if show_plots:
        for name, fig in figures.items():
            if fig is None:
                continue
            
            # Handle different backends
            if backend == 'plotly':
                fig.show()
            else:  # matplotlib or seaborn
                fig.canvas.manager.set_window_title(name.replace('_', ' ').title())
                viz.plt.show()
    
    return results


# ==================== TESTING ====================

if __name__ == "__main__":
    print("Security Classification and Portfolio Analysis System")
    print("="*80)
    
    # Test classification
    print("\n1. TESTING SECURITY CLASSIFICATION:")
    print("-"*80)
    
    classifier = SecurityClassifier()
    test_cases = [
        ("AAON INC", "COM PAR $0.004", "Physical US stock"),
        ("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X", "Leveraged short US stock ETF"),
        ("ISHARES TR", "MSCI PERU AND GL", "Plain Latin America ETF"),
        ("ZTO EXPRESS CAYMAN INC", "NOTE 1.500% 9/0", "Foreign debt"),
        ("DIREXION SHS ETF TR", "CSI 300 BULL2X", "Leveraged long China index ETF"),
    ]
    
    for issuer, title, description in test_cases:
        result = classifier.classify(issuer, title)
        print(f"\n{description}:")
        print(f"  Input: {issuer}, {title}")
        print(f"  Classification: {result}")
    
    # Test portfolio comparison
    print("\n\n2. TESTING PORTFOLIO COMPARISON:")
    print("-"*80)
    
    # Conservative portfolio: mostly US stocks and bonds
    portfolio_conservative = [
        ("AAON INC", "COM", 15000),
        ("AAR CORP", "COM", 12000),
        ("ABBOTT LABS", "COM", 18000),
        ("AIRBNB INC", "NOTE 3/1", 25000),
        ("AMERICAN AIRLINES GROUP INC", "NOTE 6.500% 7/0", 20000),
        ("ISHARES TR", "MSCI PERU AND GL", 10000),
    ]
    
    # Aggressive portfolio: leveraged ETFs and foreign exposure
    portfolio_aggressive = [
        ("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X", 15000),
        ("DIREXION SHS ETF TR", "DLY SMCAP BULL3X", 20000),
        ("GRANITESHARES ETF TR", "2X LONG NVDA DAI", 25000),
        ("DIREXION SHS ETF TR", "CSI 300 BULL2X", 18000),
        ("ZTO EXPRESS CAYMAN INC", "COM", 12000),
        ("VOLATILITY SHS TR", "2X BITCOIN STRAT", 10000),
    ]
    
    # Compare portfolios (set show_plots=True to see visualizations)
    results = compare_portfolios(
        portfolio_conservative,
        portfolio_aggressive,
        portfolio_names=('Conservative Portfolio', 'Aggressive Portfolio'),
        backend='plotly',  # Try 'seaborn' or 'matplotlib' too
        show_plots=False   # Set to True to display plots
    )
    
    print("\n\n3. KEY INSIGHTS:")
    print("-"*80)
    print(f"• Largest difference: {results['summary']['max_difference_category']}")
    print(f"  ({results['summary']['max_difference']:.1f}% weight difference)")
    print(f"• Average absolute difference: {results['summary']['mean_absolute_difference']:.1f}%")
    print(f"• Categories compared: {results['summary']['total_categories']}")
    
    print("\n" + "="*80)
    print("DEMO COMPLETE - Set show_plots=True to see visualizations")
    print("="*80)