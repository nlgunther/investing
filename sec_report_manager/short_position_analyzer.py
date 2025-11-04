"""
Short Position Analyzer for SEC Filings

Identifies and quantifies short positions from holdings data.
Compact, efficient implementation with proper regex capture groups.
"""

import pandas as pd
import re
from typing import Dict, Tuple


class ShortPositionAnalyzer:
    """Analyzes holdings to identify and quantify short positions."""
    
    # Pattern format: (regex_with_capture, short_type_name)
    # First capture group must be numeric leverage or indicator
    PATTERNS = {
        'leveraged': (r'(\d+)\s*x\s*(?:sh|short|bear|inverse)', 'Leveraged Short'),
        'ultra': (r'(ultra)(?:short|inverse|bear)', 'UltraShort'),
        'short_fund': (r'(proshares|direxion).*(?:short|bear)', 'Short ETF'),
        'inverse': (r'(inverse|bear)\b', 'Inverse ETF'),
    }
    
    def identify_shorts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify and quantify short positions.
        
        Returns DataFrame with: is_short, short_type, leverage, adjusted_value
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        # Initialize
        df['is_short'] = False
        df['short_type'] = None
        df['leverage'] = 1.0
        
        # Check put options
        if 'putcall' in df.columns:
            is_put = df['putcall'].str.contains('Put', case=False, na=False)
            df.loc[is_put, ['is_short', 'short_type']] = [True, 'Put Option']
        
        # Check title patterns
        if 'titleofclass' in df.columns:
            for name, (pattern, short_type) in self.PATTERNS.items():
                matches = df['titleofclass'].str.extract(pattern, flags=re.IGNORECASE)
                has_match = matches[0].notna()
                
                # Set short flag and type
                df.loc[has_match & ~df['is_short'], ['is_short', 'short_type']] = [True, short_type]
                
                # Extract leverage
                if name == 'leveraged':
                    df.loc[has_match, 'leverage'] = pd.to_numeric(matches[0], errors='coerce').fillna(1.0)
                elif name == 'ultra':
                    df.loc[has_match, 'leverage'] = 2.0  # Ultra typically means 2x
        
        # Calculate adjusted value
        value_col = next((c for c in ['value', 'valusd'] if c in df.columns), None)
        df['adjusted_value'] = df[value_col] * df['leverage'] if value_col and df['is_short'].any() else 0
        
        # Return shorts with relevant columns
        shorts = df[df['is_short']].copy()
        if shorts.empty:
            return pd.DataFrame()
        
        # Smart column selection
        name_col = next((c for c in ['nameofissuer', 'n', 'name'] if c in shorts.columns), None)
        qty_col = next((c for c in ['shrsorprnamt_sshprnamt', 'balance'] if c in shorts.columns), None)
        
        cols = [c for c in [name_col, 'titleofclass', 'cusip', 'putcall', value_col, 
                           qty_col, 'short_type', 'leverage', 'adjusted_value'] 
                if c and c in shorts.columns]
        
        return shorts[cols]
    
    def summarize(self, shorts: pd.DataFrame) -> Dict:
        """Generate summary statistics."""
        if shorts.empty:
            return {'total': 0, 'value': 0, 'adjusted_value': 0}
        
        value_col = 'value' if 'value' in shorts.columns else 'valusd'
        
        return {
            'total': len(shorts),
            'value': shorts[value_col].sum() if value_col in shorts.columns else 0,
            'adjusted_value': shorts['adjusted_value'].sum(),
            'by_type': shorts['short_type'].value_counts().to_dict(),
            'avg_leverage': shorts['leverage'].mean(),
            'max_leverage': shorts['leverage'].max(),
        }
    
    def top_n(self, shorts: pd.DataFrame, n: int = 10) -> pd.DataFrame:
        """Return top N shorts by adjusted value."""
        return shorts.nlargest(n, 'adjusted_value') if not shorts.empty else pd.DataFrame()


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def analyze_shorts(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """One-line analysis: (shorts_df, summary_dict)."""
    analyzer = ShortPositionAnalyzer()
    shorts = analyzer.identify_shorts(df)
    return shorts, analyzer.summarize(shorts)


def print_short_report(df: pd.DataFrame, top_n: int = 10):
    """Print formatted short position report."""
    shorts, summary = analyze_shorts(df)
    
    print("=" * 80)
    print("SHORT POSITION ANALYSIS")
    print("=" * 80)
    print(f"\nPositions:       {summary['total']}")
    print(f"Value:           ${summary['value']:,.0f}")
    print(f"Adjusted Value:  ${summary['adjusted_value']:,.0f}")
    print(f"Avg Leverage:    {summary.get('avg_leverage', 0):.2f}x")
    print(f"Max Leverage:    {summary.get('max_leverage', 0):.0f}x")
    
    if summary.get('by_type'):
        print("\nBy Type:")
        for stype, count in summary['by_type'].items():
            print(f"  {stype}: {count}")
    
    if not shorts.empty:
        print(f"\n{'=' * 80}")
        print(f"TOP {min(top_n, len(shorts))} LARGEST")
        print("=" * 80)
        
        top = shorts.nlargest(top_n, 'adjusted_value')
        name_col = next((c for c in ['nameofissuer', 'n'] if c in top.columns), None)
        cols = [c for c in [name_col, 'titleofclass', 'short_type', 'leverage', 'adjusted_value'] 
                if c and c in top.columns]
        
        with pd.option_context('display.max_colwidth', 40, 'display.width', None):
            print(top[cols].to_string(index=False))


# ============================================================================
# EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Sample demonstrating various short types
    sample = pd.DataFrame({
        'nameOfIssuer': ['NVIDIA', 'PROSHARES ULTRAPRO SHORT QQQ', 'APPLE INC', 
                        'DIREXION 2X SHORT NVDA', 'TESLA', 'INVERSE S&P ETF'],
        'titleOfClass': ['COM', 'ULTRASHORT ETF', 'COM', '2X SH', 'COM', 'INVERSE'],
        'putCall': [None, None, 'Put', None, None, None],
        'value': [1000000, 500000, 200000, 300000, 800000, 150000],
        'cusip': ['67066G104', '74347W104', '037833100', 'TEST123', '88160R101', 'TEST456']
    })
    
    print("Sample Holdings:\n")
    print(sample[['nameOfIssuer', 'titleOfClass', 'putCall', 'value']])
    print("\n")
    
    print_short_report(sample)
    
    # Programmatic access
    print("\n" + "=" * 80)
    print("PROGRAMMATIC ACCESS")
    print("=" * 80)
    
    shorts, summary = analyze_shorts(sample)
    print(f"\nFound {len(shorts)} shorts, ${summary['adjusted_value']:,.0f} exposure")
    print(f"\nDetails:\n{shorts[['nameOfIssuer', 'short_type', 'leverage', 'adjusted_value']]}")