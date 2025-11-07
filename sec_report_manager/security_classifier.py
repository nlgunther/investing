import re
from typing import Literal, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass


# ==================== DATA CLASSES ====================

@dataclass
class SecurityClassification:
    """
    Complete classification result for a security.
    
    Attributes:
        type: Security type (LEVERAGED_ETF, ETF, derivative, physical)
        asset_class: Underlying asset class (STOCK, DEBT, INDEX, COMMODITY, None)
        exposure: Directional exposure (long, short, None)
        leverage: Leverage multiplier (e.g., 2.0 for "2X", None for unleveraged)
    
    Examples:
        Physical stock: SecurityClassification(type="physical", asset_class="STOCK", 
                                               exposure=None, leverage=None)
        Leveraged ETF: SecurityClassification(type="LEVERAGED_ETF", asset_class="INDEX",
                                             exposure="long", leverage=3.0)
    """
    type: Literal["LEVERAGED_ETF", "ETF", "derivative", "physical"]
    asset_class: Optional[Literal["STOCK", "DEBT", "INDEX", "COMMODITY"]] = None
    exposure: Optional[Literal["long", "short"]] = None
    leverage: Optional[float] = None


# ==================== UTILITY CLASSES ====================

class ExposureDetector:
    """
    Detects directional exposure and leverage from security titles.
    
    Handles both explicit directional terms (LONG/SHORT/BULL/BEAR) and
    leverage multipliers (1X, 2X, 3X, etc.).
    """
    
    LONG_TERMS = {'LONG', 'BULL', 'LNG', 'BL'}
    SHORT_TERMS = {'SHORT', 'BEAR', 'INVERSE', 'SHT', 'SHR', 'INV', 'BR'}
    LEVERAGE_PATTERN = r'(-?\d+\.?\d*)\s*X'
    
    @classmethod
    def detect_exposure(cls, title: str) -> Optional[Literal["long", "short"]]:
        """
        Identify directional exposure from title terms.
        
        Returns "short" for bearish/inverse terms, "long" for bullish terms, None otherwise.
        """
        title_upper = title.upper()
        if any(term in title_upper for term in cls.SHORT_TERMS):
            return "short"
        if any(term in title_upper for term in cls.LONG_TERMS):
            return "long"
        return None
    
    @classmethod
    def extract_leverage(cls, title: str) -> Optional[float]:
        """
        Extract leverage multiplier from title.
        
        Examples: "2X" -> 2.0, "-1X" -> -1.0, "1.25X" -> 1.25
        """
        match = re.search(cls.LEVERAGE_PATTERN, title.upper())
        return float(match.group(1)) if match else None
    
    @classmethod
    def determine_final_exposure(cls, title: str) -> Optional[Literal["long", "short"]]:
        """
        Determine final exposure from directional terms.
        
        Note: Negative leverage (like "-1X") does NOT invert exposure.
        Both "-1X" and "SHORT" indicate short exposure and reinforce each other.
        """
        return cls.detect_exposure(title)


class AssetClassifier:
    """
    Identifies the underlying asset class of a security.
    
    Examines both issuer name and title to determine if the security
    relates to stocks, debt, indices, or commodities.
    """
    
    # Stock-related keywords
    STOCK_KEYWORDS = {
        'COM', 'COMMON', 'PREFERRED', 'PREF', 'EQUITY', 'SHARES', 'SHS',
        'CL A', 'CL B', 'CL C', 'CLASS A', 'CLASS B', 'CLASS C'
    }
    
    # Debt-related keywords
    DEBT_KEYWORDS = {
        'NOTE', 'BOND', 'DEBT', 'MORTGAGE', 'DEBENTURE', 'MTNF',
        'CONVERT', 'SENIOR', 'SUBORDINATED'
    }
    
    # Index and market segment keywords
    INDEX_KEYWORDS = {
        # Major indices
        'S&P', 'S&P500', 'SP500', 'RUSSELL', 'NASDAQ', 'DOW', 'DJIA',
        'MSCI', 'FTSE', 'CSI', 'NIKKEI', 'DAX', 'CAC',
        # Market segments
        'MIDCAP', 'SMCAP', 'SMALLCAP', 'LARGECAP', 'MICROCAP',
        'SMALL CAP', 'MID CAP', 'LARGE CAP',
        # Sectors
        'TECH', 'TECHNOLOGY', 'FINANCIAL', 'ENERGY', 'HEALTHCARE',
        'UTILITIES', 'INDUSTRIAL', 'CONSUMER', 'MATERIALS', 'TELECOM',
        'REAL ESTATE', 'AEROSPACE', 'TRANSPORTATION', 'RETAIL',
        # Geographic regions
        'EMERGING', 'EMG', 'ASIA', 'EUROPE', 'CHINA', 'JAPAN', 'INDIA',
        'LATIN', 'BRAZIL', 'MEXICO', 'VIETNAM', 'PERU', 'COLOMBIA',
        'PHILIPPINES', 'GLOBAL', 'INTERNATIONAL', 'WORLD', 'PACIFIC'
    }
    
    # Commodity keywords
    COMMODITY_KEYWORDS = {
        'GOLD', 'SILVER', 'PLATINUM', 'COPPER', 'ALUMINUM',
        'OIL', 'CRUDE', 'GAS', 'NATURAL GAS', 'PETROLEUM',
        'WHEAT', 'CORN', 'SOYBEAN', 'COTTON', 'SUGAR',
        'COFFEE', 'COCOA', 'LUMBER', 'CATTLE'
    }
    
    @classmethod
    def classify(cls, issuer: str, title: str) -> Optional[Literal["STOCK", "DEBT", "INDEX", "COMMODITY"]]:
        """
        Determine asset class from issuer name and title.
        
        Priority order: DEBT > STOCK > COMMODITY > INDEX
        (Debt is most specific, Index is most general)
        """
        combined = f"{issuer} {title}".upper()
        
        # Check DEBT first (most specific for physical securities)
        if any(keyword in combined for keyword in cls.DEBT_KEYWORDS):
            # Additional check for interest rates (strong debt indicator)
            if re.search(r'\d+\.?\d*%', title):
                return "DEBT"
            return "DEBT"
        
        # Check STOCK (specific equity indicators)
        if any(keyword in combined for keyword in cls.STOCK_KEYWORDS):
            return "STOCK"
        
        # Check COMMODITY (specific to raw materials)
        if any(keyword in combined for keyword in cls.COMMODITY_KEYWORDS):
            return "COMMODITY"
        
        # Check INDEX (most general - market segments and regions)
        if any(keyword in combined for keyword in cls.INDEX_KEYWORDS):
            return "INDEX"
        
        return None


# ==================== CLASSIFICATION RULES ====================

class ClassificationRule(ABC):
    """
    Abstract base for security classification rules.
    
    Each rule evaluates an (issuer, title) pair and returns a full
    SecurityClassification if the rule applies, or None otherwise.
    """
    
    @abstractmethod
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        """Evaluate if this rule classifies the security."""
        pass
    
    @property
    @abstractmethod
    def priority(self) -> int:
        """Rule priority - higher values evaluated first."""
        pass


class CorporateIssuerRule(ClassificationRule):
    """
    Corporate issuers (INC, CORP, LTD) issue physical securities.
    
    Real business companies have legal entity suffixes and rarely issue derivatives.
    This is a high-priority rule as it's highly reliable.
    """
    
    CORPORATE_SUFFIXES = {'INC', 'CORP', 'LTD', 'LP', 'LLC', 'MLP', 'CO', 'SA', 'AG', 'PLC'}
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        tokens = issuer.upper().replace('.', '').split()
        if not any(token in self.CORPORATE_SUFFIXES for token in tokens[-2:]):
            return None
        
        asset_class = AssetClassifier.classify(issuer, title)
        return SecurityClassification(type="physical", asset_class=asset_class)
    
    @property
    def priority(self) -> int:
        return 100


class LeveragedProductRule(ClassificationRule):
    """
    Products with leverage multiplier + directional terms.
    
    Identifies leveraged/inverse products that track underlying assets with amplification.
    Distinguishes between ETF-based products (LEVERAGED_ETF) and other derivatives.
    
    Examples:
        - "DIREXION SHS ETF TR, DLY AAPL BEAR 1X" -> LEVERAGED_ETF, STOCK, short, 1X
        - "GRANITESHARES ETF TR, 2X LONG NVDA DAI" -> LEVERAGED_ETF, STOCK, long, 2X
    """
    
    DIRECTIONAL_TERMS = ExposureDetector.LONG_TERMS | ExposureDetector.SHORT_TERMS
    ETF_KEYWORDS = ['ETF', 'TRUST', 'FUND', 'FDS']
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        leverage = ExposureDetector.extract_leverage(title)
        has_direction = any(term in title.upper() for term in self.DIRECTIONAL_TERMS)
        
        if not (leverage and has_direction):
            return None
        
        exposure = ExposureDetector.determine_final_exposure(title)
        asset_class = AssetClassifier.classify(issuer, title)
        is_etf = any(kw in issuer.upper() for kw in self.ETF_KEYWORDS)
        
        return SecurityClassification(
            type="LEVERAGED_ETF" if is_etf else "derivative",
            asset_class=asset_class,
            exposure=exposure,
            leverage=abs(leverage)
        )
    
    @property
    def priority(self) -> int:
        return 90


class UnderlyingAssetRule(ClassificationRule):
    """
    Title references specific tickers/indices not matching issuer.
    
    If a product's title mentions a ticker (like AAPL) or index (like S&P500)
    that doesn't appear in the issuer name, it's tracking that asset.
    """
    
    TICKER_SYMBOLS = {
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'NVDA', 'TSLA', 'META', 'COIN',
        'MSTR', 'BABA', 'INTC', 'UBER', 'PLTR', 'CRWD', 'SMCI', 'MARA',
        'RDDT', 'MRVL', 'AMD', 'NFLX', 'QCOM'
    }
    
    ETF_KEYWORDS = ['ETF', 'TRUST', 'FUND', 'FDS']
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        issuer_upper = issuer.upper()
        title_upper = title.upper()
        
        # Check for ticker symbols not in issuer name
        has_ticker = any(
            ticker in title_upper and ticker not in issuer_upper
            for ticker in self.TICKER_SYMBOLS
        )
        
        # Check for index references via AssetClassifier
        has_index = AssetClassifier.classify(issuer, title) == "INDEX"
        
        if not (has_ticker or has_index):
            return None
        
        exposure = ExposureDetector.determine_final_exposure(title)
        leverage = ExposureDetector.extract_leverage(title)
        asset_class = "STOCK" if has_ticker else AssetClassifier.classify(issuer, title)
        is_etf = any(kw in issuer_upper for kw in self.ETF_KEYWORDS)
        
        return SecurityClassification(
            type="LEVERAGED_ETF" if is_etf else "derivative",
            asset_class=asset_class,
            exposure=exposure,
            leverage=abs(leverage) if leverage else None
        )
    
    @property
    def priority(self) -> int:
        return 80


class DailyLeveragedProductRule(ClassificationRule):
    """
    Daily rebalancing products (typically leveraged/inverse ETFs).
    
    Products with "DAILY" or "DLY" are typically leveraged ETFs that
    rebalance daily to maintain target leverage.
    """
    
    DAILY_KEYWORDS = [r'\bDLY\b', r'\bDAILY\b', r'\bDEF\b', r'\bDEFIANCE\b']
    DIRECTIONAL_TERMS = ExposureDetector.LONG_TERMS | ExposureDetector.SHORT_TERMS
    ETF_KEYWORDS = ['ETF', 'TRUST', 'FUND', 'FDS']
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        title_upper = title.upper()
        has_daily = any(re.search(kw, title_upper) for kw in self.DAILY_KEYWORDS)
        leverage = ExposureDetector.extract_leverage(title)
        has_direction = any(term in title_upper for term in self.DIRECTIONAL_TERMS)
        
        if not (has_daily and (leverage or has_direction)):
            return None
        
        exposure = ExposureDetector.determine_final_exposure(title)
        asset_class = AssetClassifier.classify(issuer, title)
        is_etf = any(kw in issuer.upper() for kw in self.ETF_KEYWORDS)
        
        return SecurityClassification(
            type="LEVERAGED_ETF" if is_etf else "derivative",
            asset_class=asset_class,
            exposure=exposure,
            leverage=abs(leverage) if leverage else None
        )
    
    @property
    def priority(self) -> int:
        return 70


class SpecializedProductRule(ClassificationRule):
    """
    Specialized derivative products (crypto, VIX, branded).
    
    Handles cryptocurrency products, volatility products (VIX), and
    branded derivative products like T REX.
    """
    
    ETF_KEYWORDS = ['ETF', 'TRUST', 'FUND', 'FDS']
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        title_upper = title.upper()
        leverage = ExposureDetector.extract_leverage(title)
        
        # Crypto products (treated as commodity)
        is_crypto = any(c in title_upper for c in ['BITCOIN', 'ETHER', 'SOLANA', 'CRYPTO'])
        
        # VIX futures (volatility index)
        is_vix = 'VIX' in title_upper and ('FUT' in title_upper or leverage)
        
        # Branded products
        is_branded = ('T REX' in title_upper or 'TREX' in title_upper) and leverage
        
        if not (is_crypto or is_vix or is_branded):
            return None
        
        exposure = ExposureDetector.determine_final_exposure(title)
        asset_class = "COMMODITY" if (is_crypto or 'VIX' in title_upper) else AssetClassifier.classify(issuer, title)
        is_etf = any(kw in issuer.upper() for kw in self.ETF_KEYWORDS)
        
        return SecurityClassification(
            type="LEVERAGED_ETF" if is_etf else "derivative",
            asset_class=asset_class,
            exposure=exposure,
            leverage=abs(leverage) if leverage else None
        )
    
    @property
    def priority(self) -> int:
        return 60


class PhysicalSecurityRule(ClassificationRule):
    """
    Traditional physical securities (stocks, bonds, warrants).
    
    Identifies securities by their title containing standard security type keywords.
    This rule has lower priority as it's checked after derivative patterns.
    """
    
    PHYSICAL_PATTERNS = [
        r'\bCOM\b', r'\bCL\s*[A-Z]\b', r'\bNOTE\b', r'\bDEBT\b',
        r'\bPAR\s*\$', r'\d+\.?\d*%', r'\bREG\b', r'\bCERT\b',
        r'\bW\s+EXP\b', r'\bMTNF\b', r'\bBOND\b', r'\bPREF\b'
    ]
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        if not any(re.search(p, title.upper()) for p in self.PHYSICAL_PATTERNS):
            return None
        
        asset_class = AssetClassifier.classify(issuer, title)
        return SecurityClassification(type="physical", asset_class=asset_class)
    
    @property
    def priority(self) -> int:
        return 50


class PlainETFRule(ClassificationRule):
    """
    Non-leveraged ETFs without directional bias.
    
    Traditional index-tracking ETFs that don't use leverage or inverse strategies.
    These are passive investment vehicles that track indices 1:1.
    """
    
    ETF_ISSUERS = {
        'ETF TRUST', 'ETF TR', 'ISHARES', 'VANGUARD', 'SPDR', 'ARK ETF',
        'GLOBAL X', 'INVESCO', 'SCHWAB', 'WISDOMTREE', 'PROSHARES',
        'FIRST TRUST', 'STATE STREET'
    }
    
    def check(self, issuer: str, title: str) -> Optional[SecurityClassification]:
        issuer_upper = issuer.upper()
        if not any(etf in issuer_upper for etf in self.ETF_ISSUERS):
            return None
        
        # If has leverage or directional exposure, not a plain ETF
        if ExposureDetector.extract_leverage(title) or ExposureDetector.detect_exposure(title):
            return None
        
        asset_class = AssetClassifier.classify(issuer, title)
        return SecurityClassification(type="ETF", asset_class=asset_class or "INDEX")
    
    @property
    def priority(self) -> int:
        return 40


# ==================== MAIN CLASSIFIER ====================

class SecurityClassifier:
    """
    Rule-based security classifier with comprehensive categorization.
    
    Classifies securities by:
        - Type: LEVERAGED_ETF, ETF, derivative, physical
        - Asset Class: STOCK, DEBT, INDEX, COMMODITY
        - Exposure: long, short (for leveraged products)
        - Leverage: Multiplier (for leveraged products)
    
    Usage:
        classifier = SecurityClassifier()
        
        # Physical stock
        result = classifier.classify("AAON INC", "COM PAR $0.004")
        # -> SecurityClassification(type="physical", asset_class="STOCK", 
        #                           exposure=None, leverage=None)
        
        # Leveraged ETF
        result = classifier.classify("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X")
        # -> SecurityClassification(type="LEVERAGED_ETF", asset_class="STOCK",
        #                           exposure="short", leverage=1.0)
        
        # Plain ETF
        result = classifier.classify("ARK ETF TR", "INNOVATION ETF")
        # -> SecurityClassification(type="ETF", asset_class="INDEX",
        #                           exposure=None, leverage=None)
    """
    
    def __init__(self):
        self.rules = [
            CorporateIssuerRule(),
            LeveragedProductRule(),
            UnderlyingAssetRule(),
            DailyLeveragedProductRule(),
            SpecializedProductRule(),
            PhysicalSecurityRule(),
            PlainETFRule()
        ]
        self.rules.sort(key=lambda r: r.priority, reverse=True)
    
    def classify(self, nameofissuer: str, titleofclass: str) -> SecurityClassification:
        """
        Classify a security with full details.
        
        Args:
            nameofissuer: Issuer name (e.g., "AAON INC", "DIREXION SHS ETF TR")
            titleofclass: Security title (e.g., "COM", "DLY AAPL BEAR 1X")
        
        Returns:
            SecurityClassification with type, asset_class, exposure, and leverage
        """
        # Apply rules in priority order
        for rule in self.rules:
            result = rule.check(nameofissuer, titleofclass)
            if result:
                return result
        
        # Default fallback for unmatched ETF/Trust products
        issuer_upper = nameofissuer.upper()
        if any(kw in issuer_upper for kw in ['ETF', 'TRUST', 'FUND', 'FDS']):
            exposure = ExposureDetector.determine_final_exposure(titleofclass)
            leverage = ExposureDetector.extract_leverage(titleofclass)
            asset_class = AssetClassifier.classify(nameofissuer, titleofclass)
            
            security_type = "LEVERAGED_ETF" if (leverage or exposure) else "ETF"
            return SecurityClassification(
                type=security_type,
                asset_class=asset_class or "INDEX",
                exposure=exposure,
                leverage=abs(leverage) if leverage else None
            )
        
        # Final fallback
        return SecurityClassification(
            type="physical",
            asset_class=AssetClassifier.classify(nameofissuer, titleofclass)
        )


# ==================== CONVENIENCE FUNCTION ====================

def classify_security(nameofissuer: str, titleofclass: str) -> SecurityClassification:
    """Convenience function for one-off classifications."""
    return SecurityClassifier().classify(nameofissuer, titleofclass)


# ==================== TESTING ====================

if __name__ == "__main__":
    classifier = SecurityClassifier()
    
    test_cases = [
        # Leveraged ETFs - stocks
        ("DIREXION SHS ETF TR", "DLY AAPL BEAR 1X", "LEVERAGED_ETF", "STOCK", "short", 1.0),
        ("GRANITESHARES ETF TR", "2X LONG NVDA DAI", "LEVERAGED_ETF", "STOCK", "long", 2.0),
        
        # Leveraged ETFs - indices
        ("DIREXION SHS ETF TR", "DLY SMCAP BULL3X", "LEVERAGED_ETF", "INDEX", "long", 3.0),
        ("DIREXION SHS ETF TR", "DLY S&P500 BR 3X", "LEVERAGED_ETF", "INDEX", "short", 3.0),
        
        # Leveraged ETFs - commodities
        ("DIREXION SHS ETF TR", "DLY GOLD INDX 2X", "LEVERAGED_ETF", "COMMODITY", "long", 2.0),
        ("VOLATILITY SHS TR", "2X BITCOIN STRAT", "LEVERAGED_ETF", "COMMODITY", "long", 2.0),
        
        # Plain ETFs
        ("ARK ETF TR", "INNOVATION ETF", "ETF", "INDEX", None, None),
        ("ISHARES TR", "MSCI PERU AND GL", "ETF", "INDEX", None, None),
        ("GLOBAL X FDS", "GLBX MSCI COLUM", "ETF", "INDEX", None, None),
        
        # Physical - stocks
        ("AAON INC", "COM PAR $0.004", "physical", "STOCK", None, None),
        ("AAR CORP", "COM", "physical", "STOCK", None, None),
        ("ACV AUCTIONS INC", "COM CL A", "physical", "STOCK", None, None),
        
        # Physical - debt
        ("ZTO EXPRESS CAYMAN INC", "NOTE 1.500% 9/0", "physical", "DEBT", None, None),
        ("AIRBNB INC", "NOTE 3/1", "physical", "DEBT", None, None),
        ("AMERICAN AIRLINES GROUP INC", "NOTE 6.500% 7/0", "physical", "DEBT", None, None),
    ]
    
    print("Security Classifier - Enhanced with Asset Class Detection")
    print("=" * 110)
    print(f"{'Stat':4s} {'Issuer':30s} | {'Title':23s} | {'Type':13s} | {'Asset':8s} | {'Exp':5s} | Lev")
    print("=" * 110)
    
    correct = 0
    for issuer, title, exp_type, exp_asset, exp_exp, exp_lev in test_cases:
        result = classifier.classify(issuer, title)
        
        all_correct = (
            result.type == exp_type and
            result.asset_class == exp_asset and
            result.exposure == exp_exp and
            result.leverage == exp_lev
        )
        
        status = "✓" if all_correct else "✗"
        correct += all_correct
        
        res_exp = result.exposure or "-"
        res_lev = f"{result.leverage}X" if result.leverage else "-"
        res_asset = result.asset_class or "-"
        
        print(f"{status:4s} {issuer[:30]:30s} | {title[:23]:23s} | "
              f"{result.type:13s} | {res_asset:8s} | {res_exp:5s} | {res_lev}")
    
    print("=" * 110)
    print(f"Accuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.1f}%)")