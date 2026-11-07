"""
Asset Universe Definition for Coca-Cola ($KO) and Related Ecosystem.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssetUniverse:
    """Represents the multi-asset ecosystem relevant to Coca-Cola ($KO)."""
    target: str = "KO"
    direct_competitors: List[str] = field(default_factory=lambda: ["PEP", "KDP", "MNST", "CELH"])
    staples_peers: List[str] = field(default_factory=lambda: ["PG", "MDLZ", "KMB", "GIS", "CL"])
    bottlers: List[str] = field(default_factory=lambda: ["COKE", "KOF"])
    suppliers: List[str] = field(default_factory=lambda: ["BALL", "CCK"])
    commodities: List[str] = field(default_factory=lambda: ["CANE", "DBA", "USO", "DBB"])
    rates_and_credit: List[str] = field(default_factory=lambda: ["TLT", "IEF", "HYG", "LQD"])
    macro_and_sector: List[str] = field(default_factory=lambda: ["XLP", "SPY", "UUP", "EEM", "VIXY"])

    @property
    def all_tickers(self) -> List[str]:
        """Returns unique list of all tickers in the ecosystem."""
        ordered = (
            [self.target]
            + self.direct_competitors
            + self.staples_peers
            + self.bottlers
            + self.suppliers
            + self.commodities
            + self.rates_and_credit
            + self.macro_and_sector
        )
        # Preserve order while deduplicating
        return list(dict.fromkeys(ordered))

    @property
    def ecosystem_categories(self) -> Dict[str, List[str]]:
        """Returns categorization of all non-target assets."""
        return {
            "competitors": self.direct_competitors,
            "staples_peers": self.staples_peers,
            "bottlers": self.bottlers,
            "suppliers": self.suppliers,
            "commodities": self.commodities,
            "rates_and_credit": self.rates_and_credit,
            "macro_sector": self.macro_and_sector,
        }

    @classmethod
    def from_config(cls, config: dict) -> "AssetUniverse":
        """Instantiate from YAML configuration dictionary."""
        universe_cfg = config.get("universe", {})
        return cls(
            target=config.get("target", "KO"),
            direct_competitors=universe_cfg.get("direct_competitors", ["PEP", "KDP", "MNST", "CELH"]),
            staples_peers=universe_cfg.get("staples_peers", ["PG", "MDLZ", "KMB", "GIS", "CL"]),
            bottlers=universe_cfg.get("bottlers", ["COKE", "KOF"]),
            suppliers=universe_cfg.get("suppliers", ["BALL", "CCK"]),
            commodities=universe_cfg.get("commodities", ["CANE", "DBA", "USO", "DBB"]),
            rates_and_credit=universe_cfg.get("rates_and_credit", ["TLT", "IEF", "HYG", "LQD"]),
            macro_and_sector=universe_cfg.get("macro_and_sector", ["XLP", "SPY", "UUP", "EEM", "VIXY"]),
        )
