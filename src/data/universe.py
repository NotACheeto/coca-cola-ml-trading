"""
Asset Universe Definition for Coca-Cola ($KO) and Related Ecosystem.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class AssetUniverse:
    """Represents the cross-asset universe relevant to Coca-Cola ($KO)."""
    target: str = "KO"
    direct_competitors: List[str] = field(default_factory=lambda: ["PEP", "KDP", "MNST", "CELH"])
    bottlers: List[str] = field(default_factory=lambda: ["COKE", "KOF"])
    suppliers: List[str] = field(default_factory=lambda: ["BALL", "CCK"])
    macro_and_sector: List[str] = field(default_factory=lambda: ["XLP", "SPY", "UUP"])

    @property
    def all_tickers(self) -> List[str]:
        """Returns unique list of all tickers in the ecosystem."""
        ordered = [self.target] + self.direct_competitors + self.bottlers + self.suppliers + self.macro_and_sector
        # Preserve order while deduplicating
        return list(dict.fromkeys(ordered))

    @property
    def ecosystem_categories(self) -> Dict[str, List[str]]:
        """Returns categorization of all non-target assets."""
        return {
            "competitors": self.direct_competitors,
            "bottlers": self.bottlers,
            "suppliers": self.suppliers,
            "macro_sector": self.macro_and_sector,
        }

    @classmethod
    def from_config(cls, config: dict) -> "AssetUniverse":
        """Instantiate from YAML configuration dictionary."""
        universe_cfg = config.get("universe", {})
        return cls(
            target=config.get("target", "KO"),
            direct_competitors=universe_cfg.get("direct_competitors", ["PEP", "KDP", "MNST", "CELH"]),
            bottlers=universe_cfg.get("bottlers", ["COKE", "KOF"]),
            suppliers=universe_cfg.get("suppliers", ["BALL", "CCK"]),
            macro_and_sector=universe_cfg.get("macro_and_sector", ["XLP", "SPY", "UUP"]),
        )
