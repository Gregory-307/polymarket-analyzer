"""Configuration management for Polymarket Analyzer."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class PlatformConfig(BaseModel):
    """Configuration for a trading platform."""

    enabled: bool = True
    base_url: str
    timeout_seconds: int = 15
    rate_limit_per_second: int = 10


class PolymarketConfig(PlatformConfig):
    """Polymarket-specific configuration."""

    base_url: str = "https://clob.polymarket.com"
    gamma_url: str = "https://gamma-api.polymarket.com"
    chain_id: int = 137


class KalshiConfig(PlatformConfig):
    """Kalshi-specific configuration."""

    base_url: str = "https://api.elections.kalshi.com/trade-api/v2"


class StrategyConfig(BaseModel):
    """Base configuration for a trading strategy."""

    enabled: bool = True
    description: str = ""


class FavoriteLongshotConfig(StrategyConfig):
    """Configuration for favorite-longshot bias scanner."""

    min_probability: float = 0.90


class SingleArbConfig(StrategyConfig):
    """Configuration for single-condition arbitrage."""

    min_profit_usd: float = 0.50


class MultiArbConfig(StrategyConfig):
    """Configuration for multi-outcome arbitrage."""

    min_profit_usd: float = 1.00


class CrossPlatformConfig(StrategyConfig):
    """Configuration for cross-platform arbitrage."""

    min_spread: float = 0.02
    min_profit_after_fees: float = 0.01


class MarketMakerConfig(StrategyConfig):
    """Configuration for market making strategy."""

    enabled: bool = False
    target_spread: float = 0.02
    max_inventory: int = 100
    skew_factor: float = 0.1


class MetricConfig(BaseModel):
    """Configuration for a microstructure metric."""

    window_seconds: int = 300


class OrderImbalanceConfig(MetricConfig):
    """Configuration for order imbalance metric."""

    threshold_bullish: float = 0.3
    threshold_bearish: float = -0.3


class LiquidityDepthConfig(BaseModel):
    """Configuration for liquidity depth metric."""

    levels: list[float] = Field(default_factory=lambda: [0.01, 0.02, 0.05, 0.10])
    min_depth_usd: float = 100


class SpreadDynamicsConfig(MetricConfig):
    """Configuration for spread dynamics metric."""

    window_seconds: int = 60
    alert_threshold: float = 0.05


class ScanningConfig(BaseModel):
    """Configuration for opportunity scanning."""

    interval_seconds: int = 5
    max_markets_per_scan: int = 100
    alert_on_opportunity: bool = True


class GeneralConfig(BaseModel):
    """General application configuration."""

    log_level: str = "INFO"
    output_dir: str = "results"
    cache_ttl_seconds: int = 60


class Config(BaseModel):
    """Main configuration container."""

    general: GeneralConfig = Field(default_factory=GeneralConfig)
    polymarket: PolymarketConfig = Field(default_factory=PolymarketConfig)
    kalshi: KalshiConfig = Field(default_factory=KalshiConfig)
    favorite_longshot: FavoriteLongshotConfig = Field(
        default_factory=FavoriteLongshotConfig
    )
    single_arb: SingleArbConfig = Field(default_factory=SingleArbConfig)
    multi_arb: MultiArbConfig = Field(default_factory=MultiArbConfig)
    cross_platform: CrossPlatformConfig = Field(default_factory=CrossPlatformConfig)
    market_maker: MarketMakerConfig = Field(default_factory=MarketMakerConfig)
    order_imbalance: OrderImbalanceConfig = Field(default_factory=OrderImbalanceConfig)
    liquidity_depth: LiquidityDepthConfig = Field(default_factory=LiquidityDepthConfig)
    spread_dynamics: SpreadDynamicsConfig = Field(default_factory=SpreadDynamicsConfig)
    scanning: ScanningConfig = Field(default_factory=ScanningConfig)


class Credentials(BaseModel):
    """API credentials loaded from environment."""

    # Polymarket
    polymarket_private_key: str | None = None
    polymarket_chain_id: int = 137
    polymarket_signature_type: int = 0
    polymarket_funder_address: str | None = None

    # Kalshi
    kalshi_api_key_id: str | None = None
    kalshi_private_key_path: str | None = None

    @classmethod
    def from_env(cls) -> Credentials:
        """Load credentials from environment variables."""
        load_dotenv()
        return cls(
            polymarket_private_key=os.getenv("POLYMARKET_PRIVATE_KEY"),
            polymarket_chain_id=int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
            polymarket_signature_type=int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
            polymarket_funder_address=os.getenv("POLYMARKET_FUNDER_ADDRESS"),
            kalshi_api_key_id=os.getenv("KALSHI_API_KEY_ID"),
            kalshi_private_key_path=os.getenv("KALSHI_PRIVATE_KEY_PATH"),
        )

    @property
    def has_polymarket(self) -> bool:
        """Check if Polymarket credentials are configured."""
        return self.polymarket_private_key is not None

    @property
    def has_kalshi(self) -> bool:
        """Check if Kalshi credentials are configured."""
        return (
            self.kalshi_api_key_id is not None
            and self.kalshi_private_key_path is not None
        )


def load_config(config_path: str | Path | None = None) -> Config:
    """Load configuration from JSON file.

    Args:
        config_path: Path to config file. Defaults to configs/default.json.

    Returns:
        Loaded configuration object.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "configs" / "default.json"

    config_path = Path(config_path)

    if not config_path.exists():
        return Config()

    with open(config_path) as f:
        data = json.load(f)

    # Flatten nested config structure
    flat_data: dict[str, Any] = {}
    flat_data["general"] = data.get("general", {})

    # Platforms
    platforms = data.get("platforms", {})
    flat_data["polymarket"] = platforms.get("polymarket", {})
    flat_data["kalshi"] = platforms.get("kalshi", {})

    # Strategies
    strategies = data.get("strategies", {})
    flat_data["favorite_longshot"] = strategies.get("favorite_longshot", {})
    flat_data["single_arb"] = strategies.get("single_arb", {})
    flat_data["multi_arb"] = strategies.get("multi_arb", {})
    flat_data["cross_platform"] = strategies.get("cross_platform", {})
    flat_data["market_maker"] = strategies.get("market_maker", {})

    # Metrics
    metrics = data.get("metrics", {})
    flat_data["order_imbalance"] = metrics.get("order_imbalance", {})
    flat_data["liquidity_depth"] = metrics.get("liquidity_depth", {})
    flat_data["spread_dynamics"] = metrics.get("spread_dynamics", {})

    # Scanning
    flat_data["scanning"] = data.get("scanning", {})

    return Config(**flat_data)
