"""Financial Markets Strategy.

Compares prediction market prices on financial events to options-implied probabilities.

For events like "Will BTC exceed $100k by March?", options markets price this exactly
via Black-Scholes. If prediction markets diverge from options-implied fair value,
there's a potential opportunity.

Key advantages over other strategies:
1. Fair value is CALCULABLE (not opinion-based)
2. Hedging is POSSIBLE (can offset with options/futures)
3. Information is QUANTIFIABLE (financial data, not sentiment)

Requirements:
- External data source for options (Deribit for crypto, CME for rates)
- Black-Scholes calculator
- Market identification (find price threshold markets)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

from ..core.utils import get_logger, utc_now

if TYPE_CHECKING:
    from ..adapters.base import Market

logger = get_logger(__name__)


def normal_cdf(x: float) -> float:
    """Standard normal cumulative distribution function."""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def normal_pdf(x: float) -> float:
    """Standard normal probability density function."""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes_digital_call(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
) -> float:
    """Calculate probability that spot > strike at expiry using Black-Scholes.

    This is the fair value for a digital/binary option that pays $1 if spot > strike.

    Args:
        spot: Current spot price.
        strike: Strike price (threshold).
        time_to_expiry: Time to expiry in years.
        volatility: Annualized implied volatility (e.g., 0.60 for 60%).
        risk_free_rate: Risk-free rate (default 5%).

    Returns:
        Probability that spot > strike at expiry (0 to 1).
    """
    if time_to_expiry <= 0:
        return 1.0 if spot > strike else 0.0

    if volatility <= 0:
        return 1.0 if spot > strike else 0.0

    d2 = (
        math.log(spot / strike) + (risk_free_rate - 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * math.sqrt(time_to_expiry))

    return normal_cdf(d2)


@dataclass
class DigitalGreeks:
    """Greeks for a digital/binary option.

    Digital options have different Greeks than vanilla options because
    their payoff is discontinuous at the strike.

    Attributes:
        delta: Sensitivity to spot price change (d(price)/d(spot)).
        gamma: Second derivative wrt spot (d(delta)/d(spot)).
        theta: Time decay per day (d(price)/d(time)).
        vega: Sensitivity to volatility (d(price)/d(vol), per 1% vol change).
    """

    delta: float
    gamma: float
    theta: float  # Per day
    vega: float  # Per 1% vol change


def calculate_digital_greeks(
    spot: float,
    strike: float,
    time_to_expiry: float,
    volatility: float,
    risk_free_rate: float = 0.05,
) -> DigitalGreeks:
    """Calculate Greeks for a digital call option.

    For a digital call paying $1 if S > K at expiry:
    - Price = N(d2)
    - Delta = n(d2) / (S * sigma * sqrt(T))
    - Gamma = -n(d2) * d1 / (S^2 * sigma^2 * T)
    - Theta = n(d2) * (d1 / (2*T) + r) (per year)
    - Vega = -n(d2) * d1 / sigma

    Args:
        spot: Current spot price.
        strike: Strike price.
        time_to_expiry: Time to expiry in years.
        volatility: Annualized implied volatility.
        risk_free_rate: Risk-free rate.

    Returns:
        DigitalGreeks with all sensitivities.
    """
    if time_to_expiry <= 0 or volatility <= 0:
        return DigitalGreeks(delta=0, gamma=0, theta=0, vega=0)

    sqrt_t = math.sqrt(time_to_expiry)
    d1 = (
        math.log(spot / strike) + (risk_free_rate + 0.5 * volatility**2) * time_to_expiry
    ) / (volatility * sqrt_t)

    d2 = d1 - volatility * sqrt_t

    n_d2 = normal_pdf(d2)

    # Delta: sensitivity to spot
    delta = n_d2 / (spot * volatility * sqrt_t)

    # Gamma: rate of change of delta
    gamma = -n_d2 * d1 / (spot * spot * volatility * volatility * time_to_expiry)

    # Theta: time decay (convert to per day)
    theta_annual = n_d2 * (d1 / (2 * time_to_expiry) + risk_free_rate)
    theta = -theta_annual / 365  # Negative because value decays

    # Vega: sensitivity to volatility (per 1% change)
    vega = -n_d2 * d1 / (volatility * 100)

    return DigitalGreeks(
        delta=delta,
        gamma=gamma,
        theta=theta,
        vega=vega,
    )


@dataclass
class FinancialOpportunity:
    """Represents a financial markets mispricing opportunity.

    Attributes:
        market: The prediction market.
        underlying: The underlying asset (BTC, ETH, etc.).
        threshold: Price threshold from market question.
        expiry: Market expiration date.
        market_price: Prediction market YES price.
        fair_value: Options-implied probability.
        edge: Difference (fair_value - market_price).
        implied_vol: Implied volatility used.
        spot_price: Current spot price.
        greeks: Option Greeks (delta, gamma, theta, vega).
    """

    market: Market
    underlying: str
    threshold: float
    expiry: datetime
    market_price: float
    fair_value: float
    edge: float
    implied_vol: float
    spot_price: float
    greeks: DigitalGreeks | None = None

    @property
    def edge_pct(self) -> float:
        """Edge as percentage."""
        return self.edge * 100

    @property
    def direction(self) -> str:
        """Direction of trade."""
        if self.edge > 0:
            return "BUY"  # Market underpriced
        else:
            return "SELL"  # Market overpriced

    @property
    def hedge_ratio(self) -> float | None:
        """Suggested delta hedge ratio.

        Returns contracts of underlying needed to hedge per prediction market contract.
        Positive means buy underlying, negative means sell.
        """
        if self.greeks is None:
            return None
        # If buying prediction market (edge > 0), need to sell delta
        # If selling prediction market (edge < 0), need to buy delta
        if self.edge > 0:
            return -self.greeks.delta  # Sell this much to hedge long position
        else:
            return self.greeks.delta  # Buy this much to hedge short position


class FinancialMarketsStrategy:
    """Strategy for comparing prediction markets to options-implied probabilities.

    Identifies price threshold markets (e.g., "Will BTC > $X by date Y?")
    and calculates fair value using Black-Scholes.

    Example:
        ```python
        strategy = FinancialMarketsStrategy()

        # Parse market
        parsed = strategy.parse_price_threshold_market(market)
        if parsed:
            underlying, threshold, expiry = parsed

            # Calculate fair value (need external data)
            fair_value = strategy.calculate_fair_value(
                spot=92000,
                threshold=100000,
                expiry=expiry,
                implied_vol=0.60
            )

            # Compare to market
            edge = fair_value - market.yes_price
        ```

    Note: Full functionality requires external options data integration.
    """

    # Patterns to identify price threshold markets
    PRICE_PATTERNS = [
        # "Will BTC be above $100,000 on March 31?"
        r"will\s+(\w+)\s+(?:be\s+)?(?:above|over|exceed|greater than)\s+\$?([\d,]+[km]?)",
        # "BTC above $100k by March"
        r"(\w+)\s+(?:above|over|exceed)\s+\$?([\d,]+[km]?)",
        # "Bitcoin to hit $100,000" or "Bitcoin hit $1m"
        r"(bitcoin|btc|ethereum|eth)\s+(?:to\s+)?(?:hit|reach)\s+\$?([\d,]+[km]?)",
    ]

    # Asset name mappings
    ASSET_ALIASES = {
        "btc": "BTC",
        "bitcoin": "BTC",
        "eth": "ETH",
        "ethereum": "ETH",
        "sol": "SOL",
        "solana": "SOL",
    }

    def __init__(
        self,
        min_edge: float = 0.03,
        default_risk_free_rate: float = 0.05,
    ):
        """Initialize strategy.

        Args:
            min_edge: Minimum edge to report (default 3%).
            default_risk_free_rate: Risk-free rate for calculations.
        """
        self.min_edge = min_edge
        self.risk_free_rate = default_risk_free_rate

        # External data sources (to be integrated)
        self._spot_prices: dict[str, float] = {}
        self._implied_vols: dict[str, float] = {}

    def parse_price_threshold_market(
        self,
        market: Market,
    ) -> tuple[str, float, datetime] | None:
        """Parse a market question to extract price threshold details.

        Args:
            market: Market to parse.

        Returns:
            Tuple of (underlying_asset, threshold_price, expiry_date) or None.
        """
        question = market.question.lower()

        for pattern in self.PRICE_PATTERNS:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                asset_raw = match.group(1).lower()
                price_raw = match.group(2).replace(",", "")

                # Normalize asset name
                asset = self.ASSET_ALIASES.get(asset_raw, asset_raw.upper())

                # Parse price (handle "k" and "m" suffixes)
                price_raw_lower = price_raw.lower()
                if price_raw_lower.endswith("m"):
                    price = float(price_raw_lower[:-1]) * 1_000_000
                elif price_raw_lower.endswith("k"):
                    price = float(price_raw_lower[:-1]) * 1_000
                else:
                    price = float(price_raw)

                # Use market end date as expiry
                expiry = market.end_date or utc_now()

                return (asset, price, expiry)

        return None

    def calculate_fair_value(
        self,
        spot: float,
        threshold: float,
        expiry: datetime,
        implied_vol: float,
    ) -> float:
        """Calculate fair value probability using Black-Scholes.

        Args:
            spot: Current spot price.
            threshold: Price threshold.
            expiry: Expiration date.
            implied_vol: Annualized implied volatility.

        Returns:
            Fair value probability (0 to 1).
        """
        # Calculate time to expiry in years
        now = utc_now()
        delta = expiry - now
        time_to_expiry = max(0, delta.total_seconds() / (365.25 * 24 * 3600))

        return black_scholes_digital_call(
            spot=spot,
            strike=threshold,
            time_to_expiry=time_to_expiry,
            volatility=implied_vol,
            risk_free_rate=self.risk_free_rate,
        )

    def check_market(
        self,
        market: Market,
        spot_price: float | None = None,
        implied_vol: float | None = None,
    ) -> FinancialOpportunity | None:
        """Check a single market for financial mispricing.

        Args:
            market: Market to check.
            spot_price: Current spot price (or uses cached).
            implied_vol: Implied volatility (or uses cached/default).

        Returns:
            FinancialOpportunity if found, None otherwise.
        """
        parsed = self.parse_price_threshold_market(market)
        if not parsed:
            return None

        underlying, threshold, expiry = parsed

        # Get spot price
        spot = spot_price or self._spot_prices.get(underlying)
        if not spot:
            logger.debug(
                "financial_no_spot_price",
                market_id=market.id,
                underlying=underlying,
            )
            return None

        # Get implied volatility (default to 60% for crypto)
        vol = implied_vol or self._implied_vols.get(underlying, 0.60)

        # Calculate time to expiry
        now = utc_now()
        delta = expiry - now
        time_to_expiry = max(0, delta.total_seconds() / (365.25 * 24 * 3600))

        # Calculate fair value
        fair_value = self.calculate_fair_value(spot, threshold, expiry, vol)

        # Compare to market
        edge = fair_value - market.yes_price

        if abs(edge) < self.min_edge:
            return None

        # Calculate Greeks
        greeks = calculate_digital_greeks(
            spot=spot,
            strike=threshold,
            time_to_expiry=time_to_expiry,
            volatility=vol,
            risk_free_rate=self.risk_free_rate,
        )

        return FinancialOpportunity(
            market=market,
            underlying=underlying,
            threshold=threshold,
            expiry=expiry,
            market_price=market.yes_price,
            fair_value=fair_value,
            edge=edge,
            implied_vol=vol,
            spot_price=spot,
            greeks=greeks,
        )

    def set_spot_price(self, asset: str, price: float) -> None:
        """Set spot price for an asset.

        Args:
            asset: Asset symbol (BTC, ETH, etc.).
            price: Current spot price.
        """
        self._spot_prices[asset.upper()] = price

    def set_implied_vol(self, asset: str, vol: float) -> None:
        """Set implied volatility for an asset.

        Args:
            asset: Asset symbol.
            vol: Annualized implied volatility (0.60 = 60%).
        """
        self._implied_vols[asset.upper()] = vol

    def scan(
        self,
        markets: list[Market],
        spot_prices: dict[str, float] | None = None,
        implied_vols: dict[str, float] | None = None,
    ) -> list[FinancialOpportunity]:
        """Scan markets for financial mispricing opportunities.

        Args:
            markets: Markets to scan.
            spot_prices: Dict of asset -> spot price.
            implied_vols: Dict of asset -> implied vol.

        Returns:
            List of opportunities sorted by edge.
        """
        # Update prices
        if spot_prices:
            for asset, price in spot_prices.items():
                self.set_spot_price(asset, price)

        if implied_vols:
            for asset, vol in implied_vols.items():
                self.set_implied_vol(asset, vol)

        opportunities = []
        parsed_count = 0

        for market in markets:
            if not market.is_active:
                continue

            parsed = self.parse_price_threshold_market(market)
            if parsed:
                parsed_count += 1

            opp = self.check_market(market)
            if opp:
                opportunities.append(opp)
                logger.info(
                    "financial_opportunity",
                    market_id=market.id,
                    underlying=opp.underlying,
                    threshold=opp.threshold,
                    market_price=f"{opp.market_price:.2%}",
                    fair_value=f"{opp.fair_value:.2%}",
                    edge=f"{opp.edge:.2%}",
                )

        # Sort by absolute edge
        opportunities.sort(key=lambda x: abs(x.edge), reverse=True)

        logger.info(
            "financial_scan_complete",
            parsed=parsed_count,
            opportunities=len(opportunities),
        )

        return opportunities

    async def scan_with_live_data(
        self,
        markets: list["Market"],
    ) -> list[FinancialOpportunity]:
        """Scan markets using live data from Deribit.

        Fetches spot prices and implied volatility from Deribit options chain,
        then scans for mispriced prediction markets.

        Args:
            markets: Markets to scan.

        Returns:
            List of opportunities sorted by edge.
        """
        from ..adapters.deribit_options import DeribitOptionsClient

        async with DeribitOptionsClient() as client:
            # Fetch BTC data
            btc_spot = await client.get_spot("BTC")
            btc_vol = await client.get_atm_iv("BTC", days_to_expiry=30)
            if not btc_vol:
                btc_vol = await client.get_historical_vol("BTC")

            # Fetch ETH data
            eth_spot = await client.get_spot("ETH")
            eth_vol = await client.get_atm_iv("ETH", days_to_expiry=30)
            if not eth_vol:
                eth_vol = await client.get_historical_vol("ETH")

            # Update prices
            if btc_spot:
                self.set_spot_price("BTC", btc_spot)
            if eth_spot:
                self.set_spot_price("ETH", eth_spot)
            if btc_vol:
                self.set_implied_vol("BTC", btc_vol)
            if eth_vol:
                self.set_implied_vol("ETH", eth_vol)

            logger.info(
                "financial_markets_data",
                btc_spot=f"${btc_spot:,.0f}" if btc_spot else "N/A",
                btc_vol=f"{btc_vol:.1%}" if btc_vol else "N/A",
                eth_spot=f"${eth_spot:,.0f}" if eth_spot else "N/A",
                eth_vol=f"{eth_vol:.1%}" if eth_vol else "N/A",
            )

        return self.scan(markets)
