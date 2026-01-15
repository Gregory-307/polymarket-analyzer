"""Backtesting engine for prediction market strategies.

Validates strategy performance using historical market data.
Supports favorite-longshot bias and arbitrage strategy testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class Trade:
    """Record of a simulated trade."""

    timestamp: datetime
    market_id: str
    strategy: str
    side: str  # 'YES' or 'NO'
    entry_price: float
    size: float
    resolved: bool = False
    resolution: str | None = None  # 'YES' or 'NO'
    exit_price: float | None = None
    pnl: float = 0.0


@dataclass
class BacktestResult:
    """Results from a backtest run.

    Attributes:
        strategy: Strategy name tested.
        start_date: Backtest start date.
        end_date: Backtest end date.
        initial_capital: Starting capital.
        final_capital: Ending capital.
        total_return: Total return percentage.
        sharpe_ratio: Risk-adjusted return.
        max_drawdown: Maximum peak-to-trough decline.
        win_rate: Percentage of winning trades.
        total_trades: Number of trades executed.
        trades: List of individual trades.
    """

    strategy: str
    start_date: datetime
    end_date: datetime
    initial_capital: float
    final_capital: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    avg_trade_pnl: float
    trades: list[Trade] = field(default_factory=list)
    equity_curve: list[tuple[datetime, float]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "strategy": self.strategy,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "initial_capital": self.initial_capital,
            "final_capital": self.final_capital,
            "total_return": f"{self.total_return:.2%}",
            "sharpe_ratio": round(self.sharpe_ratio, 2),
            "max_drawdown": f"{self.max_drawdown:.2%}",
            "win_rate": f"{self.win_rate:.2%}",
            "total_trades": self.total_trades,
            "avg_trade_pnl": round(self.avg_trade_pnl, 2),
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        return f"""
Backtest Results: {self.strategy}
{'=' * 50}
Period: {self.start_date.date()} to {self.end_date.date()}
Initial Capital: ${self.initial_capital:,.2f}
Final Capital: ${self.final_capital:,.2f}
Total Return: {self.total_return:.2%}
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2%}
Win Rate: {self.win_rate:.2%}
Total Trades: {self.total_trades}
Avg Trade P&L: ${self.avg_trade_pnl:.2f}
"""


class BacktestEngine:
    """Engine for backtesting prediction market strategies.

    Simulates strategy performance using historical market data,
    accounting for:
    - Entry/exit prices
    - Position sizing
    - Market resolution outcomes
    - Transaction costs

    Example:
        ```python
        engine = BacktestEngine(initial_capital=10000)

        # Load historical data
        engine.load_data(historical_markets)

        # Run favorite-longshot backtest
        result = engine.run_favorite_longshot(
            min_probability=0.95,
            min_edge=0.01,
        )

        print(result.summary())
        ```
    """

    def __init__(
        self,
        initial_capital: float = 10000,
        max_position_pct: float = 0.05,
        transaction_cost: float = 0.001,
    ):
        """Initialize backtest engine.

        Args:
            initial_capital: Starting capital in USD.
            max_position_pct: Maximum position size as % of capital.
            transaction_cost: Transaction cost as decimal (0.001 = 0.1%).
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.transaction_cost = transaction_cost

        self._data: pd.DataFrame | None = None
        self._trades: list[Trade] = []
        self._capital = initial_capital
        self._equity_curve: list[tuple[datetime, float]] = []

    def load_data(self, data: pd.DataFrame | list[dict]) -> None:
        """Load historical market data.

        Expected columns:
        - timestamp: datetime
        - market_id: str
        - question: str
        - yes_price: float
        - no_price: float
        - resolution: str ('YES', 'NO', or None if unresolved)
        - resolved_at: datetime or None

        Args:
            data: DataFrame or list of dicts with market snapshots.
        """
        if isinstance(data, list):
            self._data = pd.DataFrame(data)
        else:
            self._data = data.copy()

        # Ensure timestamp is datetime
        if "timestamp" in self._data.columns:
            self._data["timestamp"] = pd.to_datetime(self._data["timestamp"])

        # Sort by timestamp
        self._data = self._data.sort_values("timestamp")

    def run_favorite_longshot(
        self,
        min_probability: float = 0.95,
        min_edge: float = 0.01,
        hold_to_resolution: bool = True,
    ) -> BacktestResult:
        """Backtest favorite-longshot bias strategy.

        Simulates buying high-probability outcomes and holding to resolution.

        Args:
            min_probability: Minimum probability threshold.
            min_edge: Minimum estimated edge.
            hold_to_resolution: If True, hold until market resolves.

        Returns:
            BacktestResult with performance metrics.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self._reset()

        # Group by market to avoid duplicate entries
        markets = self._data.groupby("market_id").agg({
            "timestamp": "first",
            "question": "first",
            "yes_price": "first",
            "no_price": "first",
            "resolution": "last",
        }).reset_index()

        for _, row in markets.iterrows():
            yes_price = row["yes_price"]
            no_price = row["no_price"]
            resolution = row["resolution"]

            # Skip unresolved markets
            if pd.isna(resolution):
                continue

            # Check for high-probability YES
            if yes_price >= min_probability:
                # Estimate edge (simplified)
                estimated_fair = min(0.99, yes_price + 0.02)
                edge = estimated_fair - yes_price

                if edge >= min_edge:
                    self._execute_trade(
                        timestamp=row["timestamp"],
                        market_id=row["market_id"],
                        strategy="favorite_longshot",
                        side="YES",
                        price=yes_price,
                        resolution=resolution,
                    )

            # Check for high-probability NO (low YES price)
            elif yes_price <= (1 - min_probability):
                estimated_fair = min(0.99, no_price + 0.02)
                edge = estimated_fair - no_price

                if edge >= min_edge:
                    self._execute_trade(
                        timestamp=row["timestamp"],
                        market_id=row["market_id"],
                        strategy="favorite_longshot",
                        side="NO",
                        price=no_price,
                        resolution=resolution,
                    )

        return self._generate_result("favorite_longshot")

    def run_single_arb(
        self,
        min_profit_pct: float = 0.005,
    ) -> BacktestResult:
        """Backtest single-condition arbitrage strategy.

        Simulates buying both YES and NO when sum < $1.

        Args:
            min_profit_pct: Minimum profit percentage threshold.

        Returns:
            BacktestResult with performance metrics.
        """
        if self._data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        self._reset()

        markets = self._data.groupby("market_id").agg({
            "timestamp": "first",
            "yes_price": "first",
            "no_price": "first",
            "resolution": "last",
        }).reset_index()

        for _, row in markets.iterrows():
            yes_price = row["yes_price"]
            no_price = row["no_price"]
            resolution = row["resolution"]

            if pd.isna(resolution):
                continue

            sum_prices = yes_price + no_price
            profit_pct = 1.0 - sum_prices

            if profit_pct >= min_profit_pct:
                # Execute arbitrage: buy both YES and NO
                position_size = min(
                    self._capital * self.max_position_pct,
                    self._capital * 0.5,  # Don't use more than half on one trade
                )

                # Cost to buy both
                cost = position_size
                units = cost / sum_prices

                # Guaranteed payout is $1 per unit
                payout = units
                pnl = payout - cost - (cost * self.transaction_cost * 2)

                trade = Trade(
                    timestamp=row["timestamp"],
                    market_id=row["market_id"],
                    strategy="single_arb",
                    side="BOTH",
                    entry_price=sum_prices,
                    size=units,
                    resolved=True,
                    resolution=resolution,
                    exit_price=1.0,
                    pnl=pnl,
                )

                self._trades.append(trade)
                self._capital += pnl
                self._equity_curve.append((row["timestamp"], self._capital))

        return self._generate_result("single_arb")

    def _execute_trade(
        self,
        timestamp: datetime,
        market_id: str,
        strategy: str,
        side: str,
        price: float,
        resolution: str,
    ) -> None:
        """Execute a single trade."""
        # Calculate position size
        position_size = min(
            self._capital * self.max_position_pct,
            1000,  # Max $1000 per trade
        )

        if position_size <= 0:
            return

        # Calculate units
        units = position_size / price

        # Determine P&L based on resolution
        won = (side == resolution)
        if won:
            payout = units  # $1 per unit
            pnl = payout - position_size
        else:
            pnl = -position_size

        # Subtract transaction costs
        pnl -= position_size * self.transaction_cost

        trade = Trade(
            timestamp=timestamp,
            market_id=market_id,
            strategy=strategy,
            side=side,
            entry_price=price,
            size=units,
            resolved=True,
            resolution=resolution,
            exit_price=1.0 if won else 0.0,
            pnl=pnl,
        )

        self._trades.append(trade)
        self._capital += pnl
        self._equity_curve.append((timestamp, self._capital))

    def _reset(self) -> None:
        """Reset state for new backtest."""
        self._trades = []
        self._capital = self.initial_capital
        self._equity_curve = [(datetime.min, self.initial_capital)]

    def _generate_result(self, strategy: str) -> BacktestResult:
        """Generate backtest result from trades."""
        if not self._trades:
            return BacktestResult(
                strategy=strategy,
                start_date=datetime.now(),
                end_date=datetime.now(),
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_return=0.0,
                sharpe_ratio=0.0,
                max_drawdown=0.0,
                win_rate=0.0,
                total_trades=0,
                avg_trade_pnl=0.0,
                trades=[],
            )

        # Calculate metrics
        pnls = [t.pnl for t in self._trades]
        wins = sum(1 for t in self._trades if t.pnl > 0)

        total_return = (self._capital - self.initial_capital) / self.initial_capital
        win_rate = wins / len(self._trades) if self._trades else 0
        avg_pnl = sum(pnls) / len(pnls) if pnls else 0

        # Sharpe ratio (simplified - assumes daily returns)
        if len(pnls) > 1:
            import statistics
            mean_pnl = statistics.mean(pnls)
            std_pnl = statistics.stdev(pnls) if len(pnls) > 1 else 1
            sharpe = (mean_pnl / std_pnl) * (252 ** 0.5) if std_pnl > 0 else 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for _, equity in self._equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        return BacktestResult(
            strategy=strategy,
            start_date=self._trades[0].timestamp,
            end_date=self._trades[-1].timestamp,
            initial_capital=self.initial_capital,
            final_capital=self._capital,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            total_trades=len(self._trades),
            avg_trade_pnl=avg_pnl,
            trades=self._trades,
            equity_curve=self._equity_curve,
        )
