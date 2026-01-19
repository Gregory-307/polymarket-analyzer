"""Proper backtester using real historical data.

Replaces Monte Carlo simulation with actual historical prices and resolutions.
Walk-forward validation with train/test split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING
import math

from ..core.utils import get_logger
from ..storage.database import Database

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


@dataclass
class BacktestTrade:
    """Single trade in backtest.

    Attributes:
        market_id: Market identifier.
        entry_time: Time position was entered.
        entry_price: Price at entry.
        side: 'YES' or 'NO'.
        size: Position size in dollars.
        exit_time: Time position was closed (resolution).
        outcome: 'YES', 'NO', or None if not yet resolved.
        pnl: Profit/loss in dollars.
        pnl_pct: Profit/loss percentage.
    """

    market_id: str
    entry_time: datetime
    entry_price: float
    side: str
    size: float
    exit_time: datetime | None = None
    outcome: str | None = None
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """Complete backtest results.

    Attributes:
        strategy: Strategy name.
        start_date: Backtest start date.
        end_date: Backtest end date.
        trades: List of trades executed.
        total_trades: Number of trades.
        wins: Number of winning trades.
        losses: Number of losing trades.
        win_rate: Win rate.
        total_pnl: Total profit/loss in dollars.
        total_return: Total return as percentage.
        sharpe_ratio: Sharpe ratio (annualized).
        max_drawdown: Maximum drawdown.
        profit_factor: Gross profit / gross loss.
        avg_win: Average winning trade.
        avg_loss: Average losing trade.
    """

    strategy: str
    start_date: datetime | None
    end_date: datetime | None
    trades: list[BacktestTrade] = field(default_factory=list)
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0


class HistoricalBacktester:
    """Backtester using real historical data.

    Uses actual market snapshots and resolutions from the database
    to simulate strategy performance with walk-forward validation.

    Usage:
        async with Database() as db:
            backtester = HistoricalBacktester(db)
            result = await backtester.run(
                strategy='favorite_longshot',
                min_probability=0.90,
                bet_size=25,
                capital=1000,
            )
            backtester.print_report(result)
    """

    def __init__(
        self,
        database: Database,
        train_ratio: float = 0.6,
    ):
        """Initialize backtester.

        Args:
            database: Database instance with historical data.
            train_ratio: Fraction of data for training (default: 60%).
        """
        self.db = database
        self.train_ratio = train_ratio

    async def run(
        self,
        strategy: str = "favorite_longshot",
        min_probability: float = 0.90,
        bet_size: float = 25.0,
        capital: float = 1000.0,
        max_positions: int = 10,
    ) -> BacktestResult:
        """Run backtest on historical data.

        Args:
            strategy: Strategy to backtest.
            min_probability: Minimum probability for favorite-longshot.
            bet_size: Size of each position in dollars.
            capital: Starting capital.
            max_positions: Maximum concurrent positions.

        Returns:
            BacktestResult with performance metrics.
        """
        # Get all resolutions (markets that have completed)
        resolutions = await self.db.get_resolutions(limit=100000)

        if not resolutions:
            logger.warning("No resolution data available for backtest")
            return BacktestResult(strategy=strategy, start_date=None, end_date=None)

        # Get snapshots for these markets (entry prices)
        market_data = await self._prepare_market_data(resolutions)

        if not market_data:
            logger.warning("No matching market snapshots found")
            return BacktestResult(strategy=strategy, start_date=None, end_date=None)

        # Sort by resolution date for walk-forward
        market_data.sort(key=lambda x: x["resolved_at"])

        # Split train/test
        split_idx = int(len(market_data) * self.train_ratio)
        # train_data = market_data[:split_idx]  # Reserved for parameter tuning
        test_data = market_data[split_idx:]

        logger.info(
            f"Backtest data: {len(market_data)} markets, "
            f"train={split_idx}, test={len(test_data)}"
        )

        # Run strategy on test data
        if strategy == "favorite_longshot":
            trades = self._run_favorite_longshot(
                test_data, min_probability, bet_size, capital, max_positions
            )
        else:
            logger.error(f"Unknown strategy: {strategy}")
            return BacktestResult(strategy=strategy, start_date=None, end_date=None)

        # Calculate metrics
        result = self._calculate_metrics(trades, strategy, capital)

        if test_data:
            result.start_date = test_data[0]["resolved_at"]
            result.end_date = test_data[-1]["resolved_at"]

        return result

    async def _prepare_market_data(
        self,
        resolutions: list[dict],
    ) -> list[dict]:
        """Prepare market data by matching resolutions with snapshots.

        Args:
            resolutions: List of resolution records.

        Returns:
            List of market data dicts with entry prices and outcomes.
        """
        market_data = []

        for res in resolutions:
            market_id = res["market_id"]

            # Get the first snapshot for this market (entry price)
            snapshots = await self.db.get_snapshots(market_id, limit=1)
            if not snapshots:
                continue

            # Use earliest snapshot as entry point
            snapshot = snapshots[-1] if len(snapshots) > 1 else snapshots[0]

            market_data.append({
                "market_id": market_id,
                "question": res["question"],
                "entry_price_yes": snapshot.get("yes_price", 0.5),
                "entry_price_no": snapshot.get("no_price", 0.5),
                "entry_time": snapshot.get("timestamp"),
                "outcome": res["outcome"],
                "final_price": res.get("final_price"),
                "resolved_at": res["resolved_at"],
            })

        return market_data

    def _run_favorite_longshot(
        self,
        market_data: list[dict],
        min_probability: float,
        bet_size: float,
        capital: float,
        max_positions: int,
    ) -> list[BacktestTrade]:
        """Run favorite-longshot strategy on market data.

        Args:
            market_data: List of market data dicts.
            min_probability: Minimum probability threshold.
            bet_size: Position size in dollars.
            capital: Starting capital.
            max_positions: Maximum concurrent positions.

        Returns:
            List of executed trades.
        """
        trades = []
        balance = capital
        active_positions = 0

        for market in market_data:
            # Skip if insufficient capital
            if balance < bet_size:
                break

            # Skip if at max positions
            if active_positions >= max_positions:
                continue

            # Determine if market qualifies
            yes_price = market["entry_price_yes"]
            no_price = market["entry_price_no"]

            if yes_price >= min_probability:
                side = "YES"
                entry_price = yes_price
            elif no_price >= min_probability:
                side = "NO"
                entry_price = no_price
            else:
                continue

            # Create trade
            outcome = market["outcome"]
            won = (side == outcome)

            if won:
                # Won: receive $1 per share, paid entry_price
                pnl = bet_size * (1 - entry_price) / entry_price
            else:
                # Lost: lose entire stake
                pnl = -bet_size

            trade = BacktestTrade(
                market_id=market["market_id"],
                entry_time=datetime.fromisoformat(market["entry_time"].replace("Z", "+00:00"))
                    if isinstance(market["entry_time"], str) else market["entry_time"],
                entry_price=entry_price,
                side=side,
                size=bet_size,
                exit_time=datetime.fromisoformat(market["resolved_at"].replace("Z", "+00:00"))
                    if isinstance(market["resolved_at"], str) else market["resolved_at"],
                outcome=outcome,
                pnl=pnl,
                pnl_pct=pnl / bet_size,
            )

            trades.append(trade)
            balance += pnl
            active_positions += 1

        return trades

    def _calculate_metrics(
        self,
        trades: list[BacktestTrade],
        strategy: str,
        capital: float,
    ) -> BacktestResult:
        """Calculate performance metrics from trades.

        Args:
            trades: List of executed trades.
            strategy: Strategy name.
            capital: Starting capital.

        Returns:
            BacktestResult with computed metrics.
        """
        if not trades:
            return BacktestResult(
                strategy=strategy,
                start_date=None,
                end_date=None,
            )

        # Basic counts
        wins = sum(1 for t in trades if t.pnl > 0)
        losses = sum(1 for t in trades if t.pnl <= 0)
        total_trades = len(trades)
        win_rate = wins / total_trades if total_trades > 0 else 0

        # P&L
        total_pnl = sum(t.pnl for t in trades)
        total_return = total_pnl / capital if capital > 0 else 0

        # Win/loss averages
        winning_trades = [t for t in trades if t.pnl > 0]
        losing_trades = [t for t in trades if t.pnl <= 0]
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0

        # Profit factor
        gross_profit = sum(t.pnl for t in winning_trades)
        gross_loss = abs(sum(t.pnl for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        equity = capital
        peak = capital
        max_dd = 0.0

        for trade in trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak if peak > 0 else 0
            max_dd = max(max_dd, dd)

        # Sharpe ratio (simplified - using trade returns)
        returns = [t.pnl_pct for t in trades]
        if len(returns) > 1:
            avg_return = sum(returns) / len(returns)
            std_return = math.sqrt(sum((r - avg_return) ** 2 for r in returns) / (len(returns) - 1))
            # Annualize assuming ~250 trading days, rough estimate
            sharpe = (avg_return / std_return) * math.sqrt(250) if std_return > 0 else 0
        else:
            sharpe = 0

        return BacktestResult(
            strategy=strategy,
            start_date=trades[0].entry_time if trades else None,
            end_date=trades[-1].exit_time if trades else None,
            trades=trades,
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            total_return=total_return,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            profit_factor=profit_factor,
            avg_win=avg_win,
            avg_loss=avg_loss,
        )

    def print_report(self, result: BacktestResult) -> None:
        """Print formatted backtest report.

        Args:
            result: BacktestResult to display.
        """
        print("\n" + "=" * 70)
        print(f"  BACKTEST RESULTS: {result.strategy.upper()}")
        print("=" * 70)

        if result.start_date and result.end_date:
            print(f"\n  Period: {result.start_date} to {result.end_date}")

        print(f"\n  Total Trades: {result.total_trades}")
        print(f"  Wins: {result.wins} | Losses: {result.losses}")
        print(f"  Win Rate: {result.win_rate:.1%}")

        print("\n" + "-" * 70)
        print("  P&L SUMMARY")
        print("-" * 70)

        print(f"\n  Total P&L: ${result.total_pnl:,.2f}")
        print(f"  Total Return: {result.total_return:.1%}")
        print(f"  Average Win: ${result.avg_win:,.2f}")
        print(f"  Average Loss: ${result.avg_loss:,.2f}")
        print(f"  Profit Factor: {result.profit_factor:.2f}")

        print("\n" + "-" * 70)
        print("  RISK METRICS")
        print("-" * 70)

        print(f"\n  Sharpe Ratio: {result.sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {result.max_drawdown:.1%}")

        print("\n" + "=" * 70)


async def run_walk_forward_backtest(
    database: Database,
    strategy: str = "favorite_longshot",
    n_folds: int = 5,
    **kwargs,
) -> list[BacktestResult]:
    """Run walk-forward validation with multiple folds.

    Args:
        database: Database instance.
        strategy: Strategy to test.
        n_folds: Number of folds for cross-validation.
        **kwargs: Additional arguments for backtester.

    Returns:
        List of BacktestResult for each fold.
    """
    backtester = HistoricalBacktester(database)

    # Get all resolutions
    resolutions = await database.get_resolutions(limit=100000)
    if not resolutions:
        return []

    # Sort by date
    resolutions.sort(key=lambda x: x["resolved_at"])

    fold_size = len(resolutions) // n_folds
    results = []

    for fold in range(n_folds):
        # Each fold uses data up to this point for training,
        # and the next chunk for testing
        test_start = fold * fold_size
        test_end = min((fold + 1) * fold_size, len(resolutions))

        if test_start >= len(resolutions):
            break

        # Create a temporary backtester with this fold's train_ratio
        fold_backtester = HistoricalBacktester(
            database,
            train_ratio=fold / n_folds if fold > 0 else 0.01,
        )

        result = await fold_backtester.run(strategy=strategy, **kwargs)
        results.append(result)

        logger.info(f"Fold {fold + 1}/{n_folds}: {result.total_trades} trades, "
                   f"return={result.total_return:.1%}")

    return results
