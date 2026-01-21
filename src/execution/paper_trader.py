"""Paper trading system for strategy validation.

Simulates trades without real execution to:
- Validate strategy edge before risking capital
- Track theoretical P&L over time
- Test risk management rules
- Build confidence in system behavior
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Literal
from enum import Enum

from ..core.utils import get_logger
from ..storage.database import Database

logger = get_logger(__name__)


class PositionStatus(Enum):
    """Position lifecycle status."""

    OPEN = "open"
    CLOSED_RESOLVED = "closed_resolved"  # Market resolved
    CLOSED_SOLD = "closed_sold"  # Manually closed


@dataclass
class PaperPosition:
    """A simulated position.

    Attributes:
        id: Internal position ID.
        market_id: Market identifier.
        platform: Platform name.
        question: Market question (for display).
        side: 'YES' or 'NO'.
        entry_price: Price at entry.
        size: Position size in shares.
        cost_basis: Total cost (size * entry_price).
        opened_at: When position was opened.
        status: Current status.
        exit_price: Price at exit (if closed).
        closed_at: When position was closed.
        pnl: Realized P&L (if closed).
        resolution_outcome: Market outcome if resolved.
    """

    id: str
    market_id: str
    platform: str
    question: str
    side: Literal["YES", "NO"]
    entry_price: float
    size: float
    cost_basis: float
    opened_at: datetime
    status: PositionStatus = PositionStatus.OPEN
    exit_price: float | None = None
    closed_at: datetime | None = None
    pnl: float | None = None
    resolution_outcome: str | None = None
    _current_value: float = field(default=0.0, repr=False)

    @property
    def current_value(self) -> float:
        """Current position value (set during mark-to-market)."""
        return self._current_value if self._current_value > 0 else self.cost_basis

    @current_value.setter
    def current_value(self, value: float) -> None:
        self._current_value = value

    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L based on current value."""
        if self.status != PositionStatus.OPEN:
            return 0.0
        return self.current_value - self.cost_basis

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "market_id": self.market_id,
            "platform": self.platform,
            "question": self.question,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "cost_basis": self.cost_basis,
            "opened_at": self.opened_at.isoformat(),
            "status": self.status.value,
            "exit_price": self.exit_price,
            "closed_at": self.closed_at.isoformat() if self.closed_at else None,
            "pnl": self.pnl,
            "resolution_outcome": self.resolution_outcome,
            "current_value": self.current_value,
            "unrealized_pnl": self.unrealized_pnl,
        }


@dataclass
class PaperTrade:
    """Record of a paper trade execution."""

    id: str
    position_id: str
    market_id: str
    side: Literal["YES", "NO"]
    action: Literal["BUY", "SELL"]
    price: float
    size: float
    timestamp: datetime

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "position_id": self.position_id,
            "market_id": self.market_id,
            "side": self.side,
            "action": self.action,
            "price": self.price,
            "size": self.size,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PortfolioSnapshot:
    """Point-in-time portfolio state."""

    timestamp: datetime
    cash_balance: float
    positions_value: float
    total_value: float
    unrealized_pnl: float
    realized_pnl: float
    open_positions: int


@dataclass
class PerformanceReport:
    """Performance metrics for paper trading."""

    start_date: datetime
    end_date: datetime
    starting_capital: float
    ending_capital: float
    total_return: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float  # gross profit / gross loss
    max_drawdown: float
    max_drawdown_pct: float


class PaperTrader:
    """Simulate trades without real execution.

    Usage:
        trader = PaperTrader(initial_balance=1000.0)

        # Execute a paper trade
        trade = await trader.execute_paper_trade(
            market_id="token_123",
            side="YES",
            size=100,
            price=0.65,
            question="Will BTC exceed $100k?"
        )

        # Mark positions to market
        snapshot = await trader.mark_to_market()

        # Close resolved positions
        closed = await trader.close_resolved_positions()

        # Get performance report
        report = trader.get_performance_report()
    """

    def __init__(
        self,
        initial_balance: float = 1000.0,
        database: Database | None = None,
    ):
        """Initialize paper trader.

        Args:
            initial_balance: Starting paper balance in USD.
            database: Optional database for persistence.
        """
        self.initial_balance = initial_balance
        self.cash_balance = initial_balance
        self.db = database

        self.positions: dict[str, PaperPosition] = {}
        self.trade_history: list[PaperTrade] = []
        self.snapshots: list[PortfolioSnapshot] = []

        self._realized_pnl = 0.0

    async def connect_database(self, db_path: str | None = None) -> None:
        """Connect to database for persistence.

        Args:
            db_path: Optional database path.
        """
        if self.db is None:
            self.db = Database(db_path)
            await self.db.connect()

    async def execute_paper_trade(
        self,
        market_id: str,
        side: Literal["YES", "NO"],
        size: float,
        price: float,
        platform: str = "polymarket",
        question: str = "",
    ) -> PaperTrade:
        """Execute a simulated trade.

        Args:
            market_id: Market/token identifier.
            side: 'YES' or 'NO'.
            size: Number of shares to buy.
            price: Price per share.
            platform: Platform name.
            question: Market question (for display).

        Returns:
            PaperTrade record.

        Raises:
            ValueError: If insufficient balance.
        """
        cost = size * price

        if cost > self.cash_balance:
            raise ValueError(
                f"Insufficient balance: ${self.cash_balance:.2f} < ${cost:.2f}"
            )

        # Deduct from balance
        self.cash_balance -= cost

        # Create or update position
        position_key = f"{market_id}_{side}"

        if position_key in self.positions:
            # Add to existing position
            pos = self.positions[position_key]
            new_size = pos.size + size
            new_cost = pos.cost_basis + cost
            pos.size = new_size
            pos.cost_basis = new_cost
            pos.entry_price = new_cost / new_size  # Average price
            position_id = pos.id
        else:
            # New position
            position_id = f"pos_{uuid.uuid4().hex[:8]}"

            pos = PaperPosition(
                id=position_id,
                market_id=market_id,
                platform=platform,
                question=question,
                side=side,
                entry_price=price,
                size=size,
                cost_basis=cost,
                opened_at=datetime.now(timezone.utc),
            )
            self.positions[position_key] = pos

        # Record trade
        trade = PaperTrade(
            id=f"trade_{uuid.uuid4().hex[:8]}",
            position_id=position_id,
            market_id=market_id,
            side=side,
            action="BUY",
            price=price,
            size=size,
            timestamp=datetime.now(timezone.utc),
        )
        self.trade_history.append(trade)

        logger.info(
            "paper_trade_executed",
            action="BUY",
            side=side,
            size=size,
            price=f"{price:.4f}",
            cost=f"{cost:.2f}",
            balance=f"{self.cash_balance:.2f}",
        )

        return trade

    async def mark_to_market(
        self,
        current_prices: dict[str, float] | None = None,
    ) -> PortfolioSnapshot:
        """Calculate current portfolio value at market prices.

        Args:
            current_prices: Dict of market_id -> current YES price.
                If None, uses last known prices.

        Returns:
            PortfolioSnapshot with current values.
        """
        positions_value = 0.0
        unrealized_pnl = 0.0

        for key, pos in self.positions.items():
            if pos.status != PositionStatus.OPEN:
                continue

            # Get current price
            if current_prices and pos.market_id in current_prices:
                if pos.side == "YES":
                    current_price = current_prices[pos.market_id]
                else:
                    current_price = 1.0 - current_prices[pos.market_id]
            else:
                current_price = pos.entry_price  # Use entry if no current

            pos.current_value = pos.size * current_price
            positions_value += pos.current_value
            unrealized_pnl += pos.unrealized_pnl

        snapshot = PortfolioSnapshot(
            timestamp=datetime.now(timezone.utc),
            cash_balance=self.cash_balance,
            positions_value=positions_value,
            total_value=self.cash_balance + positions_value,
            unrealized_pnl=unrealized_pnl,
            realized_pnl=self._realized_pnl,
            open_positions=sum(
                1 for p in self.positions.values() if p.status == PositionStatus.OPEN
            ),
        )

        self.snapshots.append(snapshot)
        return snapshot

    async def close_resolved_positions(
        self,
        resolutions: dict[str, str] | None = None,
    ) -> list[PaperPosition]:
        """Close any positions whose markets have resolved.

        Args:
            resolutions: Dict of market_id -> outcome ('YES' or 'NO').
                If None, fetches from database.

        Returns:
            List of closed positions.
        """
        if resolutions is None and self.db:
            # Fetch from database
            db_resolutions = await self.db.get_resolutions(limit=10000)
            resolutions = {r["market_id"]: r["outcome"] for r in db_resolutions}
        elif resolutions is None:
            resolutions = {}

        closed = []

        for key, pos in list(self.positions.items()):
            if pos.status != PositionStatus.OPEN:
                continue

            if pos.market_id not in resolutions:
                continue

            outcome = resolutions[pos.market_id]
            pos.resolution_outcome = outcome

            # Calculate P&L
            if pos.side == outcome:
                # Won - receive $1 per share
                payout = pos.size * 1.0
            else:
                # Lost - receive $0
                payout = 0.0

            pos.pnl = payout - pos.cost_basis
            pos.exit_price = 1.0 if pos.side == outcome else 0.0
            pos.closed_at = datetime.now(timezone.utc)
            pos.status = PositionStatus.CLOSED_RESOLVED

            # Update balances
            self.cash_balance += payout
            self._realized_pnl += pos.pnl

            closed.append(pos)

            logger.info(
                "paper_position_closed",
                market_id=pos.market_id,
                side=pos.side,
                outcome=outcome,
                pnl=f"{pos.pnl:+.2f}",
            )

        return closed

    def get_performance_report(self) -> PerformanceReport:
        """Generate performance report.

        Returns:
            PerformanceReport with metrics.
        """
        # Separate winning and losing closed trades
        closed_positions = [
            p
            for p in self.positions.values()
            if p.status != PositionStatus.OPEN and p.pnl is not None
        ]

        if not closed_positions:
            current_value = self.cash_balance + sum(
                p.current_value
                for p in self.positions.values()
                if p.status == PositionStatus.OPEN
            )
            return PerformanceReport(
                start_date=datetime.now(timezone.utc),
                end_date=datetime.now(timezone.utc),
                starting_capital=self.initial_balance,
                ending_capital=current_value,
                total_return=current_value - self.initial_balance,
                total_return_pct=(current_value - self.initial_balance)
                / self.initial_balance
                if self.initial_balance > 0
                else 0,
                total_trades=len(self.trade_history),
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                max_drawdown_pct=0.0,
            )

        winners = [p for p in closed_positions if p.pnl > 0]
        losers = [p for p in closed_positions if p.pnl <= 0]

        gross_profit = sum(p.pnl for p in winners) if winners else 0
        gross_loss = abs(sum(p.pnl for p in losers)) if losers else 0

        # Calculate max drawdown from snapshots
        max_drawdown = 0.0
        peak = self.initial_balance

        for snapshot in self.snapshots:
            if snapshot.total_value > peak:
                peak = snapshot.total_value
            drawdown = peak - snapshot.total_value
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        current_value = self.cash_balance + sum(
            p.current_value
            for p in self.positions.values()
            if p.status == PositionStatus.OPEN
        )

        return PerformanceReport(
            start_date=(
                self.trade_history[0].timestamp
                if self.trade_history
                else datetime.now(timezone.utc)
            ),
            end_date=datetime.now(timezone.utc),
            starting_capital=self.initial_balance,
            ending_capital=current_value,
            total_return=current_value - self.initial_balance,
            total_return_pct=(current_value - self.initial_balance)
            / self.initial_balance
            if self.initial_balance > 0
            else 0,
            total_trades=len(self.trade_history),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=len(winners) / len(closed_positions) if closed_positions else 0,
            avg_win=gross_profit / len(winners) if winners else 0,
            avg_loss=gross_loss / len(losers) if losers else 0,
            profit_factor=(
                gross_profit / gross_loss if gross_loss > 0 else float("inf")
            ),
            max_drawdown=max_drawdown,
            max_drawdown_pct=max_drawdown / peak if peak > 0 else 0,
        )

    def get_open_positions(self) -> list[PaperPosition]:
        """Get all open positions."""
        return [p for p in self.positions.values() if p.status == PositionStatus.OPEN]

    def get_position_summary(self) -> dict:
        """Get summary of all positions."""
        open_pos = self.get_open_positions()

        return {
            "cash_balance": self.cash_balance,
            "open_positions": len(open_pos),
            "total_cost_basis": sum(p.cost_basis for p in open_pos),
            "unrealized_pnl": sum(p.unrealized_pnl for p in open_pos),
            "realized_pnl": self._realized_pnl,
            "total_trades": len(self.trade_history),
        }

    def to_dict(self) -> dict:
        """Serialize full state to dictionary."""
        return {
            "initial_balance": self.initial_balance,
            "cash_balance": self.cash_balance,
            "realized_pnl": self._realized_pnl,
            "positions": [p.to_dict() for p in self.positions.values()],
            "trades": [t.to_dict() for t in self.trade_history],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PaperTrader":
        """Deserialize from dictionary."""
        trader = cls(initial_balance=data["initial_balance"])
        trader.cash_balance = data["cash_balance"]
        trader._realized_pnl = data.get("realized_pnl", 0.0)

        # Restore positions
        for p_data in data.get("positions", []):
            pos = PaperPosition(
                id=p_data["id"],
                market_id=p_data["market_id"],
                platform=p_data["platform"],
                question=p_data.get("question", ""),
                side=p_data["side"],
                entry_price=p_data["entry_price"],
                size=p_data["size"],
                cost_basis=p_data["cost_basis"],
                opened_at=datetime.fromisoformat(p_data["opened_at"]),
                status=PositionStatus(p_data["status"]),
                exit_price=p_data.get("exit_price"),
                closed_at=(
                    datetime.fromisoformat(p_data["closed_at"])
                    if p_data.get("closed_at")
                    else None
                ),
                pnl=p_data.get("pnl"),
                resolution_outcome=p_data.get("resolution_outcome"),
            )
            trader.positions[f"{pos.market_id}_{pos.side}"] = pos

        # Restore trades
        for t_data in data.get("trades", []):
            trade = PaperTrade(
                id=t_data["id"],
                position_id=t_data["position_id"],
                market_id=t_data["market_id"],
                side=t_data["side"],
                action=t_data["action"],
                price=t_data["price"],
                size=t_data["size"],
                timestamp=datetime.fromisoformat(t_data["timestamp"]),
            )
            trader.trade_history.append(trade)

        return trader

    def save_to_file(self, path: str) -> None:
        """Save state to JSON file."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Paper trader state saved to {path}")

    @classmethod
    def load_from_file(cls, path: str) -> "PaperTrader":
        """Load state from JSON file."""
        with open(path) as f:
            data = json.load(f)
        logger.info(f"Paper trader state loaded from {path}")
        return cls.from_dict(data)
