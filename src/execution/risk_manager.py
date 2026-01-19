"""Risk management with position limits and kill switches.

Provides:
- Maximum position size per market
- Maximum total exposure
- Daily loss limits (kill switch)
- Correlation limits
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from typing import TYPE_CHECKING

from ..core.utils import get_logger

if TYPE_CHECKING:
    from .position_tracker import PositionTracker
    from .order_manager import OrderManager

logger = get_logger(__name__)


@dataclass
class RiskLimits:
    """Risk limit configuration.

    Attributes:
        max_position_usd: Max position size per market in USD.
        max_position_pct_liquidity: Max position as % of market liquidity.
        max_total_exposure: Max total exposure across all positions.
        daily_loss_limit: Stop trading if daily loss exceeds this.
        max_positions: Maximum number of concurrent positions.
        max_correlated_positions: Max positions in correlated markets.
    """

    max_position_usd: float = 1000.0
    max_position_pct_liquidity: float = 0.05  # 5% of liquidity
    max_total_exposure: float = 10000.0
    daily_loss_limit: float = 500.0
    max_positions: int = 20
    max_correlated_positions: int = 3


@dataclass
class RiskCheck:
    """Result of a risk check.

    Attributes:
        passed: Whether the check passed.
        reason: Reason for failure (if applicable).
        limit_type: Type of limit that was violated.
        current_value: Current value being checked.
        limit_value: Limit that was exceeded.
    """

    passed: bool
    reason: str = ""
    limit_type: str = ""
    current_value: float = 0.0
    limit_value: float = 0.0


class RiskManager:
    """Manages trading risk with limits and kill switches.

    Usage:
        limits = RiskLimits(
            max_position_usd=500,
            daily_loss_limit=200,
        )
        risk_mgr = RiskManager(limits, position_tracker)

        # Before trading
        check = risk_mgr.check_new_order(
            market_id="token_123",
            size=100,
            price=0.55,
            market_liquidity=50000,
        )
        if not check.passed:
            print(f"Order blocked: {check.reason}")
        else:
            # Submit order
            ...

        # After a loss
        if risk_mgr.is_kill_switch_active():
            await order_manager.cancel_all()
    """

    def __init__(
        self,
        limits: RiskLimits,
        position_tracker: "PositionTracker",
        order_manager: "OrderManager | None" = None,
    ):
        """Initialize risk manager.

        Args:
            limits: Risk limit configuration.
            position_tracker: Position tracker for exposure data.
            order_manager: Optional order manager for auto-cancellation.
        """
        self.limits = limits
        self.positions = position_tracker
        self.orders = order_manager

        self._daily_pnl: float = 0.0
        self._day_start: datetime = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        self._kill_switch_active: bool = False

    def check_new_order(
        self,
        market_id: str,
        size: float,
        price: float,
        market_liquidity: float | None = None,
        category: str | None = None,
    ) -> RiskCheck:
        """Check if a new order passes risk checks.

        Args:
            market_id: Market identifier.
            size: Order size in shares.
            price: Order price.
            market_liquidity: Market liquidity in USD (optional).
            category: Market category for correlation check (optional).

        Returns:
            RiskCheck with pass/fail and reason.
        """
        # Check kill switch
        if self._kill_switch_active:
            return RiskCheck(
                passed=False,
                reason="Kill switch active - daily loss limit exceeded",
                limit_type="kill_switch",
            )

        order_notional = size * price

        # Check position size limit
        if order_notional > self.limits.max_position_usd:
            return RiskCheck(
                passed=False,
                reason=f"Order size ${order_notional:.2f} exceeds max ${self.limits.max_position_usd:.2f}",
                limit_type="max_position_usd",
                current_value=order_notional,
                limit_value=self.limits.max_position_usd,
            )

        # Check liquidity limit
        if market_liquidity and market_liquidity > 0:
            pct_liquidity = order_notional / market_liquidity
            if pct_liquidity > self.limits.max_position_pct_liquidity:
                return RiskCheck(
                    passed=False,
                    reason=f"Order is {pct_liquidity:.1%} of liquidity (max: {self.limits.max_position_pct_liquidity:.1%})",
                    limit_type="max_position_pct_liquidity",
                    current_value=pct_liquidity,
                    limit_value=self.limits.max_position_pct_liquidity,
                )

        # Check total exposure
        current_exposure = self.positions.get_total_exposure()
        new_exposure = current_exposure + order_notional

        if new_exposure > self.limits.max_total_exposure:
            return RiskCheck(
                passed=False,
                reason=f"Total exposure ${new_exposure:.2f} would exceed max ${self.limits.max_total_exposure:.2f}",
                limit_type="max_total_exposure",
                current_value=new_exposure,
                limit_value=self.limits.max_total_exposure,
            )

        # Check max positions
        num_positions = len(self.positions.get_all_positions())
        existing_pos = self.positions.get_positions_by_market(market_id)

        if not existing_pos and num_positions >= self.limits.max_positions:
            return RiskCheck(
                passed=False,
                reason=f"Already at max {self.limits.max_positions} positions",
                limit_type="max_positions",
                current_value=num_positions,
                limit_value=self.limits.max_positions,
            )

        # All checks passed
        return RiskCheck(passed=True)

    def update_daily_pnl(self, pnl: float) -> None:
        """Update daily P&L and check kill switch.

        Args:
            pnl: P&L change to add.
        """
        # Reset if new day
        now = datetime.now(timezone.utc)
        current_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        if current_day > self._day_start:
            self._daily_pnl = 0.0
            self._day_start = current_day
            self._kill_switch_active = False

        # Update P&L
        self._daily_pnl += pnl

        # Check kill switch
        if self._daily_pnl <= -self.limits.daily_loss_limit:
            self._activate_kill_switch()

    def _activate_kill_switch(self) -> None:
        """Activate kill switch due to loss limit."""
        if self._kill_switch_active:
            return

        self._kill_switch_active = True
        logger.warning(
            f"KILL SWITCH ACTIVATED: Daily loss ${abs(self._daily_pnl):.2f} "
            f"exceeds limit ${self.limits.daily_loss_limit:.2f}"
        )

    def is_kill_switch_active(self) -> bool:
        """Check if kill switch is active.

        Returns:
            True if trading should be halted.
        """
        return self._kill_switch_active

    def reset_kill_switch(self) -> None:
        """Manually reset kill switch (use with caution)."""
        self._kill_switch_active = False
        logger.info("Kill switch manually reset")

    def get_status(self) -> dict:
        """Get current risk status.

        Returns:
            Dictionary with risk metrics.
        """
        current_exposure = self.positions.get_total_exposure()
        num_positions = len(self.positions.get_all_positions())

        return {
            "kill_switch_active": self._kill_switch_active,
            "daily_pnl": self._daily_pnl,
            "daily_loss_limit": self.limits.daily_loss_limit,
            "daily_pnl_pct_of_limit": abs(self._daily_pnl) / self.limits.daily_loss_limit if self.limits.daily_loss_limit > 0 else 0,
            "current_exposure": current_exposure,
            "max_exposure": self.limits.max_total_exposure,
            "exposure_pct": current_exposure / self.limits.max_total_exposure if self.limits.max_total_exposure > 0 else 0,
            "num_positions": num_positions,
            "max_positions": self.limits.max_positions,
            "limits": {
                "max_position_usd": self.limits.max_position_usd,
                "max_position_pct_liquidity": self.limits.max_position_pct_liquidity,
                "max_total_exposure": self.limits.max_total_exposure,
                "daily_loss_limit": self.limits.daily_loss_limit,
                "max_positions": self.limits.max_positions,
            },
        }


class PaperTradingRiskManager(RiskManager):
    """Risk manager for paper trading (no real money).

    Provides the same risk checks but with relaxed limits
    and additional logging for testing.
    """

    def __init__(
        self,
        position_tracker: "PositionTracker",
        starting_capital: float = 10000.0,
    ):
        """Initialize paper trading risk manager.

        Args:
            position_tracker: Position tracker.
            starting_capital: Starting paper capital.
        """
        limits = RiskLimits(
            max_position_usd=starting_capital * 0.1,  # 10% max per position
            max_total_exposure=starting_capital * 0.5,  # 50% max total
            daily_loss_limit=starting_capital * 0.1,  # 10% daily loss limit
            max_positions=50,
        )
        super().__init__(limits, position_tracker)

        self.starting_capital = starting_capital
        self.paper_balance = starting_capital

    def check_new_order(
        self,
        market_id: str,
        size: float,
        price: float,
        market_liquidity: float | None = None,
        category: str | None = None,
    ) -> RiskCheck:
        """Check order with paper trading balance."""
        # First run normal checks
        check = super().check_new_order(
            market_id, size, price, market_liquidity, category
        )
        if not check.passed:
            return check

        # Check paper balance
        order_cost = size * price
        if order_cost > self.paper_balance:
            return RiskCheck(
                passed=False,
                reason=f"Insufficient paper balance: ${self.paper_balance:.2f} < ${order_cost:.2f}",
                limit_type="paper_balance",
                current_value=self.paper_balance,
                limit_value=order_cost,
            )

        return RiskCheck(passed=True)

    def update_paper_balance(self, delta: float) -> None:
        """Update paper trading balance.

        Args:
            delta: Amount to add/subtract.
        """
        self.paper_balance += delta
        logger.info(f"Paper balance: ${self.paper_balance:.2f} ({delta:+.2f})")

    def get_status(self) -> dict:
        """Get paper trading status."""
        status = super().get_status()
        status["paper_balance"] = self.paper_balance
        status["paper_pnl"] = self.paper_balance - self.starting_capital
        status["paper_return"] = (self.paper_balance - self.starting_capital) / self.starting_capital
        return status
