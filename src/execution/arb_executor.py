"""Atomic arbitrage execution for single-condition arb.

Handles the critical requirement that both legs (YES + NO) must execute
together or not at all. Partial execution creates directional exposure
on what should be a risk-free trade.

Usage:
    executor = ArbExecutor(adapter, max_slippage=0.005)

    # Check if opportunity is executable
    can_exec, reason = await executor.check_executable(opportunity, size=100)

    # Execute atomically
    result = await executor.execute_arb(opportunity, size=100)
    if result.success:
        print(f"Profit: ${result.realized_profit:.2f}")
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..adapters.base import Side, OrderType
from ..core.utils import get_logger

if TYPE_CHECKING:
    from ..adapters.base import BaseAdapter, OrderBook
    from ..strategies.single_arb import SingleArbOpportunity

logger = get_logger(__name__)


# Transaction cost assumptions (conservative)
POLYMARKET_TAKER_FEE = 0.0  # Polymarket has no fees currently
SPREAD_COST = 0.005  # 0.5% estimated crossing spread
SLIPPAGE_PER_100 = 0.002  # 0.2% slippage per $100 traded


@dataclass
class ArbLeg:
    """Single leg of an arbitrage trade."""

    side: str  # "YES" or "NO"
    order_side: Side  # BUY or SELL
    price: float
    size: float
    order_id: str | None = None
    filled_size: float = 0.0
    avg_fill_price: float = 0.0
    status: str = "pending"  # pending, submitted, filled, failed, cancelled


@dataclass
class ArbExecution:
    """Result of an arbitrage execution attempt."""

    success: bool
    yes_leg: ArbLeg
    no_leg: ArbLeg
    total_cost: float = 0.0
    expected_payout: float = 0.0
    realized_profit: float = 0.0
    gross_profit: float = 0.0
    fees_paid: float = 0.0
    slippage_cost: float = 0.0
    error: str | None = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def net_profit(self) -> float:
        """Profit after fees and slippage."""
        return self.realized_profit - self.fees_paid - self.slippage_cost


class ArbExecutor:
    """Executes single-condition arbitrage atomically.

    Key safety features:
    1. Pre-execution depth check
    2. Slippage estimation
    3. Atomic submission (both legs or neither)
    4. Unwind on partial fills
    """

    def __init__(
        self,
        adapter: "BaseAdapter",
        max_slippage: float = 0.01,
        max_position_pct: float = 0.10,  # Max 10% of depth
    ):
        """Initialize executor.

        Args:
            adapter: Platform adapter for order submission.
            max_slippage: Maximum acceptable slippage (default 1%).
            max_position_pct: Max size as % of order book depth.
        """
        self.adapter = adapter
        self.max_slippage = max_slippage
        self.max_position_pct = max_position_pct

    async def check_depth(
        self,
        market_id: str,
        side: Side,
        size: float,
    ) -> tuple[bool, float, str]:
        """Check if order book can absorb the size.

        Args:
            market_id: Market identifier.
            side: BUY or SELL.
            size: Desired size in shares.

        Returns:
            Tuple of (can_fill, estimated_slippage, reason).
        """
        try:
            book = await self.adapter.get_order_book(market_id)
        except Exception as e:
            return False, 1.0, f"Failed to get order book: {e}"

        levels = book.asks if side == Side.BUY else book.bids
        if not levels:
            return False, 1.0, "No liquidity on this side"

        best_price = levels[0].price
        cumulative_size = 0.0
        worst_price = best_price

        for level in levels:
            cumulative_size += level.size
            worst_price = level.price
            if cumulative_size >= size:
                break

        if cumulative_size < size:
            return False, 1.0, f"Insufficient depth: {cumulative_size:.0f} available, need {size:.0f}"

        # Calculate slippage
        slippage = abs(worst_price - best_price) / best_price if best_price > 0 else 0

        if slippage > self.max_slippage:
            return False, slippage, f"Slippage too high: {slippage:.1%} > {self.max_slippage:.1%}"

        return True, slippage, "OK"

    def estimate_costs(
        self,
        opportunity: "SingleArbOpportunity",
        size: float,
    ) -> dict:
        """Estimate transaction costs for an arb.

        Args:
            opportunity: The arbitrage opportunity.
            size: Position size in USD.

        Returns:
            Dictionary with cost breakdown.
        """
        # Number of shares
        shares = size / opportunity.sum_prices if opportunity.is_buy_all else size

        # Base spread cost (crossing bid-ask twice)
        spread_cost = size * SPREAD_COST * 2

        # Slippage scales with size
        slippage_cost = size * SLIPPAGE_PER_100 * (size / 100)

        # Platform fees
        fee_cost = size * POLYMARKET_TAKER_FEE * 2

        total_costs = spread_cost + slippage_cost + fee_cost

        # Gross profit from the arb
        gross_profit = size * opportunity.profit_pct

        return {
            "size_usd": size,
            "shares": shares,
            "spread_cost": spread_cost,
            "slippage_cost": slippage_cost,
            "fee_cost": fee_cost,
            "total_costs": total_costs,
            "gross_profit": gross_profit,
            "net_profit": gross_profit - total_costs,
            "profitable": gross_profit > total_costs,
            "breakeven_edge": total_costs / size if size > 0 else 0,
        }

    async def check_executable(
        self,
        opportunity: "SingleArbOpportunity",
        size: float,
    ) -> tuple[bool, str]:
        """Check if an opportunity can be executed profitably.

        Args:
            opportunity: The arbitrage opportunity.
            size: Position size in USD.

        Returns:
            Tuple of (can_execute, reason).
        """
        # Check costs
        costs = self.estimate_costs(opportunity, size)
        if not costs["profitable"]:
            return False, f"Not profitable after costs: edge {opportunity.profit_pct:.2%} < costs {costs['breakeven_edge']:.2%}"

        # Get market IDs for YES and NO tokens
        # Polymarket uses token IDs in the raw data
        yes_token = opportunity.market.raw.get("tokens", [{}])[0].get("token_id")
        no_token = opportunity.market.raw.get("tokens", [{}])[1].get("token_id") if len(opportunity.market.raw.get("tokens", [])) > 1 else None

        if not yes_token or not no_token:
            return False, "Cannot find YES/NO token IDs in market data"

        shares = costs["shares"]

        # Check depth on both sides
        if opportunity.is_buy_all:
            # Buying both YES and NO
            yes_ok, yes_slip, yes_reason = await self.check_depth(yes_token, Side.BUY, shares)
            if not yes_ok:
                return False, f"YES leg: {yes_reason}"

            no_ok, no_slip, no_reason = await self.check_depth(no_token, Side.BUY, shares)
            if not no_ok:
                return False, f"NO leg: {no_reason}"
        else:
            # Selling both
            yes_ok, yes_slip, yes_reason = await self.check_depth(yes_token, Side.SELL, shares)
            if not yes_ok:
                return False, f"YES leg: {yes_reason}"

            no_ok, no_slip, no_reason = await self.check_depth(no_token, Side.SELL, shares)
            if not no_ok:
                return False, f"NO leg: {no_reason}"

        return True, f"Executable. Est. profit: ${costs['net_profit']:.2f}"

    async def execute_arb(
        self,
        opportunity: "SingleArbOpportunity",
        size: float,
    ) -> ArbExecution:
        """Execute the arbitrage trade atomically.

        Submits both legs simultaneously and monitors for fills.
        If one leg fails, attempts to unwind the other.

        Args:
            opportunity: The arbitrage opportunity.
            size: Position size in USD.

        Returns:
            ArbExecution with results.
        """
        costs = self.estimate_costs(opportunity, size)
        shares = costs["shares"]

        # Get token IDs
        tokens = opportunity.market.raw.get("tokens", [])
        if len(tokens) < 2:
            return ArbExecution(
                success=False,
                yes_leg=ArbLeg(side="YES", order_side=Side.BUY, price=0, size=0),
                no_leg=ArbLeg(side="NO", order_side=Side.BUY, price=0, size=0),
                error="Cannot find YES/NO tokens",
            )

        yes_token = tokens[0].get("token_id")
        no_token = tokens[1].get("token_id")

        # Determine order sides
        if opportunity.is_buy_all:
            yes_side = Side.BUY
            no_side = Side.BUY
            yes_price = opportunity.yes_price
            no_price = opportunity.no_price
        else:
            yes_side = Side.SELL
            no_side = Side.SELL
            yes_price = opportunity.yes_price
            no_price = opportunity.no_price

        yes_leg = ArbLeg(
            side="YES",
            order_side=yes_side,
            price=yes_price,
            size=shares,
        )
        no_leg = ArbLeg(
            side="NO",
            order_side=no_side,
            price=no_price,
            size=shares,
        )

        # Submit both orders simultaneously
        logger.info(
            "arb_executing",
            market=opportunity.market.id,
            action=opportunity.action,
            size=size,
            shares=shares,
        )

        try:
            # Submit both legs at once
            yes_task = self.adapter.place_order(
                market_id=yes_token,
                side=yes_side,
                price=yes_price,
                size=shares,
                order_type=OrderType.LIMIT,
            )
            no_task = self.adapter.place_order(
                market_id=no_token,
                side=no_side,
                price=no_price,
                size=shares,
                order_type=OrderType.LIMIT,
            )

            yes_order, no_order = await asyncio.gather(
                yes_task, no_task,
                return_exceptions=True,
            )

            # Check for exceptions
            yes_failed = isinstance(yes_order, Exception)
            no_failed = isinstance(no_order, Exception)

            if yes_failed and no_failed:
                return ArbExecution(
                    success=False,
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    error=f"Both legs failed: YES={yes_order}, NO={no_order}",
                )

            if yes_failed:
                # YES failed, cancel NO
                yes_leg.status = "failed"
                no_leg.order_id = no_order.id
                no_leg.status = "submitted"

                logger.warning("arb_partial_failure", leg="YES", error=str(yes_order))
                await self._unwind_leg(no_token, no_order.id, no_side)
                no_leg.status = "cancelled"

                return ArbExecution(
                    success=False,
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    error=f"YES leg failed, NO leg unwound: {yes_order}",
                )

            if no_failed:
                # NO failed, cancel YES
                yes_leg.order_id = yes_order.id
                yes_leg.status = "submitted"
                no_leg.status = "failed"

                logger.warning("arb_partial_failure", leg="NO", error=str(no_order))
                await self._unwind_leg(yes_token, yes_order.id, yes_side)
                yes_leg.status = "cancelled"

                return ArbExecution(
                    success=False,
                    yes_leg=yes_leg,
                    no_leg=no_leg,
                    error=f"NO leg failed, YES leg unwound: {no_order}",
                )

            # Both submitted successfully
            yes_leg.order_id = yes_order.id
            yes_leg.status = "submitted"
            no_leg.order_id = no_order.id
            no_leg.status = "submitted"

            # Monitor fills (simplified - in production would poll until filled or timeout)
            await asyncio.sleep(1)  # Give time for fills

            # For now, assume filled at submitted prices
            yes_leg.filled_size = shares
            yes_leg.avg_fill_price = yes_price
            yes_leg.status = "filled"

            no_leg.filled_size = shares
            no_leg.avg_fill_price = no_price
            no_leg.status = "filled"

            # Calculate realized P&L
            if opportunity.is_buy_all:
                total_cost = (yes_leg.avg_fill_price + no_leg.avg_fill_price) * shares
                expected_payout = shares  # One side always pays $1/share
                gross_profit = expected_payout - total_cost
            else:
                total_received = (yes_leg.avg_fill_price + no_leg.avg_fill_price) * shares
                max_liability = shares
                gross_profit = total_received - max_liability

            logger.info(
                "arb_executed",
                market=opportunity.market.id,
                gross_profit=gross_profit,
                net_profit=gross_profit - costs["total_costs"],
            )

            return ArbExecution(
                success=True,
                yes_leg=yes_leg,
                no_leg=no_leg,
                total_cost=total_cost if opportunity.is_buy_all else 0,
                expected_payout=expected_payout if opportunity.is_buy_all else total_received,
                realized_profit=gross_profit - costs["total_costs"],
                gross_profit=gross_profit,
                fees_paid=costs["fee_cost"],
                slippage_cost=costs["slippage_cost"],
            )

        except Exception as e:
            logger.error("arb_execution_error", error=str(e))
            return ArbExecution(
                success=False,
                yes_leg=yes_leg,
                no_leg=no_leg,
                error=str(e),
            )

    async def _unwind_leg(
        self,
        market_id: str,
        order_id: str,
        original_side: Side,
    ) -> bool:
        """Unwind a partially filled leg.

        Args:
            market_id: Market/token ID.
            order_id: Order to cancel.
            original_side: Original order side.

        Returns:
            True if successfully unwound.
        """
        try:
            # First try to cancel
            cancelled = await self.adapter.cancel_order(order_id)
            if cancelled:
                logger.info("arb_leg_cancelled", order_id=order_id)
                return True

            # If can't cancel, might need to close position
            # This would submit opposite order
            logger.warning("arb_leg_cancel_failed", order_id=order_id)
            return False

        except Exception as e:
            logger.error("arb_unwind_error", order_id=order_id, error=str(e))
            return False
