"""Circuit breakers for automated risk control.

Provides automatic trading halts when adverse conditions are detected:
- Rapid loss detection (multiple losing trades in short window)
- Volatility breaker (prices moving too fast)
- Connection issues (API failures)
- Anomaly detection (unusual market behavior)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Callable, Any

from ..core.utils import get_logger

logger = get_logger(__name__)


class BreakerState(Enum):
    """Circuit breaker state."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Tripped - halted
    HALF_OPEN = "half_open"  # Testing recovery


@dataclass
class BreakerEvent:
    """Record of a breaker-relevant event."""

    timestamp: datetime
    event_type: str  # 'loss', 'error', 'latency', 'price_move'
    value: float
    details: dict = field(default_factory=dict)


@dataclass
class BreakerConfig:
    """Configuration for a circuit breaker.

    Attributes:
        name: Breaker identifier.
        threshold: Trigger threshold (meaning depends on type).
        window_seconds: Time window for event accumulation.
        cooldown_seconds: How long to stay open before half-open.
        half_open_max_trades: Max trades allowed in half-open state.
    """

    name: str
    threshold: float
    window_seconds: int = 300  # 5 minutes default
    cooldown_seconds: int = 600  # 10 minutes default
    half_open_max_trades: int = 3


class CircuitBreaker:
    """Base circuit breaker implementation.

    State machine:
        CLOSED -> OPEN: When threshold exceeded
        OPEN -> HALF_OPEN: After cooldown period
        HALF_OPEN -> CLOSED: If test trades succeed
        HALF_OPEN -> OPEN: If test trades fail

    Usage:
        breaker = CircuitBreaker(config)

        # Before each trade
        if breaker.is_open():
            print("Trading halted")
            return

        # After each trade outcome
        breaker.record_trade(won=True, pnl=5.0)

        # Periodically check for reset
        breaker.check_recovery()
    """

    def __init__(self, config: BreakerConfig):
        """Initialize circuit breaker.

        Args:
            config: Breaker configuration.
        """
        self.config = config
        self.state = BreakerState.CLOSED
        self.events: deque[BreakerEvent] = deque()
        self.tripped_at: datetime | None = None
        self.half_open_trades: int = 0
        self.half_open_failures: int = 0

        self._on_trip_callbacks: list[Callable[[str], Any]] = []
        self._on_reset_callbacks: list[Callable[[str], Any]] = []

    def is_open(self) -> bool:
        """Check if breaker is open (trading halted)."""
        return self.state == BreakerState.OPEN

    def allows_trading(self) -> bool:
        """Check if trading is allowed."""
        if self.state == BreakerState.CLOSED:
            return True
        if self.state == BreakerState.HALF_OPEN:
            return self.half_open_trades < self.config.half_open_max_trades
        return False

    def record_event(
        self,
        event_type: str,
        value: float,
        details: dict | None = None,
    ) -> None:
        """Record an event for threshold checking.

        Args:
            event_type: Type of event.
            value: Event value (for aggregation).
            details: Additional event details.
        """
        event = BreakerEvent(
            timestamp=datetime.now(timezone.utc),
            event_type=event_type,
            value=value,
            details=details or {},
        )
        self.events.append(event)

        # Clean old events
        self._clean_old_events()

    def _clean_old_events(self) -> None:
        """Remove events outside the window."""
        cutoff = datetime.now(timezone.utc) - timedelta(
            seconds=self.config.window_seconds
        )
        while self.events and self.events[0].timestamp < cutoff:
            self.events.popleft()

    def check_threshold(self) -> bool:
        """Check if threshold is exceeded.

        Override in subclasses for specific behavior.

        Returns:
            True if threshold exceeded (should trip).
        """
        return False

    def trip(self, reason: str = "") -> None:
        """Trip the breaker (halt trading).

        Args:
            reason: Reason for tripping.
        """
        if self.state == BreakerState.OPEN:
            return

        self.state = BreakerState.OPEN
        self.tripped_at = datetime.now(timezone.utc)

        logger.warning(
            f"circuit_breaker_tripped",
            breaker=self.config.name,
            reason=reason,
        )

        for callback in self._on_trip_callbacks:
            try:
                callback(self.config.name)
            except Exception as e:
                logger.error(f"Trip callback error: {e}")

    def check_recovery(self) -> None:
        """Check if breaker can recover (OPEN -> HALF_OPEN)."""
        if self.state != BreakerState.OPEN or self.tripped_at is None:
            return

        cooldown_end = self.tripped_at + timedelta(
            seconds=self.config.cooldown_seconds
        )

        if datetime.now(timezone.utc) >= cooldown_end:
            self.state = BreakerState.HALF_OPEN
            self.half_open_trades = 0
            self.half_open_failures = 0
            logger.info(
                f"circuit_breaker_half_open",
                breaker=self.config.name,
            )

    def record_trade_result(self, success: bool) -> None:
        """Record trade result (for half-open state testing).

        Args:
            success: Whether the trade was successful.
        """
        if self.state == BreakerState.HALF_OPEN:
            self.half_open_trades += 1
            if not success:
                self.half_open_failures += 1

            # Check if we should reset or re-trip
            if self.half_open_trades >= self.config.half_open_max_trades:
                if self.half_open_failures == 0:
                    self.reset()
                else:
                    self.trip(
                        f"Failed {self.half_open_failures}/{self.half_open_trades} "
                        f"test trades in half-open state"
                    )

    def reset(self) -> None:
        """Reset breaker to closed state."""
        if self.state == BreakerState.CLOSED:
            return

        self.state = BreakerState.CLOSED
        self.tripped_at = None
        self.events.clear()

        logger.info(f"circuit_breaker_reset", breaker=self.config.name)

        for callback in self._on_reset_callbacks:
            try:
                callback(self.config.name)
            except Exception as e:
                logger.error(f"Reset callback error: {e}")

    def on_trip(self, callback: Callable[[str], Any]) -> None:
        """Register callback for breaker trips."""
        self._on_trip_callbacks.append(callback)

    def on_reset(self, callback: Callable[[str], Any]) -> None:
        """Register callback for breaker resets."""
        self._on_reset_callbacks.append(callback)

    def get_status(self) -> dict:
        """Get breaker status."""
        return {
            "name": self.config.name,
            "state": self.state.value,
            "events_in_window": len(self.events),
            "threshold": self.config.threshold,
            "tripped_at": self.tripped_at.isoformat() if self.tripped_at else None,
            "half_open_trades": self.half_open_trades,
            "half_open_failures": self.half_open_failures,
        }


class LossBreaker(CircuitBreaker):
    """Circuit breaker that trips on cumulative losses.

    Trips when total losses in the window exceed threshold.
    """

    def record_loss(self, loss: float) -> None:
        """Record a trading loss.

        Args:
            loss: Loss amount (positive number).
        """
        self.record_event("loss", abs(loss))

        if self.check_threshold():
            total_loss = sum(e.value for e in self.events if e.event_type == "loss")
            self.trip(f"Cumulative loss ${total_loss:.2f} exceeds ${self.config.threshold:.2f}")

    def check_threshold(self) -> bool:
        """Check if loss threshold exceeded."""
        if self.state == BreakerState.OPEN:
            return False

        total_loss = sum(e.value for e in self.events if e.event_type == "loss")
        return total_loss >= self.config.threshold


class ConsecutiveLossBreaker(CircuitBreaker):
    """Circuit breaker that trips on consecutive losing trades.

    Trips when N consecutive trades are losses.
    """

    def __init__(self, config: BreakerConfig):
        super().__init__(config)
        self.consecutive_losses = 0

    def record_trade(self, won: bool) -> None:
        """Record a trade outcome.

        Args:
            won: Whether the trade was a winner.
        """
        if won:
            self.consecutive_losses = 0
            self.record_trade_result(success=True)
        else:
            self.consecutive_losses += 1
            self.record_event("loss", 1.0)
            self.record_trade_result(success=False)

            if self.consecutive_losses >= self.config.threshold:
                self.trip(f"{self.consecutive_losses} consecutive losses")

    def reset(self) -> None:
        super().reset()
        self.consecutive_losses = 0


class ErrorBreaker(CircuitBreaker):
    """Circuit breaker that trips on API/connection errors.

    Trips when error count in window exceeds threshold.
    """

    def record_error(self, error: str) -> None:
        """Record an error.

        Args:
            error: Error description.
        """
        self.record_event("error", 1.0, {"error": error})

        if self.check_threshold():
            error_count = sum(1 for e in self.events if e.event_type == "error")
            self.trip(f"{error_count} errors in {self.config.window_seconds}s window")

    def check_threshold(self) -> bool:
        """Check if error threshold exceeded."""
        if self.state == BreakerState.OPEN:
            return False

        error_count = sum(1 for e in self.events if e.event_type == "error")
        return error_count >= self.config.threshold


class CircuitBreakerManager:
    """Manages multiple circuit breakers.

    Usage:
        manager = CircuitBreakerManager()

        # Add breakers
        manager.add_loss_breaker(threshold=100, window_seconds=300)
        manager.add_consecutive_loss_breaker(threshold=5)
        manager.add_error_breaker(threshold=10, window_seconds=60)

        # Check if trading allowed
        if not manager.allows_trading():
            print("One or more breakers tripped")
            return

        # Record events
        manager.record_loss(25.0)
        manager.record_error("API timeout")
    """

    def __init__(self):
        """Initialize manager."""
        self.breakers: dict[str, CircuitBreaker] = {}

    def add_breaker(self, breaker: CircuitBreaker) -> None:
        """Add a circuit breaker."""
        self.breakers[breaker.config.name] = breaker

    def add_loss_breaker(
        self,
        threshold: float,
        window_seconds: int = 300,
        cooldown_seconds: int = 600,
    ) -> LossBreaker:
        """Add a loss-based breaker."""
        breaker = LossBreaker(
            BreakerConfig(
                name="loss",
                threshold=threshold,
                window_seconds=window_seconds,
                cooldown_seconds=cooldown_seconds,
            )
        )
        self.add_breaker(breaker)
        return breaker

    def add_consecutive_loss_breaker(
        self,
        threshold: int,
        cooldown_seconds: int = 600,
    ) -> ConsecutiveLossBreaker:
        """Add a consecutive-loss breaker."""
        breaker = ConsecutiveLossBreaker(
            BreakerConfig(
                name="consecutive_loss",
                threshold=threshold,
                window_seconds=86400,  # 24h (consecutive, not time-based)
                cooldown_seconds=cooldown_seconds,
            )
        )
        self.add_breaker(breaker)
        return breaker

    def add_error_breaker(
        self,
        threshold: int,
        window_seconds: int = 60,
        cooldown_seconds: int = 300,
    ) -> ErrorBreaker:
        """Add an error-based breaker."""
        breaker = ErrorBreaker(
            BreakerConfig(
                name="error",
                threshold=threshold,
                window_seconds=window_seconds,
                cooldown_seconds=cooldown_seconds,
            )
        )
        self.add_breaker(breaker)
        return breaker

    def allows_trading(self) -> bool:
        """Check if all breakers allow trading."""
        for breaker in self.breakers.values():
            breaker.check_recovery()  # Check for state transitions
            if not breaker.allows_trading():
                return False
        return True

    def get_open_breakers(self) -> list[str]:
        """Get names of tripped breakers."""
        return [
            name
            for name, breaker in self.breakers.items()
            if breaker.is_open()
        ]

    def record_loss(self, loss: float) -> None:
        """Record a loss to relevant breakers."""
        if "loss" in self.breakers:
            self.breakers["loss"].record_loss(loss)

    def record_trade(self, won: bool) -> None:
        """Record a trade outcome to relevant breakers."""
        if "consecutive_loss" in self.breakers:
            self.breakers["consecutive_loss"].record_trade(won)

    def record_error(self, error: str) -> None:
        """Record an error to relevant breakers."""
        if "error" in self.breakers:
            self.breakers["error"].record_error(error)

    def reset_all(self) -> None:
        """Reset all breakers."""
        for breaker in self.breakers.values():
            breaker.reset()

    def get_status(self) -> dict:
        """Get status of all breakers."""
        return {
            "allows_trading": self.allows_trading(),
            "open_breakers": self.get_open_breakers(),
            "breakers": {
                name: breaker.get_status()
                for name, breaker in self.breakers.items()
            },
        }
