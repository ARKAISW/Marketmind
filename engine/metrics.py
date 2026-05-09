"""
Metrics Engine.

Computes price series, spread, volatility, crash detection, and per-agent PnL.
Used by the simulation loop to track market health and by the dashboard for display.
"""

import math
from dataclasses import dataclass, field


@dataclass
class TickMetrics:
    """Metrics snapshot for a single tick."""
    tick: int
    mid_price: float | None
    best_bid: float | None
    best_ask: float | None
    spread: float | None
    trade_count: int
    volume: int  # total units traded this tick


class MetricsEngine:
    """
    Accumulates per-tick metrics and computes derived signals.
    """

    def __init__(self, crash_threshold: float = 0.05, crash_window: int = 5):
        self.tick_history: list[TickMetrics] = []
        self.price_series: list[float] = []  # mid prices over time
        self.crash_threshold = crash_threshold  # >5% drop
        self.crash_window = crash_window         # in 5 ticks
        self.crash_events: list[dict] = []

    def record_tick(self, metrics: TickMetrics):
        """Record metrics for one tick."""
        self.tick_history.append(metrics)
        if metrics.mid_price is not None:
            self.price_series.append(metrics.mid_price)
        self._check_crash()

    def _check_crash(self):
        """Detect crash: >threshold drop over crash_window ticks."""
        if len(self.price_series) < self.crash_window + 1:
            return
        recent = self.price_series[-(self.crash_window + 1):]
        pct_change = (recent[-1] - recent[0]) / recent[0]
        if pct_change < -self.crash_threshold:
            self.crash_events.append({
                "tick": len(self.tick_history),
                "drop_pct": pct_change * 100,
                "from_price": recent[0],
                "to_price": recent[-1],
            })

    def rolling_volatility(self, window: int = 10) -> float | None:
        """Rolling standard deviation of mid prices over last `window` ticks."""
        if len(self.price_series) < window:
            return None
        recent = self.price_series[-window:]
        mean = sum(recent) / len(recent)
        variance = sum((p - mean) ** 2 for p in recent) / len(recent)
        return math.sqrt(variance)

    def rolling_mean(self, window: int = 10) -> float | None:
        """Rolling mean of mid prices."""
        if len(self.price_series) < window:
            return None
        return sum(self.price_series[-window:]) / window

    def classify_regime(self) -> str:
        """
        Simple regime classifier based on current market conditions.
        Returns: "Efficient" | "Trending" | "Volatile" | "Crashed"
        """
        if self.crash_events and self.crash_events[-1]["tick"] >= len(self.tick_history) - 5:
            return "Crashed"

        vol = self.rolling_volatility()
        if vol is None:
            return "Efficient"

        mean = self.rolling_mean()
        if mean is None or mean == 0:
            return "Efficient"

        # Coefficient of variation
        cv = vol / mean

        if cv > 0.02:
            return "Volatile"

        # Check for trending: compare last 10 prices direction
        if len(self.price_series) >= 10:
            recent = self.price_series[-10:]
            ups = sum(1 for i in range(1, len(recent)) if recent[i] > recent[i - 1])
            downs = sum(1 for i in range(1, len(recent)) if recent[i] < recent[i - 1])
            if ups >= 7 or downs >= 7:
                return "Trending"

        return "Efficient"

    def summary(self) -> dict:
        """Summary stats for reporting."""
        return {
            "total_ticks": len(self.tick_history),
            "total_trades": sum(t.trade_count for t in self.tick_history),
            "total_volume": sum(t.volume for t in self.tick_history),
            "crash_events": len(self.crash_events),
            "current_regime": self.classify_regime(),
            "current_volatility": self.rolling_volatility(),
            "price_range": (
                (min(self.price_series), max(self.price_series))
                if self.price_series else None
            ),
        }
