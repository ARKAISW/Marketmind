"""
Metrics Engine.

Computes price series, spread, volatility, crash detection, and per-agent PnL.
Used by the simulation loop to track market health and by the dashboard for display.
"""

import math
import numpy as np
from dataclasses import dataclass, field


def calculate_sharpe_ratio(pnl_series: list[float], risk_free_rate: float = 0.0) -> float:
    """Calculate the Sharpe Ratio of a PnL series."""
    if len(pnl_series) < 5:
        return 0.0
    returns = np.diff(pnl_series)
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return float(np.mean(returns - risk_free_rate) / np.std(returns) * np.sqrt(252)) # Annualized


def calculate_max_drawdown(pnl_series: list[float]) -> float:
    """Calculate the maximum drawdown percentage from a PnL series."""
    if not pnl_series:
        return 0.0
    # Start with initial capital + pnl
    capital = 10_000.0
    equity = [capital + p for p in pnl_series]
    peak = equity[0]
    max_dd = 0.0
    for value in equity:
        if value > peak:
            peak = value
        dd = (peak - value) / peak if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def calculate_win_rate(pnl_series: list[float]) -> float:
    """Calculate the percentage of profitable ticks."""
    if len(pnl_series) < 2:
        return 0.0
    returns = np.diff(pnl_series)
    wins = len([r for r in returns if r > 0])
    return float(wins / len(returns)) if len(returns) > 0 else 0.0


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
