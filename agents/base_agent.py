"""
Base Agent — abstract class for all trading agents.

Each agent holds a charter (system prompt), position state, and cash.
On each tick, the simulation calls agent.observe() with market state
and the agent returns an Order (or None for hold).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from engine.order_book import Order, Side


@dataclass
class AgentState:
    """Mutable state tracked per agent across ticks."""
    position: int = 0        # net units held (positive = long, negative = short)
    cash: float = 10_000.0   # starting cash
    total_pnl: float = 0.0   # realized PnL
    trades_count: int = 0


class BaseAgent(ABC):
    """
    Abstract trading agent.

    Subclasses must implement:
        - charter: property returning the system prompt string
        - agent_type: property returning a human-readable type name
    """

    def __init__(self, agent_id: str, initial_cash: float = 10_000.0):
        self.agent_id = agent_id
        self.state = AgentState(cash=initial_cash)
        self.price_history: list[float] = []

    @property
    @abstractmethod
    def charter(self) -> str:
        """System prompt defining this agent's trading strategy."""
        ...

    @property
    @abstractmethod
    def agent_type(self) -> str:
        """Human-readable agent type name (e.g., 'Momentum')."""
        ...

    def update_price_history(self, price: float):
        """Called each tick to track price history for the agent's context window."""
        self.price_history.append(price)

    def update_fair_value(self, new_fv: float):
        """
        Called each tick by the simulation to broadcast true macroeconomic value drifts.
        Most agents ignore this (it's private info), but Fundamental agents use it.
        """
        pass

    def record_trade(self, side: Side, price: float, quantity: int):
        """
        Update agent state after a trade execution.
        Called by the simulation loop when a trade involves this agent.
        """
        if side == Side.BUY:
            self.state.position += quantity
            self.state.cash -= price * quantity
        elif side == Side.SELL:
            self.state.position -= quantity
            self.state.cash += price * quantity
        self.state.trades_count += 1

    def mark_to_market(self, current_price: float) -> float:
        """Calculate total PnL: cash + position value - initial cash."""
        position_value = self.state.position * current_price
        return self.state.cash + position_value - 10_000.0

    def __repr__(self) -> str:
        return (
            f"{self.agent_type}(id={self.agent_id}, "
            f"pos={self.state.position}, cash={self.state.cash:.2f})"
        )
