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
    profitable_trades: int = 0
    # To accurately track profitable trades, we need to know the entry price of positions.
    # For a hackathon, a simplified approximation is marking the trade against the current mid price.
    
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
        
        # Simplified win rate logic for the hackathon: 
        # A trade is considered "profitable" if it improves Mark-to-Market PnL against the last known price.
        # This is a heuristic.
        last_price = self.price_history[-1] if self.price_history else 100.0
        if side == Side.BUY and price < last_price:
            self.state.profitable_trades += 1
        elif side == Side.SELL and price > last_price:
            self.state.profitable_trades += 1

    def mark_to_market(self, current_price: float) -> float:
        """Calculate total PnL: cash + position value - initial cash."""
        position_value = self.state.position * current_price
        return self.state.cash + position_value - 10_000.0

    @property
    def win_rate(self) -> float:
        """Returns the percentage of profitable trades."""
        if self.state.trades_count == 0:
            return 0.0
        return (self.state.profitable_trades / self.state.trades_count) * 100.0

    def __repr__(self) -> str:
        return (
            f"{self.agent_type}(id={self.agent_id}, "
            f"pos={self.state.position}, cash={self.state.cash:.2f}, WR={self.win_rate:.1f}%)"
        )
