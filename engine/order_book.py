"""
Continuous Double Auction (CDA) Matching Engine.

Implements a limit order book with price-time priority matching.
This is the core market mechanism — all agent orders flow through here.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """A single limit order submitted by an agent."""
    agent_id: str
    side: Side
    price: float
    quantity: int
    timestamp: int  # tick number — used for time priority

    def __post_init__(self):
        if self.quantity <= 0:
            raise ValueError(f"Order quantity must be positive, got {self.quantity}")
        if self.price <= 0:
            raise ValueError(f"Order price must be positive, got {self.price}")


@dataclass
class Trade:
    """A completed trade between two orders."""
    tick: int
    price: float
    quantity: int
    buyer_id: str
    seller_id: str
    aggressor_side: Side  # who crossed the spread

    @property
    def value(self) -> float:
        return self.price * self.quantity


class OrderBook:
    """
    Limit order book with price-time priority matching.

    Bids sorted descending (best bid = highest price first).
    Asks sorted ascending (best ask = lowest price first).
    Within same price level, earlier orders match first (FIFO).
    """

    def __init__(self):
        # Lists of Order, kept sorted after each insertion
        self.bids: list[Order] = []  # sorted: highest price first, then earliest timestamp
        self.asks: list[Order] = []  # sorted: lowest price first, then earliest timestamp
        self.trade_log: list[Trade] = []
        self._tick: int = 0

    @property
    def best_bid(self) -> Optional[float]:
        """Highest bid price, or None if no bids."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Lowest ask price, or None if no asks."""
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Midpoint between best bid and best ask, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread, or None if either side is empty."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    def set_tick(self, tick: int):
        """Advance the internal tick clock. Called by the simulation loop."""
        self._tick = tick

    def submit_order(self, order: Order) -> list[Trade]:
        """
        Submit an order to the book. Attempts to match immediately.
        Any unmatched residual rests in the book.

        Returns list of trades executed by this order (possibly empty).
        """
        trades: list[Trade] = []

        if order.side == Side.BUY:
            trades = self._match_buy(order)
        elif order.side == Side.SELL:
            trades = self._match_sell(order)

        self.trade_log.extend(trades)
        return trades

    def _match_buy(self, buy_order: Order) -> list[Trade]:
        """Match an incoming buy order against resting asks."""
        trades: list[Trade] = []
        remaining_qty = buy_order.quantity

        while remaining_qty > 0 and self.asks:
            best_ask_order = self.asks[0]

            # Buy can only match if its price >= best ask price
            if buy_order.price < best_ask_order.price:
                break

            # Determine fill quantity
            fill_qty = min(remaining_qty, best_ask_order.quantity)
            fill_price = best_ask_order.price  # price-time priority: passive order's price

            trade = Trade(
                tick=self._tick,
                price=fill_price,
                quantity=fill_qty,
                buyer_id=buy_order.agent_id,
                seller_id=best_ask_order.agent_id,
                aggressor_side=Side.BUY,
            )
            trades.append(trade)

            remaining_qty -= fill_qty
            best_ask_order.quantity -= fill_qty

            # Remove fully filled ask
            if best_ask_order.quantity == 0:
                self.asks.pop(0)

        # Rest any unfilled portion in the bid book
        if remaining_qty > 0:
            resting_order = Order(
                agent_id=buy_order.agent_id,
                side=Side.BUY,
                price=buy_order.price,
                quantity=remaining_qty,
                timestamp=buy_order.timestamp,
            )
            self._insert_bid(resting_order)

        return trades

    def _match_sell(self, sell_order: Order) -> list[Trade]:
        """Match an incoming sell order against resting bids."""
        trades: list[Trade] = []
        remaining_qty = sell_order.quantity

        while remaining_qty > 0 and self.bids:
            best_bid_order = self.bids[0]

            # Sell can only match if its price <= best bid price
            if sell_order.price > best_bid_order.price:
                break

            # Determine fill quantity
            fill_qty = min(remaining_qty, best_bid_order.quantity)
            fill_price = best_bid_order.price  # passive order's price

            trade = Trade(
                tick=self._tick,
                price=fill_price,
                quantity=fill_qty,
                buyer_id=best_bid_order.agent_id,
                seller_id=sell_order.agent_id,
                aggressor_side=Side.SELL,
            )
            trades.append(trade)

            remaining_qty -= fill_qty
            best_bid_order.quantity -= fill_qty

            # Remove fully filled bid
            if best_bid_order.quantity == 0:
                self.bids.pop(0)

        # Rest any unfilled portion in the ask book
        if remaining_qty > 0:
            resting_order = Order(
                agent_id=sell_order.agent_id,
                side=Side.SELL,
                price=sell_order.price,
                quantity=remaining_qty,
                timestamp=sell_order.timestamp,
            )
            self._insert_ask(resting_order)

        return trades

    def _insert_bid(self, order: Order):
        """Insert a bid order maintaining descending price, ascending timestamp order."""
        import bisect
        # For bids: we want descending price, ascending timestamp.
        # bisect uses < operator, so we use a key that negates price but keeps timestamp positive.
        bisect.insort(self.bids, order, key=lambda x: (-x.price, x.timestamp))

    def _insert_ask(self, order: Order):
        """Insert an ask order maintaining ascending price, ascending timestamp order."""
        import bisect
        # For asks: we want ascending price, ascending timestamp.
        bisect.insort(self.asks, order, key=lambda x: (x.price, x.timestamp))

    def cancel_agent_orders(self, agent_id: str):
        """Remove all resting orders for a given agent. Used between ticks."""
        self.bids = [o for o in self.bids if o.agent_id != agent_id]
        self.asks = [o for o in self.asks if o.agent_id != agent_id]

    def clear_book(self):
        """Remove all resting orders. Used for book reset between experiments."""
        self.bids.clear()
        self.asks.clear()

    def snapshot(self) -> dict:
        """
        Return a snapshot of the current order book state.
        Used by market_state serializer to build the LLM prompt.
        """
        return {
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "spread": self.spread,
            "bid_depth": sum(o.quantity for o in self.bids),
            "ask_depth": sum(o.quantity for o in self.asks),
            "bid_levels": len(self.bids),
            "ask_levels": len(self.asks),
            "last_trade_price": self.trade_log[-1].price if self.trade_log else None,
            "last_trade_qty": self.trade_log[-1].quantity if self.trade_log else None,
            "total_trades": len(self.trade_log),
        }

    def __repr__(self) -> str:
        bb = f"{self.best_bid:.2f}" if self.best_bid else "---"
        ba = f"{self.best_ask:.2f}" if self.best_ask else "---"
        sp = f"{self.spread:.4f}" if self.spread else "---"
        return f"OrderBook(bid={bb}, ask={ba}, spread={sp}, bids={len(self.bids)}, asks={len(self.asks)})"
