"""
Unit tests for the CDA Order Book engine.

Tests per Phase 1 spec:
- Crossing orders execute correctly
- Non-crossing orders rest in book
- Price-time priority
- Partial fills
- Trade log accuracy
"""

import sys
import os

# Add parent to path so we can import marketmind
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.order_book import OrderBook, Order, Side, Trade


def test_non_crossing_orders_rest():
    """Non-crossing orders should rest in the book without matching."""
    book = OrderBook()
    book.set_tick(1)

    # Bid at 99, Ask at 101 — no cross
    bid = Order(agent_id="agent_a", side=Side.BUY, price=99.0, quantity=5, timestamp=1)
    ask = Order(agent_id="agent_b", side=Side.SELL, price=101.0, quantity=5, timestamp=1)

    trades_bid = book.submit_order(bid)
    trades_ask = book.submit_order(ask)

    assert trades_bid == [], "Non-crossing bid should not produce trades"
    assert trades_ask == [], "Non-crossing ask should not produce trades"
    assert book.best_bid == 99.0
    assert book.best_ask == 101.0
    assert book.mid_price == 100.0
    assert book.spread == 2.0
    assert len(book.bids) == 1
    assert len(book.asks) == 1
    print("✓ test_non_crossing_orders_rest")


def test_crossing_orders_execute():
    """When a buy crosses the ask, a trade should execute."""
    book = OrderBook()
    book.set_tick(1)

    # Resting ask at 100
    ask = Order(agent_id="seller", side=Side.SELL, price=100.0, quantity=5, timestamp=1)
    book.submit_order(ask)

    # Incoming buy at 100 — crosses the ask
    buy = Order(agent_id="buyer", side=Side.BUY, price=100.0, quantity=5, timestamp=2)
    trades = book.submit_order(buy)

    assert len(trades) == 1
    t = trades[0]
    assert t.price == 100.0
    assert t.quantity == 5
    assert t.buyer_id == "buyer"
    assert t.seller_id == "seller"
    assert t.aggressor_side == Side.BUY
    # Both sides fully filled — book should be empty
    assert len(book.bids) == 0
    assert len(book.asks) == 0
    print("✓ test_crossing_orders_execute")


def test_partial_fill():
    """A larger buy should partially fill against a smaller ask, with residual resting."""
    book = OrderBook()
    book.set_tick(1)

    # Resting ask: 3 units at 100
    ask = Order(agent_id="seller", side=Side.SELL, price=100.0, quantity=3, timestamp=1)
    book.submit_order(ask)

    # Incoming buy: 5 units at 100 — should fill 3, rest 2
    buy = Order(agent_id="buyer", side=Side.BUY, price=100.0, quantity=5, timestamp=2)
    trades = book.submit_order(buy)

    assert len(trades) == 1
    assert trades[0].quantity == 3
    assert len(book.asks) == 0  # ask fully consumed
    assert len(book.bids) == 1  # residual buy rests
    assert book.bids[0].quantity == 2
    assert book.bids[0].agent_id == "buyer"
    print("✓ test_partial_fill")


def test_price_priority():
    """Best price should match first (highest bid, lowest ask)."""
    book = OrderBook()
    book.set_tick(1)

    # Two bids at different prices
    bid_low = Order(agent_id="bidder_low", side=Side.BUY, price=98.0, quantity=5, timestamp=1)
    bid_high = Order(agent_id="bidder_high", side=Side.BUY, price=100.0, quantity=5, timestamp=2)
    book.submit_order(bid_low)
    book.submit_order(bid_high)

    assert book.best_bid == 100.0, f"Best bid should be 100, got {book.best_bid}"

    # Incoming sell at 99 — should match the 100 bid (better price), not the 98 bid
    sell = Order(agent_id="seller", side=Side.SELL, price=99.0, quantity=3, timestamp=3)
    trades = book.submit_order(sell)

    assert len(trades) == 1
    assert trades[0].buyer_id == "bidder_high"
    assert trades[0].price == 100.0  # fills at passive order's price
    assert trades[0].quantity == 3
    print("✓ test_price_priority")


def test_time_priority():
    """At the same price level, earlier orders should fill first (FIFO)."""
    book = OrderBook()
    book.set_tick(1)

    # Two asks at same price, different timestamps
    ask_early = Order(agent_id="early_seller", side=Side.SELL, price=100.0, quantity=5, timestamp=1)
    ask_late = Order(agent_id="late_seller", side=Side.SELL, price=100.0, quantity=5, timestamp=2)
    book.submit_order(ask_early)
    book.submit_order(ask_late)

    # Buy 3 at 100 — should match the earlier ask
    buy = Order(agent_id="buyer", side=Side.BUY, price=100.0, quantity=3, timestamp=3)
    trades = book.submit_order(buy)

    assert len(trades) == 1
    assert trades[0].seller_id == "early_seller"
    print("✓ test_time_priority")


def test_multi_level_fill():
    """A large aggressive order should sweep through multiple price levels."""
    book = OrderBook()
    book.set_tick(1)

    # Ask book: 3 @ 100, 5 @ 101, 2 @ 102
    book.submit_order(Order("s1", Side.SELL, 100.0, 3, 1))
    book.submit_order(Order("s2", Side.SELL, 101.0, 5, 2))
    book.submit_order(Order("s3", Side.SELL, 102.0, 2, 3))

    # Buy 7 @ 102 — should eat through 100 and 101 levels
    buy = Order(agent_id="buyer", side=Side.BUY, price=102.0, quantity=7, timestamp=4)
    trades = book.submit_order(buy)

    assert len(trades) == 2
    assert trades[0].price == 100.0 and trades[0].quantity == 3  # first level
    assert trades[1].price == 101.0 and trades[1].quantity == 4  # partial second level

    # s2 should have 1 unit remaining at 101
    assert len(book.asks) == 2
    assert book.asks[0].price == 101.0
    assert book.asks[0].quantity == 1
    assert book.asks[1].price == 102.0
    # No residual buy (7 filled: 3 + 4)
    assert len(book.bids) == 0
    print("✓ test_multi_level_fill")


def test_trade_log():
    """Trade log should accumulate all executed trades."""
    book = OrderBook()
    book.set_tick(1)

    book.submit_order(Order("s1", Side.SELL, 100.0, 5, 1))
    book.submit_order(Order("b1", Side.BUY, 100.0, 3, 2))

    book.set_tick(2)
    book.submit_order(Order("s2", Side.SELL, 99.0, 2, 3))
    book.submit_order(Order("b2", Side.BUY, 99.0, 2, 4))

    assert len(book.trade_log) == 2
    assert book.trade_log[0].tick == 1
    assert book.trade_log[1].tick == 2
    print("✓ test_trade_log")


def test_snapshot():
    """Snapshot should return correct book state."""
    book = OrderBook()
    book.set_tick(1)

    book.submit_order(Order("b1", Side.BUY, 99.0, 10, 1))
    book.submit_order(Order("s1", Side.SELL, 101.0, 8, 2))

    snap = book.snapshot()
    assert snap["best_bid"] == 99.0
    assert snap["best_ask"] == 101.0
    assert snap["mid_price"] == 100.0
    assert snap["spread"] == 2.0
    assert snap["bid_depth"] == 10
    assert snap["ask_depth"] == 8
    assert snap["last_trade_price"] is None  # no trades yet
    print("✓ test_snapshot")


def test_cancel_agent_orders():
    """Canceling an agent's orders should remove only their orders."""
    book = OrderBook()
    book.set_tick(1)

    book.submit_order(Order("a1", Side.BUY, 99.0, 5, 1))
    book.submit_order(Order("a2", Side.BUY, 98.0, 5, 2))
    book.submit_order(Order("a1", Side.SELL, 102.0, 5, 3))

    book.cancel_agent_orders("a1")

    assert len(book.bids) == 1
    assert book.bids[0].agent_id == "a2"
    assert len(book.asks) == 0
    print("✓ test_cancel_agent_orders")


def test_self_trade_prevention_not_required():
    """
    Note: The spec doesn't require self-trade prevention.
    Documenting that an agent CAN match against itself (this is fine in a simulation).
    """
    book = OrderBook()
    book.set_tick(1)

    book.submit_order(Order("same_agent", Side.SELL, 100.0, 5, 1))
    trades = book.submit_order(Order("same_agent", Side.BUY, 100.0, 3, 2))

    # Self-trade is allowed in this simulation
    assert len(trades) == 1
    assert trades[0].buyer_id == "same_agent"
    assert trades[0].seller_id == "same_agent"
    print("✓ test_self_trade (allowed in simulation)")


if __name__ == "__main__":
    test_non_crossing_orders_rest()
    test_crossing_orders_execute()
    test_partial_fill()
    test_price_priority()
    test_time_priority()
    test_multi_level_fill()
    test_trade_log()
    test_snapshot()
    test_cancel_agent_orders()
    test_self_trade_prevention_not_required()
    print("\n✅ All order book tests passed.")
