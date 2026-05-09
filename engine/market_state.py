"""
Market State Serializer.

Converts order book snapshot + agent-specific state into a compact string
that the LLM reads as its "user" message each tick.
"""

from engine.order_book import OrderBook


def market_state_to_string(
    book: OrderBook,
    agent_id: str,
    position: int,
    cash: float,
    price_history: list[float],
) -> str:
    """
    Build the ~150-token market state string that agents receive each tick.

    Includes: best bid, best ask, mid price, last trade price,
    agent's position, agent's cash, last 10 prices.
    """
    snap = book.snapshot()

    bb = f"{snap['best_bid']:.2f}" if snap['best_bid'] is not None else "none"
    ba = f"{snap['best_ask']:.2f}" if snap['best_ask'] is not None else "none"
    mid = f"{snap['mid_price']:.2f}" if snap['mid_price'] is not None else "none"
    spread = f"{snap['spread']:.4f}" if snap['spread'] is not None else "none"
    last_price = f"{snap['last_trade_price']:.2f}" if snap['last_trade_price'] is not None else "none"

    # Last 10 prices, formatted compact
    recent = price_history[-10:] if price_history else []
    price_str = ", ".join(f"{p:.2f}" for p in recent) if recent else "none"

    lines = [
        f"Best Bid: {bb} | Best Ask: {ba} | Mid: {mid} | Spread: {spread}",
        f"Last Trade: {last_price}",
        f"Recent Prices (last {len(recent)}): [{price_str}]",
        f"Your Position: {position} units | Your Cash: {cash:.2f}",
    ]
    return "\n".join(lines)
