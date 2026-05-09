"""Quick smoke test for all Phase 2 imports."""
import sys
sys.path.insert(0, ".")

from engine.order_book import OrderBook, Order, Side, Trade
from engine.market_state import market_state_to_string
from engine.metrics import MetricsEngine, TickMetrics
from agents.base_agent import BaseAgent, AgentState
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.fundamental_agent import FundamentalAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader
from inference.prompt_templates import MOMENTUM_CHARTER, get_fundamental_charter
from inference.vllm_client import VLLMClient, parse_llm_output

# Instantiate each agent
agents = [
    MomentumAgent("mom_1"),
    MomentumAgent("mom_2"),
    MeanReversionAgent("mr_1"),
    FundamentalAgent("fund_1", fair_value=100.0),
    MarketMakerAgent("mm_1"),
    NoiseTrader("noise_1"),
]
for a in agents:
    print(f"  {a.agent_type:15s} id={a.agent_id}")

# Verify fundamental charter has fair value
f = FundamentalAgent("f_test", fair_value=42.50)
assert "42.50" in f.charter, f"Fair value not in charter: {f.charter[:80]}"
print("  Fundamental charter injection: OK")

# Test JSON parser
assert parse_llm_output('{"action":"buy","price":99.5,"quantity":3}') is not None
assert parse_llm_output('```json\n{"action":"sell","price":101,"quantity":5}\n```') is not None
assert parse_llm_output("garbage") is None
assert parse_llm_output('{"action":"hold"}')["action"] == "hold"
print("  LLM output parser: OK")

# Test market state serializer
book = OrderBook()
book.set_tick(1)
book.submit_order(Order("b1", Side.BUY, 99.0, 5, 1))
book.submit_order(Order("s1", Side.SELL, 101.0, 5, 2))
state_str = market_state_to_string(book, "m1", 0, 10000.0, [100.0, 99.5, 100.2])
assert "Best Bid: 99.00" in state_str
assert "Best Ask: 101.00" in state_str
assert "Your Position: 0 units" in state_str
print("  Market state serializer: OK")

print("\nAll Phase 2 checks passed.")
