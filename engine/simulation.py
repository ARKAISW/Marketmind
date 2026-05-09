"""
Simulation Engine — the core tick loop.

Each tick:
1. Build market state string for each agent
2. Dispatch all agents concurrently (LLM or offline fallback)
3. Collect orders from responses
4. Submit orders to the order book (matching happens automatically)
5. Update agent positions and cash from executed trades
6. Record metrics
7. Log to CSV
"""

import asyncio
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

from engine.order_book import OrderBook, Order, Side, Trade
from engine.market_state import market_state_to_string
from engine.metrics import MetricsEngine, TickMetrics
from agents.base_agent import BaseAgent
from agents.market_maker_agent import MarketMakerAgent
from inference.vllm_client import VLLMClient, LLMResponse, parse_llm_output


@dataclass
class SimulationConfig:
    """Configuration for a simulation run."""
    num_ticks: int = 100
    initial_price: float = 100.0
    use_llm: bool = False          # False = offline deterministic mode
    vllm_base_url: str = "http://localhost:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_key: str = "EMPTY"    # For HF serverless or secured endpoints
    output_dir: str = "output"
    seed: int = 42
    log_to_csv: bool = True
    base_volatility: float = 0.005 # Random walk std dev for true fair value per tick
    warmup_ticks: int = 15         # Ticks to run in offline mode before LLM takes over
    enable_seed_liquidity: bool = False  # Turned off by default to allow pure LLM market
    fee_per_trade: float = 0.01          # Transaction cost to prevent wash trading


class SimulationEngine:
    """
    Core simulation loop.

    Orchestrates: agents → LLM/offline → orders → order book → trades → metrics.
    """

    def __init__(self, agents: list[BaseAgent], config: SimulationConfig):
        self.agents = agents
        self.config = config
        self.book = OrderBook()
        self.metrics = MetricsEngine()
        self.price_history: list[float] = [config.initial_price]
        self.true_fair_value: float = config.initial_price
        self.tick = 0

        # LLM client (only initialized if use_llm=True)
        self.llm_client: VLLMClient | None = None
        if config.use_llm:
            self.llm_client = VLLMClient(
                base_url=config.vllm_base_url,
                api_key=config.vllm_api_key,
                model=config.vllm_model,
            )

        # CSV logging
        self.csv_rows: list[dict] = []
        self.trade_rows: list[dict] = []
        self.agent_pnl_rows: list[dict] = []

        # Seed for reproducibility
        random.seed(config.seed)

        # Latency tracking for AMD benchmarking
        self.latencies: list[float] = []

    def run(self):
        """Run the full simulation synchronously."""
        for _ in self.run_generator():
            pass

    def run_generator(self):
        """Run the simulation as a generator, yielding after each tick. Useful for live UIs."""
        print(f"Starting simulation: {self.config.num_ticks} ticks, "
              f"{'LLM' if self.config.use_llm else 'offline'} mode, "
              f"{len(self.agents)} agents")
        print(f"Initial price: {self.config.initial_price}")
        print("-" * 60)

        self._seed_book()

        # Run the async loop synchronously step-by-step to yield
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            for tick in range(1, self.config.num_ticks + 1):
                loop.run_until_complete(self._tick_logic(tick))
                yield tick
        finally:
            loop.close()

    async def _run_async(self):
        """Async tick loop."""
        for tick in range(1, self.config.num_ticks + 1):
            await self._tick_logic(tick)

    async def _tick_logic(self, tick: int):
        """Logic for a single tick."""
        self.tick = tick
        self.book.set_tick(tick)

        # --- Exogenous Volatility (Realism update) ---
        # Drift the true macroeconomic fair value using a geometric random walk
        drift = random.gauss(0, self.config.base_volatility)
        self.true_fair_value *= (1 + drift)
        self.true_fair_value = max(0.01, self.true_fair_value)  # Prevent negative/zero prices

        # Broadcast new macroeconomic reality to agents (only Fundamental agents care)
        for agent in self.agents:
            agent.update_fair_value(self.true_fair_value)

        # 1. Dispatch all agents, collect orders
        orders = await self._dispatch_agents()

        # 2. Process intentional cancellations (agents submit "cancel" in orders)
        # The mandatory wipe is removed to create a true CDA

        # 3. Refresh seed liquidity if enabled
        if self.config.enable_seed_liquidity:
            self.book.cancel_agent_orders("_seed")
            self._refresh_seed_liquidity()

        # 4. Submit orders to the book
        tick_trades: list[Trade] = []
        for order in orders:
            if order.side == "cancel":
                # Special case for explicit cancel
                self.book.cancel_agent_orders(order.agent_id)
            else:
                trades = self.book.submit_order(order)
                tick_trades.extend(trades)

        # 5. Update agent states from trades
        for trade in tick_trades:
            self._apply_trade(trade)

        # 6. Record mid price (always append to keep continuous series)
        mid = self.book.mid_price
        effective_price = mid if mid is not None else self.price_history[-1]
        self.price_history.append(effective_price)
        for agent in self.agents:
            agent.update_price_history(effective_price)

        # 7. Record metrics
        tick_metrics = TickMetrics(
            tick=tick,
            mid_price=mid,
            best_bid=self.book.best_bid,
            best_ask=self.book.best_ask,
            spread=self.book.spread,
            trade_count=len(tick_trades),
            volume=sum(t.quantity for t in tick_trades),
        )
        self.metrics.record_tick(tick_metrics)

        # 8. CSV row
        self._record_csv_row(tick, tick_metrics, tick_trades)

        # Progress
        if tick % 10 == 0 or tick == 1:
            regime = self.metrics.classify_regime()
            price_str = f"{mid:.2f}" if mid else "---"
            spread_str = f"{self.book.spread:.4f}" if self.book.spread else "---"
            print(f"  Tick {tick:4d} | Price: {price_str} | "
                  f"Spread: {spread_str} | Trades: {len(tick_trades)} | "
                  f"Regime: {regime}")

    def _seed_book(self):
        """Place initial orders so the book isn't empty on tick 1."""
        p = self.config.initial_price
        self.book.set_tick(0)
        # Seed bid and ask around initial price
        self.book.submit_order(Order("_seed", Side.BUY, round(p * 0.995, 2), 10, 0))
        self.book.submit_order(Order("_seed", Side.SELL, round(p * 1.005, 2), 10, 0))

    def _refresh_seed_liquidity(self):
        """
        Place thin background liquidity each tick.
        Represents passive external market participants — prevents book
        from fully drying up in experiments without a market maker.
        """
        p = self.price_history[-1]
        self.book.submit_order(Order("_seed", Side.BUY, round(p * 0.993, 2), 3, self.tick))
        self.book.submit_order(Order("_seed", Side.SELL, round(p * 1.007, 2), 3, self.tick))

    async def _dispatch_agents(self) -> list[Order]:
        """
        Get orders from all agents for this tick.
        Uses offline mode during warmup_ticks, then switches to LLM.
        """
        if self.config.use_llm and self.tick > self.config.warmup_ticks:
            print(f"DEBUG: Tick {self.tick} - DISPATCHING LLM AGENTS (Model: {self.config.vllm_model})")
            # Sleep slightly to prevent hammering free tier APIs (like Groq) and hitting immediate 429s
            await asyncio.sleep(1.0)
            return await self._dispatch_llm()
        else:
            mode = "WARMUP" if self.config.use_llm else "OFFLINE"
            if self.tick % 10 == 0 or self.tick == 1:
                print(f"DEBUG: Tick {self.tick} - Dispatching {mode} agents")
            return self._dispatch_offline()

    # ── LLM mode ──────────────────────────────────────────────────

    async def _dispatch_llm(self) -> list[Order]:
        """Dispatch agents via vLLM. Uses asyncio.gather for concurrency."""
        assert self.llm_client is not None

        requests: list[tuple[str, str, str]] = []
        for agent in self.agents:
            state_str = market_state_to_string(
                self.book, agent.agent_id,
                agent.state.position, agent.state.cash,
                agent.price_history,
            )
            # All agents, including MarketMaker, now make a single batched call
            requests.append((agent.agent_id, agent.charter, state_str))

        t0 = time.perf_counter()
        responses = await self.llm_client.batch_infer(requests)
        batch_latency = (time.perf_counter() - t0) * 1000
        self.latencies.append(batch_latency)

        return self._responses_to_orders(responses)

    # ── Offline mode (deterministic fallback) ─────────────────────

    def _dispatch_offline(self) -> list[Order]:
        """
        Deterministic order generation based on agent charter logic.
        No LLM needed — used for local dev and testing.
        """
        orders: list[Order] = []
        # Fall back to last known price when book is empty
        mid = self.book.mid_price or self.price_history[-1]

        for agent in self.agents:
            agent_orders = self._offline_agent_logic(agent, mid)
            orders.extend(agent_orders)

        return orders

    def _offline_agent_logic(self, agent: BaseAgent, mid: float) -> list[Order]:
        """Generate orders using simple heuristics matching each agent's charter."""
        orders: list[Order] = []
        # No more mandatory cancel order at start - allow persistence

        prices = agent.price_history[-10:] if agent.price_history else [mid]

        agent_type = agent.agent_type

        if agent_type == "Momentum":
            if len(prices) >= 3:
                trend = prices[-1] - prices[-3]
                if trend > 0.1:
                    price = round(self.book.best_ask * 1.002 if self.book.best_ask else mid * 1.003, 2)
                    orders.append(Order(agent.agent_id, Side.BUY, price, random.randint(5, 15), self.tick))
                elif trend < -0.1:
                    price = round(self.book.best_bid * 0.998 if self.book.best_bid else mid * 0.997, 2)
                    orders.append(Order(agent.agent_id, Side.SELL, price, random.randint(5, 15), self.tick))

        elif agent_type == "MeanReversion":
            if len(prices) >= 5:
                mean = sum(prices) / len(prices)
                variance = sum((p - mean) ** 2 for p in prices) / len(prices)
                std = math.sqrt(variance) if variance > 0 else 0.01
                z = (mid - mean) / std if std > 0 else 0
                if z > 1.5:
                    price = round(self.book.best_bid * 0.998 if self.book.best_bid else mid * 0.997, 2)
                    orders.append(Order(agent.agent_id, Side.SELL, price, random.randint(5, 10), self.tick))
                elif z < -1.5:
                    price = round(self.book.best_ask * 1.002 if self.book.best_ask else mid * 1.003, 2)
                    orders.append(Order(agent.agent_id, Side.BUY, price, random.randint(5, 10), self.tick))

        elif agent_type == "Fundamental":
            from agents.fundamental_agent import FundamentalAgent
            if isinstance(agent, FundamentalAgent):
                fv = agent.fair_value
                gap = (mid - fv) / fv
                if gap < -0.03:
                    price = round(self.book.best_ask * 1.001 if self.book.best_ask else mid * 1.003, 2)
                    orders.append(Order(agent.agent_id, Side.BUY, price, random.randint(5, 10), self.tick))
                elif gap > 0.03:
                    price = round(self.book.best_bid * 0.999 if self.book.best_bid else mid * 0.997, 2)
                    orders.append(Order(agent.agent_id, Side.SELL, price, random.randint(5, 10), self.tick))

        elif agent_type == "MarketMaker":
            # Market Maker manages its own cancellations to keep a tight spread
            orders.append(Order(agent.agent_id, "cancel", 1.0, 1, self.tick))
            # Always post both sides
            bid_price = round(mid * 0.998, 2) # Tighter spread
            ask_price = round(mid * 1.002, 2) # Tighter spread
            qty = 20
            if abs(agent.state.position) > 20:
                qty = 10  # reduce size when inventory is large
            orders.append(Order(agent.agent_id, Side.BUY, bid_price, qty, self.tick))
            orders.append(Order(agent.agent_id, Side.SELL, ask_price, qty, self.tick))

        elif agent_type == "NoiseTrader":
            # Very aggressive random action to stir the market
            action = random.choice(["buy", "sell"]) # No more 'hold'
            if action == "buy":
                # Buy at a slight premium to hit the ask
                price = round(mid * random.uniform(1.001, 1.006), 2)
                orders.append(Order(agent.agent_id, Side.BUY, price, random.randint(5, 20), self.tick))
            else:
                # Sell at a slight discount to hit the bid
                price = round(mid * random.uniform(0.994, 0.999), 2)
                orders.append(Order(agent.agent_id, Side.SELL, price, random.randint(5, 20), self.tick))

        return orders

    # ── Response → Order conversion ───────────────────────────────

    def _responses_to_orders(self, responses: dict[str, LLMResponse]) -> list[Order]:
        """Convert LLM responses to Order objects."""
        orders: list[Order] = []

        for req_id, resp in responses.items():
            agent_id = req_id

            if resp.action == "hold":
                continue

            items_to_process = []
            if resp.action == "orders" and resp.orders:
                items_to_process.extend(resp.orders)
            else:
                items_to_process.append({"action": resp.action, "price": resp.price, "quantity": resp.quantity})

            for item in items_to_process:
                action = item.get("action")
                if action == "hold":
                    continue

                if action == "cancel":
                    # We pass a dummy order with side="cancel" to signal the loop to cancel this agent's orders
                    orders.append(Order(agent_id=agent_id, side="cancel", price=1.0, quantity=1, timestamp=self.tick))
                    continue

                side = Side.BUY if action == "buy" else Side.SELL
                try:
                    order = Order(
                        agent_id=agent_id,
                        side=side,
                        price=item.get("price"),
                        quantity=item.get("quantity"),
                        timestamp=self.tick,
                    )
                    orders.append(order)
                except ValueError:
                    # Invalid price/quantity — skip
                    continue

        return orders

    # ── Trade application ─────────────────────────────────────────

    def _apply_trade(self, trade: Trade):
        """Update agent states after a trade."""
        buyer = self._find_agent(trade.buyer_id)
        seller = self._find_agent(trade.seller_id)

        if buyer:
            buyer.record_trade(Side.BUY, trade.price, trade.quantity)
            buyer.state.cash -= self.config.fee_per_trade * trade.quantity
        if seller:
            seller.record_trade(Side.SELL, trade.price, trade.quantity)
            seller.state.cash -= self.config.fee_per_trade * trade.quantity

        # Log trade
        self.trade_rows.append({
            "tick": trade.tick,
            "price": trade.price,
            "quantity": trade.quantity,
            "buyer": trade.buyer_id,
            "seller": trade.seller_id,
            "aggressor": trade.aggressor_side.value,
        })

    def _find_agent(self, agent_id: str) -> BaseAgent | None:
        """Find agent by ID. Returns None for seed orders."""
        for agent in self.agents:
            if agent.agent_id == agent_id:
                return agent
        return None  # seed orders have agent_id="_seed"

    # ── CSV logging ───────────────────────────────────────────────

    def _record_csv_row(self, tick: int, metrics: TickMetrics, trades: list[Trade]):
        """Record a row for the tick-level CSV."""
        # Use mid_price if available, else last known price
        current_price = metrics.mid_price if metrics.mid_price else self.price_history[-1]
        
        self.csv_rows.append({
            "tick": tick,
            "price": current_price,
            "mid_price": current_price,
            "best_bid": metrics.best_bid,
            "best_ask": metrics.best_ask,
            "spread": metrics.spread,
            "trade_count": metrics.trade_count,
            "volume": metrics.volume,
            "regime": self.metrics.classify_regime(),
            "true_fair_value": self.true_fair_value,
        })

        # Agent PnL snapshot
        for agent in self.agents:
            self.agent_pnl_rows.append({
                "tick": tick,
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type,
                "position": agent.state.position,
                "cash": round(agent.state.cash, 2),
                "pnl": round(agent.mark_to_market(current_price), 2),
                "trades": agent.state.trades_count,
                "win_rate": agent.win_rate,
            })

    def _write_csvs(self):
        """Write all logged data to CSV files."""
        out = Path(self.config.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self._write_csv(out / "ticks.csv", self.csv_rows)
        self._write_csv(out / "trades.csv", self.trade_rows)
        self._write_csv(out / "agent_pnl.csv", self.agent_pnl_rows)

        print(f"\nCSVs written to {out}/")

    @staticmethod
    def _write_csv(path: Path, rows: list[dict]):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    # ── Summary ───────────────────────────────────────────────────

    def _print_summary(self):
        """Print end-of-simulation summary."""
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)

        summary = self.metrics.summary()
        print(f"  Ticks:         {summary['total_ticks']}")
        print(f"  Total Trades:  {summary['total_trades']}")
        print(f"  Total Volume:  {summary['total_volume']}")
        print(f"  Crash Events:  {summary['crash_events']}")
        print(f"  Final Regime:  {summary['current_regime']}")
        if summary['price_range']:
            lo, hi = summary['price_range']
            print(f"  Price Range:   {lo:.2f} — {hi:.2f}")
        if summary['current_volatility']:
            print(f"  Volatility:    {summary['current_volatility']:.4f}")

        # Agent leaderboard
        current_price = self.price_history[-1] if self.price_history else self.config.initial_price
        print(f"\n  Agent PnL Leaderboard (mark-to-market at {current_price:.2f}):")
        print(f"  {'Agent':<25s} {'Type':<15s} {'Pos':>6s} {'Cash':>10s} {'PnL':>10s} {'Trades':>7s}")
        print(f"  {'-'*25} {'-'*15} {'-'*6} {'-'*10} {'-'*10} {'-'*7}")

        ranked = sorted(self.agents, key=lambda a: a.mark_to_market(current_price), reverse=True)
        for agent in ranked:
            pnl = agent.mark_to_market(current_price)
            print(f"  {agent.agent_id:<25s} {agent.agent_type:<15s} "
                  f"{agent.state.position:>6d} {agent.state.cash:>10.2f} "
                  f"{pnl:>10.2f} {agent.state.trades_count:>7d}")

        # Latency stats (for AMD benchmarking)
        if self.latencies:
            avg_lat = sum(self.latencies) / len(self.latencies)
            print(f"\n  Avg batch latency: {avg_lat:.1f} ms")
            print(f"  Throughput: {len(self.agents) / (avg_lat / 1000):.1f} decisions/sec")
