"""
MarketMind — Entry Point.

Run a multi-agent market simulation.

Usage:
    python run_simulation.py                    # offline mode, 100 ticks
    python run_simulation.py --ticks 200        # offline, 200 ticks
    python run_simulation.py --llm              # vLLM mode (requires server running)
    python run_simulation.py --llm --url http://host:8000/v1
"""

import argparse
import sys

from engine.simulation import SimulationEngine, SimulationConfig
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.fundamental_agent import FundamentalAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader


def build_default_agents() -> list:
    """
    Default agent composition: the baseline 5-agent mix.
    Per spec Experiment A: 2 momentum + 1 mean-reversion + 1 fundamental + 1 noise.
    Plus 1 market maker for liquidity.
    """
    return [
        MomentumAgent("momentum_1"),
        MomentumAgent("momentum_2"),
        MeanReversionAgent("meanrev_1"),
        FundamentalAgent("fundamental_1", fair_value=100.0),
        MarketMakerAgent("marketmaker_1"),
        NoiseTrader("noise_1"),
    ]


def main():
    parser = argparse.ArgumentParser(description="MarketMind Simulation")
    parser.add_argument("--ticks", type=int, default=100, help="Number of simulation ticks")
    parser.add_argument("--price", type=float, default=100.0, help="Initial price")
    parser.add_argument("--llm", action="store_true", help="Use vLLM inference (requires server)")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1", help="vLLM server URL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--api-key", type=str, default="EMPTY", help="API Key for Hugging Face Serverless or other secured endpoints")
    parser.add_argument("--output", type=str, default="output", help="Output directory for CSVs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()

    config = SimulationConfig(
        num_ticks=args.ticks,
        initial_price=args.price,
        use_llm=args.llm,
        vllm_base_url=args.url,
        vllm_model=args.model,
        vllm_api_key=args.api_key,
        output_dir=args.output,
        seed=args.seed,
    )

    agents = build_default_agents()

    engine = SimulationEngine(agents, config)
    engine.run()


if __name__ == "__main__":
    main()
