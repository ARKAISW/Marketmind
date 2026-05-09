"""
Experiment A — Baseline Run.

Agent composition: 2 momentum + 1 mean-reversion + 1 fundamental + 1 market maker + 1 noise.
Hypothesis: Prices stay near fair value. Market is relatively efficient.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.simulation import SimulationEngine, SimulationConfig
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.fundamental_agent import FundamentalAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader
from experiments.plot_utils import plot_experiment


def run():
    agents = [
        MomentumAgent("momentum_1"),
        MomentumAgent("momentum_2"),
        MeanReversionAgent("meanrev_1"),
        FundamentalAgent("fundamental_1", fair_value=100.0),
        MarketMakerAgent("marketmaker_1"),
        NoiseTrader("noise_1"),
    ]

    config = SimulationConfig(
        num_ticks=200,
        initial_price=100.0,
        use_llm=False,
        output_dir="output/experiment_a_baseline",
        seed=42,
    )

    engine = SimulationEngine(agents, config)
    engine.run()

    # Generate plots
    plot_experiment(
        engine,
        title="Experiment A — Baseline (Equal Mix)",
        output_dir=config.output_dir,
        fair_value=100.0,
    )

    return engine


if __name__ == "__main__":
    run()
