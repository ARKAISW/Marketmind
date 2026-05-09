"""
Experiment B — Momentum Overload.

Agent composition: 4 momentum + 1 noise. No fundamental anchor, no market maker.
Hypothesis: Price trends away from fair value → bubble formation.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.simulation import SimulationEngine, SimulationConfig
from agents.momentum_agent import MomentumAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader
from experiments.plot_utils import plot_experiment


def run():
    agents = [
        MomentumAgent("momentum_1"),
        MomentumAgent("momentum_2"),
        MomentumAgent("momentum_3"),
        MomentumAgent("momentum_4"),
        MarketMakerAgent("marketmaker_1"),
        NoiseTrader("noise_1"),
    ]

    config = SimulationConfig(
        num_ticks=200,
        initial_price=100.0,
        use_llm=False,
        output_dir="output/experiment_b_momentum",
        seed=42,
    )

    engine = SimulationEngine(agents, config)
    engine.run()

    # Generate plots
    plot_experiment(
        engine,
        title="Experiment B — Momentum Overload (No Anchor)",
        output_dir=config.output_dir,
        fair_value=100.0,
    )

    return engine


if __name__ == "__main__":
    run()
