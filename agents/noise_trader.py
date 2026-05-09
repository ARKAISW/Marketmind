"""
Noise Trader — acts on random signals to create market friction.

Charter: Random buy/sell within 1% of mid, small quantities.
"""

from agents.base_agent import BaseAgent
from inference.prompt_templates import NOISE_TRADER_CHARTER


class NoiseTrader(BaseAgent):

    @property
    def charter(self) -> str:
        return NOISE_TRADER_CHARTER

    @property
    def agent_type(self) -> str:
        return "NoiseTrader"
