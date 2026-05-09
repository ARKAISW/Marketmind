"""
Momentum Agent — buys into rising prices, sells into falling.

Charter: Short memory window, high turnover, trend-following.
"""

from agents.base_agent import BaseAgent
from inference.prompt_templates import MOMENTUM_CHARTER


class MomentumAgent(BaseAgent):

    @property
    def charter(self) -> str:
        return MOMENTUM_CHARTER

    @property
    def agent_type(self) -> str:
        return "Momentum"
