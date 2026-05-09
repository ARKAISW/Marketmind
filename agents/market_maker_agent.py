"""
Market Maker Agent — provides liquidity by quoting both sides.

Charter: Posts bid below mid, ask above mid. Manages inventory risk.
Per spec, this agent is called twice per tick (bid + ask).
"""

from agents.base_agent import BaseAgent
from inference.prompt_templates import MARKET_MAKER_CHARTER


class MarketMakerAgent(BaseAgent):

    @property
    def charter(self) -> str:
        return MARKET_MAKER_CHARTER

    @property
    def agent_type(self) -> str:
        return "MarketMaker"
