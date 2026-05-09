"""
Fundamental Agent — trades based on private fair value estimate.

Charter: Buys below fair value, sells above. Patient, slow to act.
"""

from agents.base_agent import BaseAgent
from inference.prompt_templates import get_fundamental_charter


class FundamentalAgent(BaseAgent):

    def __init__(self, agent_id: str, fair_value: float = 100.0, initial_cash: float = 10_000.0):
        super().__init__(agent_id, initial_cash)
        self.fair_value = fair_value

    def update_fair_value(self, new_fv: float):
        """Called by the simulation engine if the asset's true value drifts."""
        self.fair_value = new_fv

    @property
    def charter(self) -> str:
        # Re-evaluate dynamically in case fair_value changed
        return get_fundamental_charter(self.fair_value)

    @property
    def agent_type(self) -> str:
        return "Fundamental"
