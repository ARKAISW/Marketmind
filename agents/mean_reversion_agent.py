"""
Mean Reversion Agent — fades moves away from rolling mean.

Charter: Uses z-score thresholds against 10-tick rolling average.
"""

from agents.base_agent import BaseAgent
from inference.prompt_templates import MEAN_REVERSION_CHARTER


class MeanReversionAgent(BaseAgent):

    @property
    def charter(self) -> str:
        return MEAN_REVERSION_CHARTER

    @property
    def agent_type(self) -> str:
        return "MeanReversion"
