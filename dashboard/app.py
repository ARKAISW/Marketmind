"""
MarketMind Streamlit Dashboard.

Live playback of the market simulation.
Lets the user dynamically change agent composition and watch the emergent behavior.
"""

import sys
import os
import time
import pandas as pd
import streamlit as st

# Ensure we can import from the rest of the project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.simulation import SimulationEngine, SimulationConfig
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.fundamental_agent import FundamentalAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader
from dashboard.plots import plot_price_chart, plot_agent_pnl, plot_spread

st.set_page_config(page_title="MarketMind Simulation", layout="wide", page_icon="📈")


def build_agents(n_mom: int, n_mr: int, n_fund: int, n_noise: int, n_mm: int) -> list:
    """Build the agent pool based on slider inputs."""
    agents = []
    for i in range(n_mom):
        agents.append(MomentumAgent(f"momentum_{i+1}"))
    for i in range(n_mr):
        agents.append(MeanReversionAgent(f"meanrev_{i+1}"))
    for i in range(n_fund):
        agents.append(FundamentalAgent(f"fundamental_{i+1}", fair_value=100.0))
    for i in range(n_noise):
        agents.append(NoiseTrader(f"noise_{i+1}"))
    for i in range(n_mm):
        agents.append(MarketMakerAgent(f"marketmaker_{i+1}"))
    return agents


def main():
    st.title("📈 MarketMind: Agent-Based Market Simulation")
    st.markdown("Observe emergent market behavior based on LLM agent composition.")

    # Sidebar: Agent Composition
    st.sidebar.header("⚙️ Agent Composition")
    st.sidebar.markdown("Change the mix of traders to test market stability.")

    n_mom = st.sidebar.slider("Momentum Traders", 0, 10, 2)
    n_mr = st.sidebar.slider("Mean Reversion Traders", 0, 10, 1)
    n_fund = st.sidebar.slider("Fundamental Traders (Anchor)", 0, 10, 1)
    n_noise = st.sidebar.slider("Noise Traders", 0, 10, 1)
    n_mm = st.sidebar.slider("Market Makers", 0, 5, 1)

    st.sidebar.markdown("---")
    num_ticks = st.sidebar.slider("Simulation Ticks", 50, 500, 150, step=50)
    playback_speed = st.sidebar.slider("Playback Speed (ms)", 0, 200, 50, step=10)

    if st.sidebar.button("🚀 Run Simulation", type="primary"):
        run_simulation_and_play(n_mom, n_mr, n_fund, n_noise, n_mm, num_ticks, playback_speed)
    else:
        st.info("Configure your agents in the sidebar and click **Run Simulation**.")


def run_simulation_and_play(n_mom, n_mr, n_fund, n_noise, n_mm, num_ticks, playback_speed):
    # Setup
    agents = build_agents(n_mom, n_mr, n_fund, n_noise, n_mm)
    if not agents:
        st.error("You need at least one agent to run a simulation!")
        return

    config = SimulationConfig(
        num_ticks=num_ticks,
        initial_price=100.0,
        use_llm=False,  # Dashboard uses offline mode for fast iteration
        log_to_csv=False,
    )

    engine = SimulationEngine(agents, config)
    
    with st.spinner(f"Running simulation offline ({num_ticks} ticks)..."):
        engine.run()

    # Pre-extract data for playback
    ticks_data = engine.csv_rows
    pnl_data = engine.agent_pnl_rows

    st.success(f"Simulation generated! Playing back...")

    # Layout for playback
    col1, col2 = st.columns([3, 1])
    
    with col1:
        price_placeholder = st.empty()
        spread_placeholder = st.empty()
    
    with col2:
        regime_placeholder = st.empty()
        st.markdown("### Agent Leaderboard")
        leaderboard_placeholder = st.empty()

    # Data structures for incremental plotting
    curr_ticks = []
    curr_prices = []
    curr_spreads = []
    curr_pnls = {agent.agent_id: [] for agent in agents}

    # Playback Loop
    for tick_idx in range(len(ticks_data)):
        tick_info = ticks_data[tick_idx]
        t = tick_info["tick"]
        
        curr_ticks.append(t)
        curr_prices.append(tick_info["mid_price"] if tick_info["mid_price"] is not None else 100.0)
        curr_spreads.append(tick_info["spread"] if tick_info["spread"] is not None else 0.0)

        # Update PnLs for this tick
        tick_pnl_rows = [row for row in pnl_data if row["tick"] == t]
        for row in tick_pnl_rows:
            curr_pnls[row["agent_id"]].append(row["pnl"])

        # Render charts every N ticks to save Streamlit rendering time (if very fast)
        # or every tick if speed allows.
        price_fig = plot_price_chart(curr_ticks, curr_prices, true_fair_values=[100.0] * len(curr_ticks))
        price_placeholder.plotly_chart(price_fig, use_container_width=True, key=f"p_{t}")

        spread_fig = plot_spread(curr_ticks, curr_spreads)
        spread_placeholder.plotly_chart(spread_fig, use_container_width=True, key=f"s_{t}")

        # Update Regime
        regime = tick_info["regime"]
        color = "green" if regime == "Efficient" else "orange" if regime == "Trending" else "red"
        regime_placeholder.markdown(f"### Market Regime: <span style='color:{color}'>{regime}</span>", unsafe_allow_html=True)

        # Update Leaderboard
        # Sort current agents by their latest PnL
        current_leaderboard = sorted(
            [{"Agent": row["agent_id"], "Type": row["agent_type"], "PnL": f"${row['pnl']:.2f}", "Pos": row["position"]} for row in tick_pnl_rows],
            key=lambda x: float(x["PnL"].replace('$', '')),
            reverse=True
        )
        df_leaderboard = pd.DataFrame(current_leaderboard)
        leaderboard_placeholder.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

        # Pause for animation effect
        if playback_speed > 0:
            time.sleep(playback_speed / 1000.0)


if __name__ == "__main__":
    main()
