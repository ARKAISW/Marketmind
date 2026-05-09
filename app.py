"""
MarketMind — Gradio Dashboard
A premium trading terminal UI for multi-agent financial market simulation.
Optimized for Hugging Face Spaces deployment.
"""

import sys
import os
import time
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import gradio as gr
from datetime import datetime

# Ensure imports work from this directory
sys.path.insert(0, os.path.dirname(__file__))

from engine.simulation import SimulationEngine, SimulationConfig
from agents.momentum_agent import MomentumAgent
from agents.mean_reversion_agent import MeanReversionAgent
from agents.fundamental_agent import FundamentalAgent
from agents.market_maker_agent import MarketMakerAgent
from agents.noise_trader import NoiseTrader


# ─── AGENT BUILDER ────────────────────────────────────────────────
def build_agents(n_mom, n_mr, n_fund, n_noise, n_mm):
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


# ─── CHART BUILDERS ───────────────────────────────────────────────

COLORS = {
    "price": "#00d4ff",
    "fair_value": "#ff3366",
    "spread": "#ffaa00",
    "volume": "#7c4dff",
    "bg": "rgba(0,0,0,0)",
    "grid": "rgba(255,255,255,0.04)",
    "text": "#8892b0",
    "agents": ["#00d4ff", "#00ff88", "#ff3366", "#ffaa00", "#7c4dff",
               "#ff6b9d", "#c084fc", "#34d399", "#f87171", "#fbbf24"],
}


def build_main_chart(ticks_data):
    """Build the primary price + volume + spread multi-panel chart."""
    ticks = [r["tick"] for r in ticks_data]
    prices = [r["mid_price"] if r["mid_price"] else 100.0 for r in ticks_data]
    fair_vals = [r.get("true_fair_value", 100.0) for r in ticks_data]
    spreads = [r["spread"] if r["spread"] else 0.0 for r in ticks_data]
    volumes = [r["volume"] for r in ticks_data]
    regimes = [r["regime"] for r in ticks_data]

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=None,
    )

    # Price line
    fig.add_trace(go.Scatter(
        x=ticks, y=prices,
        mode="lines",
        line=dict(color=COLORS["price"], width=2.5),
        name="Market Price",
        fill="tozeroy",
        fillcolor="rgba(0, 212, 255, 0.05)",
    ), row=1, col=1)

    # Fair value
    fig.add_trace(go.Scatter(
        x=ticks, y=fair_vals,
        mode="lines",
        line=dict(color=COLORS["fair_value"], width=1.5, dash="dot"),
        name="Fair Value",
    ), row=1, col=1)

    # Regime color bands
    regime_colors = {"Efficient": "rgba(0,255,136,0.06)",
                     "Trending": "rgba(255,170,0,0.06)",
                     "Volatile": "rgba(255,51,102,0.06)",
                     "Crashed": "rgba(255,0,0,0.10)"}
    prev_regime = regimes[0] if regimes else "Efficient"
    band_start = ticks[0] if ticks else 0
    for i, (t, reg) in enumerate(zip(ticks, regimes)):
        if reg != prev_regime or i == len(ticks) - 1:
            fig.add_vrect(
                x0=band_start, x1=t,
                fillcolor=regime_colors.get(prev_regime, "rgba(0,0,0,0)"),
                layer="below", line_width=0, row=1, col=1,
            )
            band_start = t
            prev_regime = reg

    # Volume bars
    fig.add_trace(go.Bar(
        x=ticks, y=volumes,
        marker_color=COLORS["volume"],
        opacity=0.6,
        name="Volume",
    ), row=2, col=1)

    # Spread
    fig.add_trace(go.Scatter(
        x=ticks, y=spreads,
        mode="lines",
        line=dict(color=COLORS["spread"], width=2),
        fill="tozeroy",
        fillcolor="rgba(255,170,0,0.08)",
        name="Spread",
    ), row=3, col=1)

    # Layout
    # Add minimal range to avoid the "zoomed in on noise" look
    prices_only = [t["price"] for t in ticks_data] if ticks_data else [100.0]
    min_p, max_p = min(prices_only), max(prices_only)
    if max_p - min_p < 1.0:
        center = (max_p + min_p) / 2
        min_p, max_p = center - 0.5, center + 0.5

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="JetBrains Mono, monospace", color=COLORS["text"]),
        height=620,
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        ),
        showlegend=True,
    )
    
    # Force Y axis range on the price plot (row 1)
    fig.update_yaxes(range=[min_p * 0.998, max_p * 1.002], row=1, col=1)
    
    # Hide plotly modebar tools for cleaner UI
    fig.update_layout(modebar_remove=['zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'])

    for row in range(1, 4):
        fig.update_xaxes(
            gridcolor=COLORS["grid"], zeroline=False,
            showticklabels=(row == 3), row=row, col=1,
        )
        fig.update_yaxes(
            gridcolor=COLORS["grid"], zeroline=False,
            row=row, col=1,
        )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Vol", row=2, col=1)
    fig.update_yaxes(title_text="Spread", row=3, col=1)
    fig.update_xaxes(title_text="Tick", row=3, col=1)

    return fig


def build_pnl_chart(pnl_data, agents):
    """Build the agent PnL leaderboard chart."""
    fig = go.Figure()

    agent_ids = [a.agent_id for a in agents]
    for idx, aid in enumerate(agent_ids):
        agent_rows = [r for r in pnl_data if r["agent_id"] == aid]
        ticks = [r["tick"] for r in agent_rows]
        pnls = [r["pnl"] for r in agent_rows]
        fig.add_trace(go.Scatter(
            x=ticks, y=pnls,
            mode="lines",
            line=dict(color=COLORS["agents"][idx % len(COLORS["agents"])], width=2),
            name=aid,
        ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=COLORS["bg"],
        plot_bgcolor=COLORS["bg"],
        font=dict(family="JetBrains Mono, monospace", color=COLORS["text"]),
        height=350,
        margin=dict(l=50, r=20, t=30, b=30),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
        ),
        yaxis_title="PnL ($)",
        xaxis_title="Tick",
        xaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
        yaxis=dict(gridcolor=COLORS["grid"], zeroline=False),
    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.15)", line_width=1)
    fig.update_layout(dragmode=False, hovermode="x unified")
    fig.update_layout(modebar_orientation='h', modebar_remove=['zoom', 'pan', 'select', 'lasso2d', 'zoomIn2d', 'zoomOut2d', 'autoScale2d', 'resetScale2d'])

    return fig


def build_leaderboard(agent_pnl_rows, ticks_data):
    """Create a pandas dataframe for the agent leaderboard with advanced metrics."""
    from engine.metrics import calculate_sharpe_ratio, calculate_max_drawdown, calculate_win_rate
    
    if not agent_pnl_rows:
        return pd.DataFrame()

    # Map of agent_id -> list of PnL values
    pnl_map = {}
    for row in agent_pnl_rows:
        aid = row["agent_id"]
        if aid not in pnl_map:
            pnl_map[aid] = []
        pnl_map[aid].append(row["pnl"])

    leaderboard_data = []
    for aid, pnl_series in pnl_map.items():
        final_pnl = pnl_series[-1]
        sharpe = calculate_sharpe_ratio(pnl_series)
        mdd = calculate_max_drawdown(pnl_series)
        wr = calculate_win_rate(pnl_series)
        
        leaderboard_data.append({
            "Agent ID": aid,
            "Total PnL": f"${final_pnl:,.2f}",
            "Sharpe": f"{sharpe:.2f}",
            "Max DD": f"{mdd:.1%}",
            "Win Rate": f"{wr:.1%}"
        })

    df = pd.DataFrame(leaderboard_data)
    if not df.empty:
        df = df.sort_values(by="Total PnL", ascending=False)
    return df


def build_stats_html(ticks_data, pnl_data, elapsed):
    """Build the live stats panel as HTML."""
    from datetime import datetime
    if not ticks_data:
        return "<p>No data</p>"

    last = ticks_data[-1]
    first_price = ticks_data[0]["mid_price"] or 100.0
    last_price = last["mid_price"] or 100.0
    pct_change = ((last_price - first_price) / first_price) * 100
    total_volume = sum(r["volume"] for r in ticks_data)
    total_trades = sum(r["trade_count"] for r in ticks_data)
    avg_spread = np.mean([r["spread"] for r in ticks_data if r["spread"]]) if ticks_data else 0
    regime = last.get("regime", "Unknown")
    timestamp = datetime.now().strftime("%H:%M:%S")

    regime_colors = {
        "Efficient": "#00ff88",
        "Trending": "#ffaa00",
        "Volatile": "#ff3366",
        "Crashed": "#ff0000",
    }
    rc = regime_colors.get(regime, "#8892b0")

    return f"""
    <div class="stats-container" style="display:grid; grid-template-columns: 1fr 1fr; gap: 12px;">
        <div class="stat-card">
            <div class="stat-label">TOTAL TRADES</div>
            <div class="stat-value">{total_trades}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">VOLUME</div>
            <div class="stat-value">{total_volume:,}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">AVG SPREAD</div>
            <div class="stat-value">{avg_spread:.4f}</div>
        </div>
        <div class="stat-card">
            <div class="stat-label">LAST UPDATE</div>
            <div class="stat-value">{datetime.now().strftime('%H:%M:%S')}</div>
        </div>
    </div>
    """


# ─── SIMULATION RUNNER ────────────────────────────────────────────

def run_simulation(n_mom, n_mr, n_fund, n_noise, n_mm,
                   num_ticks, warmup_ticks, volatility, use_llm, hf_token, hf_model, vllm_url,
                   progress=gr.Progress()):
    """Run the full simulation and return all visualization components."""
    print(f"DEBUG: Starting simulation - LLM: {use_llm}, URL: {vllm_url}")
    
    agents = build_agents(int(n_mom), int(n_mr), int(n_fund), int(n_noise), int(n_mm))
    if not agents:
        raise gr.Error("Add at least one agent to run the simulation.")

    config = SimulationConfig(
        num_ticks=int(num_ticks),
        initial_price=100.0,
        use_llm=use_llm,
        vllm_base_url=vllm_url if vllm_url else "https://api-inference.huggingface.co/v1",
        vllm_model=hf_model if hf_model else "Qwen/Qwen2.5-7B-Instruct",
        log_to_csv=False,
        base_volatility=volatility,
        warmup_ticks=int(warmup_ticks),
        enable_seed_liquidity=True, 
        fee_per_trade=0.01
    )

    engine = SimulationEngine(agents, config)

    if use_llm and hf_token and engine.llm_client:
        import openai
        engine.llm_client.client = openai.AsyncOpenAI(
            base_url=config.vllm_base_url,
            api_key=hf_token,
        )

    try:
        t0 = time.time()
        
        # Ensure output directory exists for CSV generation
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Run the simulation fully without yielding intermediate plots to prevent UI flickering
        print("DEBUG: Executing simulation loop...")
        for tick in engine.run_generator():
            progress(tick / int(num_ticks), desc=f"Simulating market dynamics... {tick}/{num_ticks}")
        
        print(f"DEBUG: Simulation complete in {time.time()-t0:.2f}s")
        
        # Build final results
        ticks_data = engine.csv_rows
        pnl_data = engine.agent_pnl_rows
        
        if not ticks_data:
            raise ValueError("No data produced during simulation.")

        main_chart = build_main_chart(ticks_data)
        pnl_chart = build_pnl_chart(pnl_data, agents)
        leaderboard = build_leaderboard(pnl_data, ticks_data)
        stats_html = build_stats_html(ticks_data, pnl_data, time.time() - t0)
        
        # Create temporary export file
        export_path = "marketmind_simulation.csv"
        pd.DataFrame(ticks_data).to_csv(export_path, index=False)
        
        # After simulation is done, write CSVs
        engine._write_csvs()
        
        return main_chart, pnl_chart, leaderboard, stats_html, export_path
    except Exception as e:
        print(f"CRITICAL ERROR in run_simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        raise gr.Error(f"Simulation Failed: {str(e)}")


# ─── CUSTOM CSS ───────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Global ─────────────────────────────────────── */
html, body {
    background: #0a0b10 !important;
}
.gradio-container {
    max-width: 100% !important;
    font-family: 'Inter', sans-serif !important;
    background: linear-gradient(160deg, #0a0b10 0%, #111827 50%, #0d1117 100%) !important;
    min-height: 100vh;
}
.main {
    background: transparent !important;
}
footer {
    display: none !important;
}

/* ── Top Bar ────────────────────────────────────── */
.title-bar {
    background: linear-gradient(135deg, rgba(0,212,255,0.08), rgba(124,77,255,0.08));
    border: 1px solid rgba(0,212,255,0.12);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.title-bar::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #00d4ff, #7c4dff, #ff3366);
}
.title-bar h1 {
    margin: 0 0 4px 0;
    font-size: 2em;
    font-weight: 800;
    background: linear-gradient(135deg, #00d4ff 0%, #7c4dff 50%, #ff3366 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -1px;
}
.title-bar p {
    margin: 0;
    color: #8892b0;
    font-size: 0.95em;
    max-width: 700px;
}

/* ── Stat Cards ─────────────────────────────────── */
.stat-card {
    background: rgba(17,24,39,0.7);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.stat-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65em;
    color: #5a6785;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.stat-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.3em;
    font-weight: 600;
    color: #e2e8f0;
}
.stat-delta {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85em;
    font-weight: 500;
}

/* ── Panel Sections ─────────────────────────────── */
.panel-header {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75em;
    color: #00d4ff;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 16px 0 8px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(0,212,255,0.15);
}

/* ── Gradio Overrides ───────────────────────────── */
.dark .block {
    background: rgba(17,24,39,0.5) !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-radius: 12px !important;
}
.dark .label-wrap {
    color: #8892b0 !important;
}
.dark input, .dark textarea, .dark select {
    background: rgba(15,20,35,0.8) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
.dark .primary {
    background: linear-gradient(135deg, #00d4ff 0%, #7c4dff 100%) !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0,212,255,0.25) !important;
}
.dark .primary:hover {
    box-shadow: 0 6px 25px rgba(0,212,255,0.4) !important;
    transform: translateY(-1px) !important;
}
.dark table {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85em !important;
}

/* ── Scrollbar ──────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: rgba(0,0,0,0.2); }
::-webkit-scrollbar-thumb { background: rgba(0,212,255,0.3); border-radius: 3px; }
"""


# ─── GRADIO APP ───────────────────────────────────────────────────

def create_app():
    with gr.Blocks(
        title="MarketMind | Multi-Agent Market Simulation",
    ) as app:

        # ── Title Bar ──
        gr.HTML("""
        <div class="title-bar">
            <h1>⚡ MarketMind</h1>
            <p>Multi-agent financial market simulation powered by LLM agents competing inside a 
            continuous double auction. Adjust the agent composition to discover if the market 
            self-organizes to efficiency — or collapses into chaos.</p>
        </div>
        """)

        with gr.Row():
            # ══════════════════════════════════════════════
            # LEFT PANEL — Controls
            # ══════════════════════════════════════════════
            with gr.Column(scale=1, min_width=280):

                gr.HTML('<div class="panel-header">⚙ Engine</div>')
                use_llm = gr.Checkbox(label="Live LLM Mode", value=False,
                                      info="Use HF Serverless API for live inference")
                hf_token = gr.Textbox(label="HF Token", type="password",
                                      placeholder="hf_...", visible=False)
                hf_model = gr.Textbox(label="Model ID", value="Qwen/Qwen2.5-7B-Instruct",
                                      visible=False)
                vllm_url = gr.Textbox(label="Inference Base URL", 
                                      value="https://api-inference.huggingface.co/v1",
                                      placeholder="http://YOUR_AMD_IP:8000/v1",
                                      visible=False)

                use_llm.change(
                    lambda v: (gr.update(visible=v), gr.update(visible=v), gr.update(visible=v)),
                    inputs=[use_llm],
                    outputs=[hf_token, hf_model, vllm_url],
                )

                gr.HTML('<div class="panel-header">🧬 Agent Composition</div>')
                n_mom = gr.Slider(0, 10, value=2, step=1, label="Momentum Traders")
                n_mr = gr.Slider(0, 10, value=1, step=1, label="Mean Reversion")
                n_fund = gr.Slider(0, 10, value=1, step=1, label="Fundamental")
                n_noise = gr.Slider(0, 10, value=1, step=1, label="Noise Traders")
                n_mm = gr.Slider(0, 5, value=1, step=1, label="Market Makers")

                gr.HTML('<div class="panel-header">🔧 Parameters</div>')
                num_ticks = gr.Slider(20, 500, value=150, step=10, label="Simulation Ticks")
                warmup_ticks = gr.Slider(0, 50, value=15, step=5, label="Market Warm-up (Ticks)",
                                         info="Establishing baseline before LLMs take over")
                volatility = gr.Slider(0.0, 0.05, value=0.005, step=0.001,
                                       label="Market Volatility")

                run_btn = gr.Button("▶  Execute Simulation", variant="primary", size="lg")

                # Stats panel (populated after simulation)
                gr.HTML('<div class="panel-header">📊 Session Stats</div>')
                stats_panel = gr.HTML("<p style='color:#5a6785;text-align:center;padding:20px;'>Run a simulation to see stats</p>")
                
                gr.HTML('<div class="panel-header">💾 Export Data</div>')
                export_file = gr.File(label="📥 Download Tick Data (CSV)", interactive=False)

            # ══════════════════════════════════════════════
            # RIGHT PANEL — Charts & Results
            # ══════════════════════════════════════════════
            with gr.Column(scale=3):

                main_chart = gr.Plot(label="Market Overview", elem_classes=["chart-panel"])
                pnl_chart = gr.Plot(label="Agent PnL Tracker")
                leaderboard = gr.DataFrame(
                    label="🏆 Global Performance Metrics",
                    interactive=False,
                    wrap=True,
                )

        # ── Wire up the button ──
        run_btn.click(
            fn=run_simulation,
            inputs=[n_mom, n_mr, n_fund, n_noise, n_mm,
                    num_ticks, warmup_ticks, volatility, use_llm, hf_token, hf_model, vllm_url],
            outputs=[main_chart, pnl_chart, leaderboard, stats_panel, export_file],
        )

    return app


# ─── ENTRY POINT ──────────────────────────────────────────────────

if __name__ == "__main__":
    app = create_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS,
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.cyan,
            secondary_hue=gr.themes.colors.purple,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
            font_mono=gr.themes.GoogleFont("JetBrains Mono"),
        ),
    )
