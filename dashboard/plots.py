"""
Plotly charting utilities for the Streamlit dashboard.

Generates interactive charts for the live dashboard playback:
1. Real-time price chart
2. Agent PnL leaderboard
3. Bid-ask spread chart
"""

import plotly.graph_objects as go


def plot_price_chart(ticks: list[int], prices: list[float], true_fair_values: list[float] | None = None) -> go.Figure:
    """Live price series with a dynamic fair value reference line."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ticks, y=prices,
        mode="lines",
        line=dict(color="#2196F3", width=2),
        name="Mid Price"
    ))
    
    # Fair value anchor line (dynamic)
    if true_fair_values:
        fig.add_trace(go.Scatter(
            x=ticks,
            y=true_fair_values,
            mode="lines",
            line=dict(color="#F44336", width=1, dash="dash"),
            name="True Fair Value"
        ))
    
    fig.update_layout(
        title="Live Market Price",
        xaxis_title="Tick",
        yaxis_title="Price",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_agent_pnl(ticks: list[int], agent_pnls: dict[str, list[float]]) -> go.Figure:
    """Agent PnL over time."""
    fig = go.Figure()
    
    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#00BCD4"]
    
    for i, (agent_id, pnls) in enumerate(agent_pnls.items()):
        fig.add_trace(go.Scatter(
            x=ticks, y=pnls,
            mode="lines",
            line=dict(color=colors[i % len(colors)], width=2),
            name=agent_id
        ))
        
    fig.update_layout(
        title="Agent PnL (Mark-to-Market)",
        xaxis_title="Tick",
        yaxis_title="PnL",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig


def plot_spread(ticks: list[int], spreads: list[float]) -> go.Figure:
    """Bid-ask spread over time."""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ticks, y=spreads,
        mode="lines",
        line=dict(color="#FF9800", width=2),
        fill="tozeroy",
        name="Spread"
    ))
    
    fig.update_layout(
        title="Bid-Ask Spread",
        xaxis_title="Tick",
        yaxis_title="Spread",
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return fig
