"""
Plotting utilities for experiment results.

Generates static PNG plots for each experiment:
1. Price series with fair value line
2. Bid-ask spread over time
3. Agent PnL over time
4. Trade volume per tick
"""

import os
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def plot_experiment(engine, title: str, output_dir: str, fair_value: float = 100.0):
    """
    Generate all experiment plots and save to output_dir.
    Falls back to text summary if matplotlib is not installed.
    """
    if not HAS_MPL:
        print(f"[WARN] matplotlib not installed — skipping plots for {title}")
        _text_summary(engine, title, output_dir)
        return

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    _plot_price_series(engine, title, fair_value, out)
    _plot_spread(engine, title, out)
    _plot_agent_pnl(engine, title, out)
    _plot_volume(engine, title, out)

    print(f"Plots saved to {out}/")


def _plot_price_series(engine, title: str, fair_value: float, out: Path):
    """Mid price over time with fair value reference line."""
    ticks = [m.tick for m in engine.metrics.tick_history if m.mid_price is not None]
    prices = [m.mid_price for m in engine.metrics.tick_history if m.mid_price is not None]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(ticks, prices, linewidth=1.5, color="#2196F3", label="Mid Price")
    ax.axhline(y=fair_value, color="#F44336", linestyle="--", linewidth=1, alpha=0.7, label=f"Fair Value ({fair_value})")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Price")
    ax.set_title(f"{title}\nPrice Series")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "price_series.png", dpi=150)
    plt.close(fig)


def _plot_spread(engine, title: str, out: Path):
    """Bid-ask spread over time."""
    ticks = [m.tick for m in engine.metrics.tick_history if m.spread is not None]
    spreads = [m.spread for m in engine.metrics.tick_history if m.spread is not None]

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.fill_between(ticks, spreads, alpha=0.4, color="#FF9800")
    ax.plot(ticks, spreads, linewidth=1, color="#E65100")
    ax.set_xlabel("Tick")
    ax.set_ylabel("Spread")
    ax.set_title(f"{title}\nBid-Ask Spread")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "spread.png", dpi=150)
    plt.close(fig)


def _plot_agent_pnl(engine, title: str, out: Path):
    """Per-agent PnL over time."""
    # Collect PnL per agent per tick
    agent_data: dict[str, list[tuple[int, float]]] = {}

    for row in engine.agent_pnl_rows:
        key = f"{row['agent_id']} ({row['agent_type']})"
        if key not in agent_data:
            agent_data[key] = []
        agent_data[key].append((row["tick"], row["pnl"]))

    colors = ["#2196F3", "#4CAF50", "#F44336", "#FF9800", "#9C27B0", "#00BCD4", "#795548", "#607D8B"]
    fig, ax = plt.subplots(figsize=(12, 5))

    for i, (label, data) in enumerate(agent_data.items()):
        ticks = [d[0] for d in data]
        pnls = [d[1] for d in data]
        color = colors[i % len(colors)]
        ax.plot(ticks, pnls, linewidth=1.2, label=label, color=color)

    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Tick")
    ax.set_ylabel("PnL")
    ax.set_title(f"{title}\nAgent PnL")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "agent_pnl.png", dpi=150)
    plt.close(fig)


def _plot_volume(engine, title: str, out: Path):
    """Trade volume per tick."""
    ticks = [m.tick for m in engine.metrics.tick_history]
    volumes = [m.volume for m in engine.metrics.tick_history]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.bar(ticks, volumes, width=1.0, color="#4CAF50", alpha=0.7)
    ax.set_xlabel("Tick")
    ax.set_ylabel("Volume")
    ax.set_title(f"{title}\nTrade Volume per Tick")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / "volume.png", dpi=150)
    plt.close(fig)


def _text_summary(engine, title: str, output_dir: str):
    """Fallback text summary when matplotlib is unavailable."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary = engine.metrics.summary()
    with open(out / "summary.txt", "w") as f:
        f.write(f"{title}\n{'=' * len(title)}\n\n")
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    print(f"Text summary saved to {out}/summary.txt")
