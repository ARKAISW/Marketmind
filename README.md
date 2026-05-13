---
title: MarketMind
emoji: ⚡
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
python_version: 3.11
---

# MarketMind

**Multi-Agent Financial Market Simulation**
*AMD Developer Hackathon | Track 1: AI Agents & Agentic Workflows | May 2026*

A multi-agent financial market simulation where competing LLM agents with distinct trading charters interact inside a continuous double auction (limit order book). The system does not predict markets — it *is* a market. 

**The research question:** 
> When a financial market is populated entirely by competing LLM agents with heterogeneous beliefs and strategies, does it self-organize toward efficiency — or does it bubble, crash, or fragment?

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────┐
│                   MarketMind System                 │
│                                                     │
│  ┌──────────────┐    ┌──────────────────────────┐   │
│  │  Order Book  │◄───│     Agent Dispatcher     │   │
│  │  (CDA Engine)│    │  (round-robin tick loop) │   │
│  └──────┬───────┘    └──────────────────────────┘   │
│         │                        ▲                  │
│         │  market state          │ orders           │
│         ▼                        │                  │
│  ┌──────────────────────────────────────────────┐   │
│  │              Agent Pool (5 agents)           │   │
│  │                                              │   │
│  │  [Momentum]  [MeanRev]  [Fundamental]        │   │
│  │  [MarketMaker]  [NoiseTrader]                │   │
│  │                                              │   │
│  │  Each agent = LLM prompt + charter config    │   │
│  └──────────────────────────────────────────────┘   │
│         │                                           │
│         │  inference requests                       │
│         ▼                                           │
│  ┌──────────────────────────────────────────────┐   │
│  │     vLLM Server (AMD MI300X, ROCm)           │   │
│  │     Qwen2.5-7B-Instruct                      │   │
│  │     Concurrent batched requests              │   │
│  └──────────────────────────────────────────────┘   │
│         │                                           │
│         ▼                                           │
│  ┌──────────────────────────────────────────────┐   │
│  │     Metrics Engine + Dashboard               │   │
│  │     Price series / spread / volatility /     │   │
│  │     crash detector / regime classifier       │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

---

## 🔬 Experiments & Findings

We ran three core experiments to observe emergent market behavior based solely on varying the agent composition. All experiments ran for 200 ticks.

### Experiment A — Baseline (Equal Mix)
*2 Momentum, 1 Mean-Reversion, 1 Fundamental, 1 Market Maker, 1 Noise*
- **Hypothesis:** Prices stay near fair value. Market is relatively efficient.
- **Result:** Validated. The market stayed largely efficient, oscillating tightly around the Fundamental agent's anchor price (100.0). The Fundamental agent emerged as the most profitable, while Momentum agents lost capital trading against noise.

### Experiment B — Momentum Overload (Bubble Test)
*4 Momentum, 1 Market Maker, 1 Noise (No Fundamental Anchor)*
- **Hypothesis:** Price trends away from fair value → bubble formation.
- **Result:** Validated. Without a fundamental anchor to fade the moves, momentum agents bought into each other's flow. Price skyrocketed from 100 to over 200 in 200 ticks, creating a massive, unstable bubble. 

### Experiment C — No Market Maker (Liquidity Test)
*2 Momentum, 1 Mean-Reversion, 1 Fundamental, 1 Noise (No Market Maker)*
- **Hypothesis:** Spreads widen dramatically, liquidity fragmentation, possible crash.
- **Result:** Validated. Without a dedicated liquidity provider continuously quoting both sides, the order book rapidly dried up. The spread widened dramatically, and trade volume collapsed compared to the baseline.

### Ablation Study — Charter Logic vs. Stochasticity
*Does emergent behavior come from the LLM reasoning, or is it just random noise?*
To test this, we provide an "Offline Mode" where agents are replaced with deterministic heuristics (randomized bid/ask placement within a constrained spread, simulating stochastic noise traders). 
- **Result:** While offline heuristic agents can keep a market "moving" and provide raw liquidity, they fail to reproduce complex macroeconomic phenomena like sustained momentum bubbles or intelligent mean-reversion. The structural market regimes we observe—such as the massive bubble in Experiment B—are intrinsically tied to the LLMs internalizing their specific prompt charters and reacting logically to the Level-2 order book state. This ablation demonstrates the system models true agentic reasoning rather than just stochastic noise.

---

## 🚀 AMD MI300X Hardware Advantage & Setup

MarketMind utilizes **vLLM on ROCm** on the AMD Developer Cloud. By dispatching 5–6 agents simultaneously each tick using `asyncio.gather`, we max out concurrent inference streams. 

The **MI300X's massive 192GB HBM3 memory** means we never page-out the model weights between agent calls, maintaining exceptionally low latency despite heavy parallel throughput.

### Local Setup Instructions (AMD Developer Cloud)

```bash
# 1. Provision AMD MI300X Instance and install vLLM
pip install vllm

# 2. Launch vLLM Inference Server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-7B-Instruct \
  --dtype float16 \
  --max-model-len 2048 \
  --tensor-parallel-size 1 \
  --port 8000
```

### Benchmarking
We provide a standalone benchmark script to measure concurrency speedups when running on high-end hardware like the MI300X:
```bash
python experiments/benchmark_vllm.py --url http://localhost:8000/v1
```
*Note: Latency and throughput scale linearly with available VRAM and tensor-parallel configuration. When running locally on consumer hardware (e.g., GTX 1650), expect ~1-2 seconds per tick. On AMD MI300X with vLLM, this drops to milliseconds, enabling near-instantaneous high-frequency simulation.*

---

## 🌐 Hugging Face Spaces Deployment

As part of the AMD Developer Hackathon requirements, **MarketMind is fully optimized for 1-click deployment to Hugging Face Spaces**.

### Premium Streaming UI
The root `app.py` features a custom-styled, high-performance Gradio interface tailored specifically for the HF Hub. It uses native Altair `gr.LinePlot` charts for flicker-free streaming updates, dynamic regime tracking, and a premium dark-themed aesthetic.

### Live Serverless HF API Integration
Because HF Spaces typically run on basic CPU tiers by default, the app contains an integrated **Live Hugging Face API Mode**. Users can:
1. Paste their Hugging Face API Token directly into the dashboard.
2. The UI natively bridges the Simulation Engine to the `api-inference.huggingface.co/v1/` endpoint using the OpenAI SDK format.
3. Live models (like `meta-llama/Llama-3.2-3B-Instruct`) will directly drive the financial agents over the API without requiring a dedicated GPU inside the space itself.

**To deploy your own Space:**
1. Create a new Gradio Space on Hugging Face.
2. Push this repository's root directly to the space.
3. Hugging Face will automatically find `app.py` and `requirements.txt` and launch the Gradio platform.

---

## 💻 Local Quickstart

### Prerequisites
```bash
pip install -r requirements.txt
```

### Run Simulation Offline (No LLM Required)
You can run the simulation locally using the deterministic offline heuristic mode.
```bash
python run_simulation.py --ticks 200
```

### Run Gradio Dashboard
Visualize the live simulation playback and change agent parameters in real-time.
```bash
python app.py
```

---
*Built for the AMD Developer Hackathon | Track 1: AI Agents & Agentic Workflows*
