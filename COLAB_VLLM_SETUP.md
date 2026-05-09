# Running MarketMind vLLM Backend on Google Colab

If you need to run a full 150-tick simulation for your README data but are hitting rate limits on free-tier APIs (like Groq) or waiting for AMD MI300X credits, you can easily spin up a **free, high-throughput vLLM backend** using Google Colab's T4 or L4 GPUs.

This allows you to bypass all rate limits and process ticks in milliseconds.

## Setup Instructions

1. Open [Google Colab](https://colab.research.google.com/).
2. Create a new Notebook.
3. Go to **Runtime -> Change runtime type** and select **T4 GPU** (or L4 if available).
4. Paste the following code into a single cell and run it:

```python
# 1. Install vLLM and ngrok
!pip install vllm pyngrok -q

# 2. Authenticate with ngrok (Get your free token at https://dashboard.ngrok.com/get-started/your-authtoken)
from pyngrok import ngrok
ngrok.set_auth_token("YOUR_NGROK_AUTH_TOKEN_HERE")

# 3. Open a tunnel to port 8000
public_url = ngrok.connect(8000).public_url
print(f"\\n{'='*50}\\n✅ YOUR INFERENCE BASE URL IS:\\n{public_url}/v1\\n{'='*50}\\n")

# 4. Start the vLLM Server in the background
# We use Qwen2.5-7B-Instruct (quantized) or a smaller model that fits in T4 VRAM (16GB)
import os
os.environ["HUGGING_FACE_HUB_TOKEN"] = "YOUR_HF_READ_TOKEN_HERE" # Only needed for gated models

!python -m vllm.entrypoints.openai.api_server \\
    --model Qwen/Qwen2.5-7B-Instruct \\
    --dtype half \\
    --max-model-len 2048 \\
    --gpu-memory-utilization 0.9 \\
    --port 8000
```

## How to use this with MarketMind

1. Copy the `YOUR INFERENCE BASE URL` that gets printed out (e.g., `https://1234-abcd.ngrok-free.app/v1`).
2. Open your MarketMind Gradio Dashboard.
3. Under **Live LLM Settings**, select the **"Custom"** preset.
4. Paste the ngrok URL into the **Inference Base URL** field.
5. Set Model ID to `Qwen/Qwen2.5-7B-Instruct` (or whatever model you ran).
6. Set API Key to anything (e.g., `EMPTY`), since your local Colab server doesn't require a key.
7. Check **Live LLM Mode** and hit Execute!

Your simulation will now run at maximum speed without any rate limits!
