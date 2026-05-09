"""
Async vLLM inference client.

Wraps the OpenAI-compatible endpoint served by vLLM on AMD MI300X.
All agent calls go through here, batched via asyncio.gather().
"""

import asyncio
import json
import time
from dataclasses import dataclass

import openai


@dataclass
class LLMResponse:
    """Parsed response from the LLM."""
    action: str       # "buy", "sell", "hold", "cancel"
    price: float
    quantity: int
    raw_text: str
    latency_ms: float
    success: bool
    orders: list[dict] = None  # Added for multiple orders


# Default hold response for when LLM returns garbage
HOLD_RESPONSE = LLMResponse(
    action="hold", price=0.0, quantity=0,
    raw_text="fallback_hold", latency_ms=0.0, success=False,
    orders=[]
)


def parse_llm_output(raw: str) -> dict | None:
    """
    Parse the LLM's JSON output. Returns dict or None on failure.
    Handles common LLM failure modes: markdown wrapping, trailing text.
    """
    text = raw.strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (```json and ```)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                data = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                return None
        else:
            return None

    if "orders" in data and isinstance(data["orders"], list):
        parsed_orders = []
        for o in data["orders"]:
            action = o.get("action", "").lower()
            if action not in ("buy", "sell", "hold", "cancel"):
                continue
            if action in ("hold", "cancel"):
                parsed_orders.append({"action": action, "price": 0.0, "quantity": 0})
            else:
                try:
                    price = float(o.get("price", 0))
                    quantity = int(o.get("quantity", 0))
                    if price > 0 and quantity > 0:
                        parsed_orders.append({"action": action, "price": round(price, 2), "quantity": min(quantity, 10)})
                except (ValueError, TypeError):
                    continue
        return {"orders": parsed_orders}

    # Validate required fields
    action = data.get("action", "").lower()
    if action not in ("buy", "sell", "hold", "cancel"):
        return None

    if action in ("hold", "cancel"):
        return {"action": action, "price": 0.0, "quantity": 0}

    try:
        price = float(data.get("price", 0))
        quantity = int(data.get("quantity", 0))
    except (ValueError, TypeError):
        return None

    if price <= 0 or quantity <= 0:
        return None

    # Clamp quantity to spec max
    quantity = min(quantity, 10)

    return {"action": action, "price": round(price, 2), "quantity": quantity}


class VLLMClient:
    """
    Async client for vLLM's OpenAI-compatible API.

    Usage:
        client = VLLMClient(base_url="http://localhost:8000/v1")
        responses = await client.batch_infer(requests)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "EMPTY",
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        max_tokens: int = 64,
        temperature: float = 0.8,
    ):
        self.client = openai.AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    async def infer(self, system_prompt: str, user_message: str) -> LLMResponse:
        """Single inference call. Returns parsed LLMResponse."""
        t0 = time.perf_counter()
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                response_format={"type": "json_object"},
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            raw_text = response.choices[0].message.content or ""
            latency_ms = (time.perf_counter() - t0) * 1000

            parsed = parse_llm_output(raw_text)
            if parsed is None:
                return LLMResponse(
                    action="hold", price=0.0, quantity=0,
                    raw_text=raw_text, latency_ms=latency_ms, success=False,
                    orders=[]
                )

            if "orders" in parsed:
                return LLMResponse(
                    action="orders", price=0.0, quantity=0,
                    raw_text=raw_text, latency_ms=latency_ms, success=True,
                    orders=parsed["orders"]
                )

            return LLMResponse(
                action=parsed["action"],
                price=parsed["price"],
                quantity=parsed["quantity"],
                raw_text=raw_text,
                latency_ms=latency_ms,
                success=True,
                orders=[]
            )

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000
            return LLMResponse(
                action="hold", price=0.0, quantity=0,
                raw_text=f"ERROR: {e}", latency_ms=latency_ms, success=False,
                orders=[]
            )

    async def batch_infer(
        self, requests: list[tuple[str, str, str]]
    ) -> dict[str, LLMResponse]:
        """
        Batch inference for multiple agents concurrently.

        Args:
            requests: list of (agent_id, system_prompt, user_message)

        Returns:
            dict mapping agent_id → LLMResponse
        """
        async def _call(agent_id: str, sys_prompt: str, user_msg: str):
            resp = await self.infer(sys_prompt, user_msg)
            return agent_id, resp

        tasks = [_call(aid, sp, um) for aid, sp, um in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: dict[str, LLMResponse] = {}
        for result in results:
            if isinstance(result, Exception):
                # Shouldn't happen since infer() catches exceptions, but be safe
                continue
            agent_id, response = result
            output[agent_id] = response

        return output
