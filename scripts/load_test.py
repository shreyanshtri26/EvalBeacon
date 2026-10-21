"""
load_test.py
============
Generates realistic telemetry by simulating LLM API traffic patterns.

Principal SRE Note:
    Use this to:
    1. Validate the observability stack is working end-to-end
    2. Pre-populate dashboards with meaningful data
    3. Stress test the OTel Collector pipeline

Usage:
    python scripts/load_test.py --rps 10 --duration 60
    python scripts/load_test.py --scenario spike  # 10x traffic spike for 30s
"""

import argparse
import asyncio
import random
import time

import httpx

BASE_URL = "http://localhost:8000/api/v1"

SAMPLE_MESSAGES = [
    "Explain the concept of SLOs in 3 sentences.",
    "What is OpenTelemetry and why should I use it?",
    "How do distributed traces work?",
    "Explain the four golden signals of monitoring.",
    "What is the difference between metrics, logs, and traces?",
    "Describe a runbook for high LLM latency.",
    "What is tail sampling in OpenTelemetry?",
    "How do I calculate error budget burn rate?",
    "Explain exponential backoff and jitter.",
    "What are the DORA metrics?",
]

MODELS = [
    "gpt-4o-mini",
    "gpt-4o",
]


async def send_chat_request(client: httpx.AsyncClient, scenario: str = "normal") -> dict:
    """Send a single chat request and return timing info."""
    message = random.choice(SAMPLE_MESSAGES)
    model = random.choice(MODELS)

    # Simulate different traffic patterns
    if scenario == "spike":
        max_tokens = random.randint(512, 2048)
    elif scenario == "error":
        model = "invalid-model-xyz"  # Will cause API error
        max_tokens = 100
    else:
        max_tokens = random.randint(100, 512)

    start = time.perf_counter()
    try:
        response = await client.post(
            f"{BASE_URL}/chat",
            json={
                "message": message,
                "model": model,
                "max_tokens": max_tokens,
            },
            timeout=30.0,
        )
        duration = time.perf_counter() - start

        if response.status_code == 200:
            data = response.json()
            return {
                "status": "success",
                "duration": duration,
                "model": data.get("model"),
                "input_tokens": data.get("input_tokens"),
                "output_tokens": data.get("output_tokens"),
                "cost": data.get("estimated_cost_usd"),
            }
        else:
            return {"status": "error", "code": response.status_code, "duration": duration}

    except Exception as exc:
        return {"status": "exception", "error": str(exc), "duration": time.perf_counter() - start}


async def run_load_test(
    rps: float = 5.0,
    duration: int = 60,
    scenario: str = "normal",
) -> None:
    """
    Run a load test generating `rps` requests per second for `duration` seconds.
    """
    print(f"\n🚀 Starting load test: {rps} RPS, {duration}s, scenario={scenario}")
    print(f"   Target: {BASE_URL}\n")

    interval = 1.0 / rps
    end_time = time.time() + duration
    results = []

    async with httpx.AsyncClient() as client:
        while time.time() < end_time:
            task = asyncio.create_task(send_chat_request(client, scenario))
            results.append(task)

            # Print progress every 10 requests
            if len(results) % 10 == 0:
                elapsed = duration - (end_time - time.time())
                print(f"   ⏳ {elapsed:.0f}s elapsed, {len(results)} requests sent...")

            await asyncio.sleep(interval)

        # Wait for all in-flight requests
        completed = await asyncio.gather(*results, return_exceptions=True)

    # Print summary
    successes = [r for r in completed if isinstance(r, dict) and r.get("status") == "success"]
    errors = [r for r in completed if isinstance(r, dict) and r.get("status") != "success"]

    if successes:
        durations = [r["duration"] for r in successes]
        durations.sort()
        p50 = durations[len(durations) // 2]
        p99 = durations[int(len(durations) * 0.99)]
        total_cost = sum(r.get("cost", 0) or 0 for r in successes)

        print("\n✅ Load Test Summary")
        print(f"   Total requests:  {len(completed)}")
        print(f"   Successful:      {len(successes)} ({100*len(successes)/len(completed):.1f}%)")
        print(f"   Errors:          {len(errors)}")
        print(f"   Latency p50:     {p50*1000:.0f}ms")
        print(f"   Latency p99:     {p99*1000:.0f}ms")
        print(f"   Estimated cost:  ${total_cost:.4f} USD")
        print("\n📊 View results in Grafana: http://localhost:3000")
        print("🔍 View traces in Jaeger:   http://localhost:16686")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Observability Load Test")
    parser.add_argument("--rps", type=float, default=5.0, help="Requests per second")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument(
        "--scenario",
        choices=["normal", "spike", "error"],
        default="normal",
        help="Traffic scenario to simulate",
    )
    args = parser.parse_args()

    asyncio.run(run_load_test(args.rps, args.duration, args.scenario))
