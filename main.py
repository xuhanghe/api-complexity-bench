"""Example entry point that wires up BenchMCPClient with a sample config.

This script demonstrates how you might instantiate the benchmarking client,
retrieve tools using the configured strategy, and trigger a simple tool call.
It is intended as illustrative scaffolding and may need adjustments for your
local environment (for example, ensuring the sample server command is valid).
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

CONFIG_FILENAME = "bench-client-config.example.json"
CLIENT_MODULE_NAME = "bench_client"
CLIENT_MODULE_PATH = Path(__file__).resolve().parent / "bench-client.py"


def _load_bench_client_module() -> Any:
    """Dynamically import the bench-client implementation."""
    spec = importlib.util.spec_from_file_location(CLIENT_MODULE_NAME, CLIENT_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load bench-client module from {CLIENT_MODULE_PATH}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def run_demo() -> None:
    bench_module = _load_bench_client_module()
    BenchMCPClient = bench_module.BenchMCPClient  # type: ignore[attr-defined]

    config_path = Path(__file__).resolve().parent / CONFIG_FILENAME
    if not config_path.exists():
        raise FileNotFoundError(f"Expected config file at {config_path}")

    async with BenchMCPClient(config_path) as client:
        # Simulate an incoming user intent that will drive tool retrieval.
        request_text = "Fetch the front page for https://example.com and summarize it."
        session_context = [
            "Prefer concise summaries",
            "Avoid destructive operations",
        ]
        retrieval = await client.handle_user_request(request_text, previous_context=session_context)
        print("Tool retrieval result:", retrieval)

        # If the strategy returned tool names, attempt to call the first one
        # using placeholder arguments. Wrap in error handling to keep the demo resilient.
        result_field = retrieval.get("result")
        if isinstance(result_field, list) and result_field:
            chosen_tool = result_field[0]
            try:
                call_result = await client.call_tool(
                    chosen_tool,
                    arguments={"url": "https://example.com"},
                )
                print("Sample call_tool response:", call_result)
            except Exception as exc:  # noqa: PERF203 - demo resilience
                print(f"Tool call failed for {chosen_tool}: {exc}")
        else:
            print("Context injection output:\n", result_field)


def main() -> None:
    try:
        asyncio.run(run_demo())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
