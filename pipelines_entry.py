import asyncio
import os
import sys
from concurrent.futures import ThreadPoolExecutor

# Make the repo root importable so `examples.customer_service.main` resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import Runner
from examples.customer_service.main import triage_agent, AirlineAgentContext

# Captures handoffs + sub-agent attribution for the multi-agent trace.
try:
    from pipelines.odyssey.adapters.openai_agents import pipelines_run_hooks
except Exception:
    pipelines_run_hooks = None


def run(task_input, *, proxy_url, run_token):
    instruction = (
        task_input.get("user_instruction")
        or task_input.get("input")
        or task_input.get("prompt")
        or "I'd like to change my seat."
    )
    kwargs = {"context": AirlineAgentContext()}
    if pipelines_run_hooks is not None:
        kwargs["hooks"] = pipelines_run_hooks()

    async def _go():
        return await Runner.run(triage_agent, instruction, **kwargs)

    # The platform runs this entrypoint inside an already-running event loop
    # (the E2B code-interpreter kernel), so Runner.run_sync() — which starts
    # its own loop — raises. Offload to a fresh thread with no loop of its own.
    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(lambda: asyncio.run(_go())).result()

    return {"final_response": str(result.final_output)}
