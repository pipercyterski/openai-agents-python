import os
import sys

# Make the repo root importable so `examples.customer_service.main` resolves.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from agents import Runner
from examples.customer_service.main import triage_agent, AirlineAgentContext

# Installed via the Setup command (branch SDK from the monorepo). Captures
# handoffs + sub-agent attribution. Degrades gracefully if absent.
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
    result = Runner.run_sync(triage_agent, instruction, **kwargs)
    return {"final_response": str(result.final_output)}
