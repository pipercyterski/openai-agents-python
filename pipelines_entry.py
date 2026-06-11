import asyncio
from concurrent.futures import ThreadPoolExecutor

from agents import Agent, Runner, handoff
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX

from pipelines.odyssey import async_proxy_call
from pipelines.odyssey.context import set_current
from pipelines.odyssey.envelope import Envelope
from pipelines.odyssey.adapters.openai_agents import (
    pipelines_function_tool,
    pipelines_run_hooks,
    forward_run_result_events,
)

# Map Agent.name -> declared actor_id so observed handoff edges line up with
# the topology you declared in the form.
NAME_TO_ACTOR = {
    "Triage Agent": "triage_agent",
    "FAQ Agent": "faq_agent",
    "Seat Booking Agent": "seat_booking_agent",
}


def _build_agents(envelope):
    # Per-sub-agent proxy handles → deterministic actor attribution. Each tool
    # call goes through the platform proxy (simulated) and is stamped with the
    # owning sub-agent.
    faq = envelope.for_actor("faq_agent")
    seat = envelope.for_actor("seat_booking_agent")

    @pipelines_function_tool(actor=faq, name_override="faq_lookup_tool")
    async def faq_lookup_tool(question: str) -> str:
        return await async_proxy_call("faq_lookup_tool", {"question": question})

    @pipelines_function_tool(actor=seat, name_override="update_seat")
    async def update_seat(confirmation_number: str, new_seat: str) -> str:
        return await async_proxy_call(
            "update_seat",
            {"confirmation_number": confirmation_number, "new_seat": new_seat},
        )

    faq_agent = Agent(
        name="FAQ Agent",
        handoff_description="Answers airline FAQ questions.",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are an FAQ agent. Use faq_lookup_tool to answer the customer's "
            "question — do not rely on your own knowledge. If you can't help, "
            "transfer back to the triage agent."
        ),
        tools=[faq_lookup_tool],
    )
    seat_booking_agent = Agent(
        name="Seat Booking Agent",
        handoff_description="Updates the seat on a booking.",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX}\n"
            "You are a seat booking agent. Collect the confirmation number and "
            "desired seat (ask only if missing), then call update_seat. For "
            "anything off-topic, transfer back to the triage agent."
        ),
        tools=[update_seat],
    )
    triage_agent = Agent(
        name="Triage Agent",
        handoff_description="Routes the customer to the right specialist.",
        instructions=(
            f"{RECOMMENDED_PROMPT_PREFIX} You are a triage agent. Delegate each "
            "part of the customer's request to the FAQ agent or the seat booking "
            "agent as appropriate."
        ),
        handoffs=[
            handoff(agent=faq_agent, tool_name_override="transfer_to_faq_agent"),
            handoff(agent=seat_booking_agent, tool_name_override="transfer_to_seat_booking_agent"),
        ],
    )
    faq_agent.handoffs.append(handoff(agent=triage_agent, tool_name_override="transfer_to_triage_agent"))
    seat_booking_agent.handoffs.append(handoff(agent=triage_agent, tool_name_override="transfer_to_triage_agent"))
    return triage_agent


def run(task_input, *, proxy_url, run_token):
    instruction = (
        task_input.get("user_instruction")
        or task_input.get("input")
        or task_input.get("prompt")
        or "I'd like to change my seat."
    )

    def _worker():
        # Bind the run envelope (proxy_url + run_token) for the SDK, INSIDE this
        # worker thread — ContextVars don't cross the ThreadPoolExecutor boundary.
        envelope = Envelope.from_env()

        async def _go():
            triage = _build_agents(envelope)
            result = await Runner.run(
                triage,
                instruction,
                hooks=pipelines_run_hooks(name_to_actor=NAME_TO_ACTOR.get),
            )
            forward_run_result_events(result)  # post assistant messages / reasoning
            return result

        with set_current(envelope):
            return asyncio.run(_go())  # platform kernel already has a running loop

    with ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(_worker).result()

    return {"final_response": str(result.final_output)}
