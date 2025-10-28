"""Strategy 2: Writing Context

Add a hook that interrupts for approval and learns from user edits.
"""

from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass
from typing import Any

load_dotenv()

from langchain_openai import ChatOpenAI
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    after_model,
    AgentState,
)
from langchain.agents import create_agent
from langgraph.runtime import Runtime
from langgraph.types import interrupt

llm = ChatOpenAI(model="gpt-4o")


@dataclass
class EmailAssistantContext:
    user_name: str
    timezone: str


# Define tools
@tool
def send_email(
    subject: str, content: str, runtime: ToolRuntime[EmailAssistantContext]
) -> str:
    """Send email reply.

    Args:
        subject: Subject line
        content: Email body
    """
    return f"✅ Email sent from {runtime.context.user_name}!\n\nSubject: {subject}\n\n{content}"


# Strategy 1: Dynamic prompt that reads from runtime context and store
@dynamic_prompt
async def personalized_prompt(request: ModelRequest) -> str:
    """Build system prompt from runtime context and store."""
    user = request.runtime.context.user_name
    tz = request.runtime.context.timezone
    time = datetime.now().strftime("%I:%M %p")

    # Read user preferences from store
    tone_pref = await request.runtime.store.aget(("prefs",), f"{user}/tone")
    tone = tone_pref.value if tone_pref else "professional"

    return f"""You are an email assistant for {user}.
Please sign all emails with the name {user}.

Time: {time} ({tz})
Tone: {tone}

Draft email replies using send_email tool."""


# Strategy 2: Learn from user edits
async def learn_from_edit(original: dict, edited: dict, runtime: Runtime) -> None:
    """Use LLM to extract preferences from a single edit."""
    user = runtime.context.user_name

    # Single LLM call to analyze edit and extract tone preference
    analysis_prompt = f"""Compare these two email drafts and determine the user's preferred tone.

ORIGINAL:
Subject: {original.get("subject", "")}
Body: {original.get("content", "")}

EDITED:
Subject: {edited.get("subject", "")}
Body: {edited.get("content", "")}

Based on the changes (length, formality, word choice, greetings/closings), what tone does the user prefer?

Respond with ONLY ONE WORD: "casual" or "professional" """

    response = llm.invoke([{"role": "user", "content": analysis_prompt}])
    tone_pref = response.content.strip().lower()

    if tone_pref in ["casual", "professional"]:
        await runtime.store.aput(("prefs",), f"{user}/tone", tone_pref)
        print(f"📚 Learned preference: {tone_pref} tone")


@after_model
async def review_and_learn_hook(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    """Show draft for review using HITL syntax. Learn from edits using LLM."""
    from utils import create_interrupt_request, create_review_config

    messages = state.get("messages", [])
    if not messages:
        return None

    response = messages[-1]

    # Only interrupt for send_email tool calls
    if not hasattr(response, "tool_calls") or not response.tool_calls:
        return None

    # Filter for send_email calls
    send_email_calls = [tc for tc in response.tool_calls if tc["name"] == "send_email"]

    if not send_email_calls:
        return None

    # Create HITL request using new syntax (helper functions in utils.py)
    action_requests = [create_interrupt_request(tc) for tc in send_email_calls]
    review_configs = [create_review_config(tc["name"]) for tc in send_email_calls]

    hitl_request = {
        "action_requests": action_requests,
        "review_configs": review_configs,
    }

    # Show draft and wait for decisions
    hitl_response = interrupt(hitl_request)
    decisions = hitl_response["decisions"]

    # Process decisions and learn from edits
    for i, decision in enumerate(decisions):
        if decision["type"] == "edit":
            original_args = send_email_calls[i]["args"]
            edited_action = decision["edited_action"]
            edited_args = edited_action["args"]

            # Learn from the edit
            await learn_from_edit(original_args, edited_args, runtime)

            # Update the tool call with edited version
            original_index = response.tool_calls.index(send_email_calls[i])
            response.tool_calls[original_index]["args"] = edited_args

    return None


# Build Agent v2: Selecting + Writing Context
agent_v2 = create_agent(
    model="openai:gpt-4o",
    tools=[send_email],
    middleware=[personalized_prompt, review_and_learn_hook],
    context_schema=EmailAssistantContext,
    name="agent_v2",
)


"""
EXAMPLE MESSAGES TO COPY/PASTE INTO LANGGRAPH STUDIO:

Test: Human-in-the-loop with learning
--------------------------------------
Respond to: **From:** bob@company.com
**Subject:** Coffee?

Want to grab coffee next week?

NOTE: This agent will pause and show you a draft email for review.
You can either:
- Approve the draft as-is
- Edit the draft (the agent will learn your tone preference from your edits!)

When you edit the draft, the agent will use an LLM to analyze the differences
and automatically learn whether you prefer a "casual" or "professional" tone.
This preference is stored and will be used in future emails.
"""
