"""Strategy 1: Selecting Context

Create a dynamic prompt that reads from runtime context and store.
"""

from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass

load_dotenv()

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent


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


# Build Agent v1: Selecting Context Only
agent_v1 = create_agent(
    model="openai:gpt-4o",
    tools=[send_email],
    middleware=[personalized_prompt],
    context_schema=EmailAssistantContext,
    name="agent_v1",
)


"""
EXAMPLE MESSAGES TO COPY/PASTE INTO LANGGRAPH STUDIO:

Test 1: Professional tone (default)
------------------------------------
Respond to: **From:** alice@company.com
**Subject:** Q4 Report

Hi Sydney, do you have the Q4 report ready? Thanks, Alice


Test 2: Casual tone (change tone preference first)
---------------------------------------------------
To change tone preference, you can use the store in Studio or run:
store.put(("prefs",), "Sydney/tone", "casual")

Then send:
Respond to: **From:** alice@company.com
**Subject:** Q4 Report

Hi Sydney, do you have the Q4 report ready? Thanks, Alice
"""
