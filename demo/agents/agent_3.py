"""Strategy 3: Summarizing Context

Add summarization middleware to manage long conversations.
"""

from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass

load_dotenv()

from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import (
    dynamic_prompt,
    ModelRequest,
    SummarizationMiddleware,
)
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


# Strategy 3: Summarization middleware
summarization = SummarizationMiddleware(
    model="gpt-4o-mini",
    max_tokens_before_summary=500,
    messages_to_keep=4,
)


# Build Agent v3: Selecting + Writing + Summarizing
agent_v3 = create_agent(
    model="openai:gpt-4o",
    tools=[send_email],
    middleware=[personalized_prompt, summarization],
    context_schema=EmailAssistantContext,
    name="agent_v3",
)


"""
EXAMPLE MESSAGES TO COPY/PASTE INTO LANGGRAPH STUDIO:

Test: Long email chain with summarization
------------------------------------------
Send these emails one at a time in the same thread to see summarization in action:

Email 1:
Respond to: **From:** product@company.com
**Subject:** Product Launch

Can your team handle the documentation by March 15th?

Email 2:
Respond to: **From:** marketing@company.com
**Subject:** Marketing Materials

We need your input on the marketing copy.

Email 3:
Respond to: **From:** legal@company.com
**Subject:** Legal Review

All docs need disclaimers by March 20th.

Email 4:
Respond to: **From:** ceo@company.com
**Subject:** Status check

Can you give me a status update on all the documentation deliverables?

NOTE: As the conversation gets longer, the summarization middleware will automatically
condense earlier messages while keeping recent context. This prevents the context
from growing too large while maintaining conversation continuity.
"""
