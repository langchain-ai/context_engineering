"""Strategy 4: Isolating Context

Add a calendar subagent with specialized tools.
"""

from dotenv import load_dotenv
from datetime import datetime
from dataclasses import dataclass

load_dotenv()

from langchain.tools import tool, ToolRuntime
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from deepagents import create_deep_agent


@dataclass
class EmailAssistantContext:
    user_name: str
    timezone: str


# Main agent tools
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


# Calendar subagent tools
@tool
def check_calendar(date: str, runtime: ToolRuntime[EmailAssistantContext]) -> str:
    """Check calendar for a date.

    Args:
        date: Date in YYYY-MM-DD format
    """
    return f"""Events on {date}:
- 9:00 AM: Standup (30 min)
- 2:00 PM: Client Meeting (1 hr)

Free: 10 AM-1 PM, 3-5 PM"""


@tool
def schedule_meeting(
    date: str, time: str, title: str, runtime: ToolRuntime[EmailAssistantContext]
) -> str:
    """Schedule a meeting.

    Args:
        date: Date in YYYY-MM-DD
        time: Time in HH:MM
        title: Meeting title
    """
    return f"✅ Scheduled '{title}' for {date} at {time}"


calendar_tools = [check_calendar, schedule_meeting]


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


# Build Agent v4: All strategies including subagent
agent_v4 = create_deep_agent(
    model="openai:gpt-4o",
    tools=[send_email],
    middleware=[personalized_prompt],
    subagents=[
        {
            "name": "calendar",
            "description": "Use for calendar/scheduling tasks",
            "system_prompt": "You are a calendar assistant. Help with scheduling.",
            "tools": calendar_tools,
        }
    ],
    context_schema=EmailAssistantContext,
)


"""
EXAMPLE MESSAGES TO COPY/PASTE INTO LANGGRAPH STUDIO:

Test: Subagent delegation for calendar tasks
---------------------------------------------
Respond to: **From:** carol@company.com
**Subject:** Meeting Request

Can we meet on March 20th to discuss the project? I'm flexible on timing.

Check calendar for 2024-03-20 and suggest times.

NOTE: This agent combines all strategies:
1. Selecting Context - Uses personalized prompt with runtime context
2. Writing Context - (Not active in this demo, but can be added)
3. Summarizing Context - (Not active in this demo, but can be added)
4. Isolating Context - Delegates calendar operations to a specialized subagent

The main agent will recognize that calendar checking is needed and automatically
delegate to the calendar subagent, which has specialized tools (check_calendar,
schedule_meeting). The subagent will check availability and return results to
the main agent, which will then draft an appropriate email response.
"""
