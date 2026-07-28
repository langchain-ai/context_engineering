"""Shared utilities for ce_min tutorial notebooks."""

from typing import Any


def format_email(from_addr: str, subject: str, body: str) -> str:
    """Format email as markdown for the agent."""
    return f"""**From:** {from_addr}
**Subject:** {subject}

{body}"""


def create_interrupt_request(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Format a tool call as an interrupt request using HITL syntax.

    Creates an ActionRequest with a nicely formatted description for email drafts.
    """
    subject = tool_call["args"].get("subject", "")
    content = tool_call["args"].get("content", "")

    return {
        "name": tool_call["name"],
        "args": tool_call["args"],
        "description": f"""📧 **Draft Email**

**Subject:** {subject}

**Body:**
{content}

---
*Review the draft above and choose to approve or edit.*"""
    }


def create_review_config(tool_name: str) -> dict[str, Any]:
    """Create review config for a tool with approve/edit decisions."""
    return {
        "action_name": tool_name,
        "allowed_decisions": ["approve", "edit"]
    }


def create_review_ui(draft_subject: str, draft_content: str):
    """Create an interactive ipywidgets UI for reviewing email drafts.

    Returns:
        A tuple of (ui_widget, get_decision_function)
        - ui_widget: Display this widget to show the review interface
        - get_decision_function: Call this to get the user's decision

    Example:
        >>> ui, get_decision = create_review_ui("Re: Meeting", "Let's meet tomorrow")
        >>> display(ui)
        >>> decision = get_decision()  # Returns decision dict after user clicks button
    """
    try:
        import ipywidgets as widgets
        from IPython.display import display, clear_output
    except ImportError:
        raise ImportError(
            "ipywidgets is required for interactive UI. "
            "Install with: pip install ipywidgets"
        )

    # Store the user's decision
    decision = {"type": None, "args": None}

    # Create text areas for editing
    subject_input = widgets.Textarea(
        value=draft_subject,
        description='Subject:',
        layout=widgets.Layout(width='100%', height='50px')
    )

    content_input = widgets.Textarea(
        value=draft_content,
        description='Body:',
        layout=widgets.Layout(width='100%', height='200px')
    )

    # Output area for feedback
    output = widgets.Output()

    # Button handlers
    def on_approve(b):
        with output:
            clear_output()
            decision["type"] = "approve"
            print("✅ Draft approved!")

    def on_edit(b):
        with output:
            clear_output()
            decision["type"] = "edit"
            decision["args"] = {
                "subject": subject_input.value,
                "content": content_input.value
            }
            print("✏️  Draft edited and approved!")

    # Create buttons
    approve_btn = widgets.Button(
        description='✅ Approve',
        button_style='success',
        tooltip='Approve the draft as-is',
        layout=widgets.Layout(width='150px')
    )
    approve_btn.on_click(on_approve)

    edit_btn = widgets.Button(
        description='✏️  Save Edits',
        button_style='primary',
        tooltip='Save your edits and approve',
        layout=widgets.Layout(width='150px')
    )
    edit_btn.on_click(on_edit)

    # Layout
    ui = widgets.VBox([
        widgets.HTML("<h3>📧 Review Draft Email</h3>"),
        subject_input,
        content_input,
        widgets.HBox([approve_btn, edit_btn]),
        output
    ])

    def get_decision():
        """Get the user's decision. Blocks until a button is clicked."""
        # Wait for user to click a button
        while decision["type"] is None:
            import time
            time.sleep(0.1)

        if decision["type"] == "approve":
            return {"type": "approve"}
        elif decision["type"] == "edit":
            return {
                "type": "edit",
                "edited_action": {
                    "name": "send_email",
                    "args": decision["args"]
                }
            }

    return ui, get_decision
