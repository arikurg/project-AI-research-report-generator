"""
My Custom AI Agent

This is a template for building your own AI agent.
You can customize this for different use cases.
"""

import logging
from typing import Annotated, Any, Sequence

from langchain_core.runnables import RunnableConfig
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from . import tools
from .prompts import agent_prompt

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3

# Initialize the LLM
llm = ChatNVIDIA(model="meta/llama-3.3-70b-instruct", temperature=0)
llm_with_tools = llm.bind_tools([tools.search_tavily])  # Add your tools here


class AgentState(BaseModel):
    """State for your custom agent."""

    user_input: str
    # Add more fields as needed for your specific agent
    messages: Annotated[Sequence[Any], add_messages] = []
    # Chat history


async def tool_node(state: AgentState):
    """Execute tool calls."""
    _LOGGER.info("Executing tool calls.")
    outputs = []
    for tool_call in state.messages[-1].tool_calls:
        _LOGGER.info("Executing tool call: %s", tool_call["name"])
        tool = getattr(tools, tool_call["name"])
        tool_result = await tool.ainvoke(tool_call["args"])
        outputs.append(
            {
                "role": "tool",
                "content": str(tool_result),
                "name": tool_call["name"],
                "tool_call_id": tool_call["id"],
            }
        )
    return {"messages": outputs}


async def call_model(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Call the LLM with the current state."""
    _LOGGER.info("Calling model.")

    # Customize this prompt for your specific agent
    system_prompt = agent_prompt.format(user_input=state.user_input)

    for count in range(_MAX_LLM_RETRIES):
        messages = [{"role": "system", "content": system_prompt}] + list(state.messages)
        response = await llm_with_tools.ainvoke(messages, config)

        if response:
            return {"messages": [response]}

        _LOGGER.debug(
            "Retrying LLM call. Attempt %d of %d", count + 1, _MAX_LLM_RETRIES
        )

    raise RuntimeError("Failed to call model after %d attempts.", _MAX_LLM_RETRIES)


def has_tool_calls(state: AgentState) -> bool:
    """Check if the last message has tool calls."""
    messages = state.messages
    if not messages:
        return False
    last_message = messages[-1]
    return bool(last_message.tool_calls)


# Build the workflow
workflow = StateGraph(AgentState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges(
    "agent",
    has_tool_calls,
    {
        True: "tools",
        False: END,
    },
)
workflow.add_edge("tools", "agent")

graph = workflow.compile()
