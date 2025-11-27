"""
Researcher agent for gathering information about a topic.
"""

import asyncio
import logging
import os
from typing import Annotated, Any, Sequence, cast

from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel

from . import tools
from .prompts import researcher_instructions

_LOGGER = logging.getLogger(__name__)
_MAX_LLM_RETRIES = 3
_QUERIES_PER_SECTION = 5
_THROTTLE_LLM_CALLS = os.getenv("THROTTLE_LLM_CALLS", "0")

# Use OpenAI instead of NVIDIA
llm = ChatOpenAI(
    model="gpt-4o-mini",  # Using GPT-4o-mini for cost efficiency
    temperature=0,
    api_key=os.getenv("OPENAI_API_KEY"),
)


class ResearcherState(BaseModel):
    topic: str
    number_of_queries: int
    messages: Annotated[Sequence[Any], add_messages] = []


async def research_model(state: ResearcherState, config: RunnableConfig):
    """Generate research queries and execute them."""
    _LOGGER.info("Calling model.")

    # Create research queries
    research_queries = [
        f"{state.topic} overview",
        f"latest developments in {state.topic}",
        f"technical details of {state.topic}",
        f"real-world applications of {state.topic}",
        f"future trends in {state.topic}",
    ]

    # Execute search queries
    search_results = []
    for query in research_queries:
        try:
            result = await tools.search_tavily([query])
            search_results.append(result)
        except Exception as e:
            _LOGGER.warning(f"Search failed for query '{query}': {e}")

    # Combine search results
    combined_research = "\n\n".join(search_results)

    return {
        "messages": [
            {
                "role": "user",
                "content": f"Research on {state.topic}:\n{combined_research}",
            }
        ]
    }


# Build the graph
workflow = StateGraph(ResearcherState)

# Add nodes
workflow.add_node("researcher", research_model)

# Add edges
workflow.add_edge(START, "researcher")
workflow.add_edge("researcher", END)

# Compile the graph
graph = workflow.compile()
