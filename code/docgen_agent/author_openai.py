"""
Author agent for writing individual sections of the report.
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
from .prompts import section_writer_instructions

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


class Section(BaseModel):
    name: str
    content: str


class SectionWriterState(BaseModel):
    index: int
    section: Section
    topic: str
    messages: Annotated[Sequence[Any], add_messages] = []


async def research_section(state: SectionWriterState, config: RunnableConfig):
    """Research the section."""
    _LOGGER.info("Researching section: %s", state.section.name)

    # Create research queries for this section
    research_queries = [
        f"{state.section.name} {state.topic}",
        f"latest developments in {state.section.name} {state.topic}",
        f"technical details of {state.section.name} {state.topic}",
        f"real-world examples of {state.section.name} {state.topic}",
        f"best practices for {state.section.name} {state.topic}",
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
                "content": f"Research for {state.section.name}:\n{combined_research}",
            }
        ]
    }


async def writing_model(state: SectionWriterState, config: RunnableConfig):
    """Write the section content."""
    _LOGGER.info("Writing section: %s", state.section.name)

    model = llm.with_structured_output(str)  # type: ignore

    system_prompt = section_writer_instructions.format(
        section_name=state.section.name,
        topic=state.topic,
    )

    for count in range(_MAX_LLM_RETRIES):
        messages = [{"role": "system", "content": system_prompt}] + list(state.messages)
        response = await model.ainvoke(messages, config)
        if response:
            response = cast(str, response)
            state.section.content = response
            return state
        _LOGGER.debug(
            "Retrying LLM call. Attempt %d of %d", count + 1, _MAX_LLM_RETRIES
        )

    raise RuntimeError("Failed to call model after %d attempts.", _MAX_LLM_RETRIES)


# Build the graph
workflow = StateGraph(SectionWriterState)

# Add nodes
workflow.add_node("research", research_section)
workflow.add_node("writer", writing_model)

# Add edges
workflow.add_edge(START, "research")
workflow.add_edge("research", "writer")
workflow.add_edge("writer", END)

# Compile the graph
graph = workflow.compile()
