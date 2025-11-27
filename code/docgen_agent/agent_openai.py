"""
The main agent that orchestrates the report generation process using OpenAI.
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

from . import author, researcher
from .prompts import report_planner_instructions

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


class Report(BaseModel):
    title: str
    sections: list[author.Section]


class AgentState(BaseModel):
    topic: str
    report_structure: str
    report_plan: Report | None = None
    report: str | None = None
    messages: Annotated[Sequence[Any], add_messages] = []


async def topic_research(state: AgentState, config: RunnableConfig):
    """Research the topic of the document."""
    _LOGGER.info("Performing initial topic research.")

    researcher_state = researcher.ResearcherState(
        topic=state.topic,
        number_of_queries=_QUERIES_PER_SECTION,
        messages=state.messages,
    )

    research = await researcher.graph.ainvoke(researcher_state, config)

    return {"messages": research.get("messages", [])}


async def report_planner(state: AgentState, config: RunnableConfig):
    """Call the model."""
    _LOGGER.info("Calling report planner.")

    model = llm.with_structured_output(Report)  # type: ignore

    system_prompt = report_planner_instructions.format(
        topic=state.topic,
        report_structure=state.report_structure,
    )
    for count in range(_MAX_LLM_RETRIES):
        messages = [{"role": "system", "content": system_prompt}] + list(state.messages)
        response = await model.ainvoke(messages, config)
        if response:
            response = cast(Report, response)
            state.report_plan = response
            return state
        _LOGGER.debug(
            "Retrying LLM call. Attempt %d of %d", count + 1, _MAX_LLM_RETRIES
        )

    raise RuntimeError("Failed to call model after %d attempts.", _MAX_LLM_RETRIES)


async def section_author_orchestrator(state: AgentState, config: RunnableConfig):
    """Orchestrate the section authoring process."""
    if not state.report_plan:
        raise ValueError("Report plan is not set.")

    _LOGGER.info("Orchestrating the section authoring process.")

    writers = []
    for idx, section in enumerate(state.report_plan.sections):
        _LOGGER.info("Creating author agent for section: %s", section.name)

        section_writer_state = author.SectionWriterState(
            index=idx,
            section=section,
            topic=state.topic,
            messages=state.messages,
        )
        writers.append(author.graph.ainvoke(section_writer_state, config))

    all_sections = []
    if _THROTTLE_LLM_CALLS == "1":
        # Throttle LLM calls by writing one section at a time
        for writer in writers:
            section_result = await writer
            all_sections.append(section_result)
    else:
        # Write all sections in parallel
        all_sections = await asyncio.gather(*writers)

    _LOGGER.info("Finished section: %s", section.name)

    return {"messages": all_sections[-1].get("messages", [])}


async def report_author(state: AgentState, config: RunnableConfig):
    """Author the final report."""
    if not state.report_plan:
        raise ValueError("Report plan is not set.")

    _LOGGER.info("Authoring the report.")

    model = llm.with_structured_output(str)  # type: ignore

    system_prompt = """You are an expert report writer. Compile all the sections into a comprehensive, well-structured report. 
    Ensure the report flows logically and maintains professional formatting."""

    # Combine all sections into a single report
    report_content = "\n\n".join(
        [
            f"## {section.name}\n{section.content}"
            for section in state.report_plan.sections
        ]
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Compile this into a final report:\n\n{report_content}",
        },
    ]

    response = await model.ainvoke(messages, config)
    state.report = response

    return state


# Build the graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("topic_research", topic_research)
workflow.add_node("report_planner", report_planner)
workflow.add_node("section_author_orchestrator", section_author_orchestrator)
workflow.add_node("report_author", report_author)

# Add edges
workflow.add_edge(START, "topic_research")
workflow.add_edge("topic_research", "report_planner")
workflow.add_edge("report_planner", "section_author_orchestrator")
workflow.add_edge("section_author_orchestrator", "report_author")
workflow.add_edge("report_author", END)

# Compile the graph
graph = workflow.compile()
