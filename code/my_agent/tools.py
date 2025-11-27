"""
Tools for your custom agent.

You can add your own custom tools here or import existing ones.
"""

# Import existing tools from the docgen_agent
from docgen_agent import tools as existing_tools

# Re-export the search tool
search_tavily = existing_tools.search_tavily

# Add your custom tools here
# Example:
# @tool
# def my_custom_tool(input_text: str) -> str:
#     """Your custom tool description."""
#     # Your tool implementation
#     return f"Processed: {input_text}"
