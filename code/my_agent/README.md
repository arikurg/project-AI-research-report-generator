# My Custom AI Agent

This is a template for building your own AI agent for the hackathon. You can customize it for different use cases.

## Quick Start

1. **Open the client notebook**: `my_agent_client.ipynb`
2. **Run the cells** to test your agent
3. **Customize the agent** by editing the files below

## Files to Customize

### `agent.py` - Main Agent Logic
- **AgentState**: Add fields for your specific use case
- **call_model()**: Customize how the agent processes requests
- **Workflow**: Add new nodes or modify the flow

### `prompts.py` - System Prompts
- **agent_prompt**: Main system prompt for your agent
- **Example prompts**: Research, Code, Data, Content agents

### `tools.py` - Agent Tools
- **Import existing tools**: Search, etc.
- **Add custom tools**: Create new functions with `@tool` decorator

### `my_agent_client.ipynb` - User Interface
- **Input handling**: How users interact with your agent
- **Output display**: How results are shown

## Agent Ideas

### Research Agent
- Gathers information on any topic
- Uses search tools to find current data
- Organizes findings in structured format

### Code Assistant
- Helps with programming questions
- Provides code examples and explanations
- Suggests best practices

### Data Analyst
- Analyzes data and provides insights
- Creates visualizations
- Generates reports

### Content Creator
- Writes articles, social media posts
- Researches topics thoroughly
- Adapts writing style to audience

### Task Planner
- Breaks down complex tasks
- Creates step-by-step plans
- Tracks progress

### Customer Support
- Answers common questions
- Provides troubleshooting help
- Escalates complex issues

## Adding Custom Tools

```python
from langchain_core.tools import tool

@tool
def my_custom_tool(input_text: str) -> str:
    """Description of what your tool does."""
    # Your tool implementation
    return f"Processed: {input_text}"
```

## Testing Your Agent

1. Run `my_agent_client.ipynb`
2. Modify the `user_input` variable
3. Check the output and conversation history
4. Iterate and improve!

## Tips

- **Start simple**: Get basic functionality working first
- **Test frequently**: Run your agent often to catch issues
- **Use existing tools**: Leverage the search and other tools
- **Customize prompts**: The system prompt is crucial for behavior
- **Add logging**: Use `_LOGGER.info()` to debug your agent

Good luck with your hackathon project! 🚀 