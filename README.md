AI Research Report Generator

A Flask-based application that uses autonomous AI agents to research topics and generate professional reports. Built with LangGraph and Tavily, it supports both NVIDIA NIM and OpenAI models.

🚀 Key Features

Autonomous Research: Agents search the web and verify facts using Tavily API.

Multi-Agent Workflow: Specialized agents plan, research, and write the report.

Dual Model Support: Run with either NVIDIA NIM or OpenAI backends.

Web Interface: Simple Flask UI for easy topic submission and report viewing.

🛠️ Technical Architecture

Flask Backend

The core application is built on Flask, serving as the bridge between the user interface and the AI agents.

API Design: exposes RESTful endpoints (POST /generate, GET /health) to handle report requests and monitor system status.

Concurrency Control: Implements thread-safe rate limiting using threading.Lock to manage API usage and ensure stability during heavy workloads.

Dynamic Rendering: Utilizes Jinja2 templating to serve a responsive, client-side application that updates in real-time.

Environment Management: securely loads configuration and API keys using python-dotenv.

Agentic Workflow (LangGraph)

The AI logic uses a state-based graph architecture:

State Management: A shared AgentState object persists context across the workflow.

Cyclic Graph: The Researcher, Planner, and Author nodes interact iteratively, allowing the AI to refine its search and content generation strategies dynamically before finalizing the report.

⚡ Quick Start

1. Installation

git clone <your-repo-url>
cd project-ai-research-report-generator
pip install -r requirements.txt


2. Configuration

Create a secrets.env file in the root directory and add your API keys:

NVIDIA_API_KEY=nvapi-...
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...


3. Usage

Run with NVIDIA/Default Agents:

python app.py


Run with OpenAI Agents:

python app_openai.py


Open your browser to http://localhost:5001, enter a topic, and click Generate.
