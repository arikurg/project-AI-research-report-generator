#!/usr/bin/env python3
"""
Flask web application for the Document Generation Agent using OpenAI
"""

import asyncio
import logging
import os
import sys
import threading
import time
import traceback

from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session
from openai import OpenAI

# Load environment variables
load_dotenv("secrets.env")
load_dotenv("variables.env")

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple rate limiting
request_lock = threading.Lock()
last_request_time = 0
MIN_REQUEST_INTERVAL = 10  # Minimum seconds between requests

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def create_html_template():
    """Create the HTML template for the web interface."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document Generation Agent (OpenAI)</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 15px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content {
            padding: 40px;
        }
        
        .form-group {
            margin-bottom: 25px;
        }
        
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 600;
            color: #333;
        }
        
        input[type="text"], textarea {
            width: 100%;
            padding: 15px;
            border: 2px solid #e1e5e9;
            border-radius: 8px;
            font-size: 16px;
            transition: border-color 0.3s ease;
        }
        
        input[type="text"]:focus, textarea:focus {
            outline: none;
            border-color: #667eea;
        }
        
        textarea {
            resize: vertical;
            min-height: 150px;
        }
        
        .btn {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 15px 30px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s ease;
        }
        
        .btn:hover {
            transform: translateY(-2px);
        }
        
        .btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .loading {
            display: none;
            text-align: center;
            padding: 20px;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .result {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        
        .result h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .report-content {
            background: white;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e1e5e9;
            white-space: pre-wrap;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.6;
            max-height: 600px;
            overflow-y: auto;
        }
        
        .error {
            background: #fee;
            color: #c33;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #c33;
            margin-top: 20px;
        }
        
        .examples {
            margin-top: 30px;
            padding: 25px;
            background: #f8f9fa;
            border-radius: 8px;
        }
        
        .examples h3 {
            color: #333;
            margin-bottom: 15px;
        }
        
        .example-item {
            margin-bottom: 15px;
            padding: 15px;
            background: white;
            border-radius: 6px;
            border-left: 3px solid #667eea;
        }
        
        .example-item h4 {
            color: #667eea;
            margin-bottom: 8px;
        }
        
        .example-item p {
            color: #666;
            font-size: 14px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🤖 Document Generation Agent (OpenAI)</h1>
            <p>AI-powered research and report generation using OpenAI</p>
        </div>
        
        <div class="content">
            <form id="reportForm">
                <div class="form-group">
                    <label for="topic">📝 Topic:</label>
                    <input type="text" id="topic" name="topic" placeholder="Enter the topic for your report (e.g., 'Advantages of using GPUs for AI training')" required>
                </div>
                
                <div class="form-group">
                    <label for="report_structure">📋 Report Structure:</label>
                    <textarea id="report_structure" name="report_structure" placeholder="Describe the structure and requirements for your report..." required>This report type focuses on comprehensive analysis.

The report structure should include:
1. Introduction (no research needed)
- Brief overview of the topic area
- Context and importance

2. Main Body Sections:
- Detailed analysis of key aspects
- Technical details and implementations
- Real-world examples and use cases

3. Conclusion (no research needed)
- Summary of key findings
- Final recommendations

4. Additional sections as needed based on the topic</textarea>
                </div>
                
                <button type="submit" class="btn" id="generateBtn">
                    🚀 Generate Report
                </button>
            </form>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p>🤖 AI Agent is researching and writing your report...</p>
                <p><small>This may take a few minutes as the agent searches the web and generates content.</small></p>
            </div>
            
            <div id="result" class="result" style="display: none;">
                <h3>📄 Generated Report</h3>
                <div id="reportContent" class="report-content"></div>
            </div>
            
            <div id="error" class="error" style="display: none;"></div>
            
            <div class="examples">
                <h3>💡 Example Topics</h3>
                <div class="example-item">
                    <h4>Technology</h4>
                    <p>"Advantages of using GPUs for AI training"</p>
                </div>
                <div class="example-item">
                    <h4>Business</h4>
                    <p>"Impact of remote work on productivity and company culture"</p>
                </div>
                <div class="example-item">
                    <h4>Science</h4>
                    <p>"Latest developments in quantum computing and its applications"</p>
                </div>
                <div class="example-item">
                    <h4>Health</h4>
                    <p>"Benefits and risks of intermittent fasting for weight loss"</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        document.getElementById('reportForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const topic = document.getElementById('topic').value;
            const reportStructure = document.getElementById('report_structure').value;
            const generateBtn = document.getElementById('generateBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            
            // Show loading, hide results and errors
            loading.style.display = 'block';
            result.style.display = 'none';
            error.style.display = 'none';
            generateBtn.disabled = true;
            generateBtn.textContent = '⏳ Generating...';
            
            try {
                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        topic: topic,
                        report_structure: reportStructure
                    })
                });
                
                const data = await response.json();
                
                if (response.ok && data.success) {
                    document.getElementById('reportContent').textContent = data.report;
                    result.style.display = 'block';
                } else {
                    error.textContent = data.error || 'An error occurred while generating the report.';
                    error.style.display = 'block';
                }
            } catch (err) {
                error.textContent = 'Network error: ' + err.message;
                error.style.display = 'block';
            } finally {
                loading.style.display = 'none';
                generateBtn.disabled = false;
                generateBtn.textContent = '🚀 Generate Report';
            }
        });
    </script>
</body>
</html>"""

    with open("templates/index.html", "w") as f:
        f.write(html_content)


@app.route("/")
def index():
    """Main page with the document generation form."""
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_report():
    """Generate a report based on the form data using OpenAI."""
    global last_request_time

    # Rate limiting
    with request_lock:
        current_time = time.time()
        time_since_last = current_time - last_request_time

        if time_since_last < MIN_REQUEST_INTERVAL:
            wait_time = MIN_REQUEST_INTERVAL - time_since_last
            return (
                jsonify(
                    {
                        "error": f"Rate limit exceeded. Please wait {wait_time:.1f} seconds before making another request."
                    }
                ),
                429,
            )

        last_request_time = current_time

    try:
        data = request.get_json()
        topic = data.get("topic", "").strip()
        report_structure = data.get("report_structure", "").strip()

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        if not report_structure:
            return jsonify({"error": "Report structure is required"}), 400

        logger.info(f"Generating report for topic: {topic}")

        # Create a comprehensive prompt for OpenAI
        prompt = f"""You are an expert research and report writer. Create a comprehensive, well-structured report on the topic: "{topic}"

Report Structure Requirements:
{report_structure}

Please write a detailed, professional report that includes:
1. An engaging introduction that provides context
2. Multiple well-researched sections covering different aspects of the topic
3. Real-world examples and applications
4. Technical details where relevant
5. A comprehensive conclusion with key takeaways
6. Professional formatting with clear headings and subheadings

Make the report informative, well-researched, and suitable for a professional audience."""

        # Call OpenAI
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert research and report writer. Create comprehensive, well-structured reports with professional formatting.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4000,
        )

        report = response.choices[0].message.content

        return jsonify({"success": True, "report": report, "topic": topic})

    except Exception as e:
        logger.error(f"Error generating report: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Error generating report: {str(e)}"}), 500


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "healthy"})


if __name__ == "__main__":
    # Install Flask if not already installed
    try:
        import flask
    except ImportError:
        logger.info("Installing Flask...")
        os.system("pip3 install flask>=3.0.0")

    # Create templates directory if it doesn't exist
    os.makedirs("templates", exist_ok=True)

    # Create the HTML template
    create_html_template()

    logger.info("Starting Flask app on http://localhost:5001")
    app.run(debug=True, host="0.0.0.0", port=5001)
