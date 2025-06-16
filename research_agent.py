from openai import OpenAI
import requests
import json
from bs4 import BeautifulSoup
import os
import sys
from typing import Dict, Any, List
import re
from datetime import datetime
from agents import Agent, WebSearchTool, function_tool

REPLICATE_TOKEN = os.environ.get('REPLICATE_API_TOKEN')

@function_tool
def get_schema(model_owner: str, model_name: str) -> str:
    """Get the input schema for a Replicate model."""
    if not REPLICATE_TOKEN:
        return "Error: REPLICATE_API_TOKEN environment variable not set"
        
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}"}
    try:
        model_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}"
        model_resp = requests.get(model_url, headers=headers)
        if model_resp.status_code != 200:
            return f"Error: Could not fetch model info: {model_resp.text}"
        version_id = model_resp.json()["latest_version"]["id"]
        
        version_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/versions/{version_id}"
        version_resp = requests.get(version_url, headers=headers)
        if version_resp.status_code != 200:
            return f"Error: Could not fetch version info: {version_resp.text}"
        input_schema = version_resp.json()["openapi_schema"]["components"]["schemas"]["Input"]
        return json.dumps(input_schema, indent=2)
    except Exception as e:
        return f"Error fetching schema: {str(e)}"

@function_tool
def save_results(output: str) -> str:
    """Save the research results to a database."""
    timestamp = datetime.now()
    print(f"\nResults saved at {timestamp}:")
    print(output)
    return "Results saved successfully"

def get_model_schema(model_path):
    """Scrape the schema directly from Replicate's API schema endpoint."""
    url = f"https://replicate.com/{model_path}/api/schema#input-schema"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            # Parse HTML and find the schema
            soup = BeautifulSoup(response.text, 'html.parser')
            # Look for the schema in a pre or code tag
            schema_element = soup.find('pre') or soup.find('code')
            if schema_element:
                return schema_element.text
            return "Error: Could not find schema in the page"
        else:
            return f"Error: Could not fetch schema. Status code: {response.status_code}"
    except Exception as e:
        return f"Error fetching schema: {str(e)}"

def get_replicate_input_schema(model_owner: str, model_name: str) -> dict:
    """
    Retrieve the input schema for a Replicate model using the official API.
    Returns the openapi_schema.components.schemas.Input JSON.
    """
    if not REPLICATE_TOKEN:
        return {"error": "REPLICATE_API_TOKEN environment variable not set"}
        
    headers = {"Authorization": f"Token {REPLICATE_TOKEN}"}
    try:
        # Step 1: Get the latest version ID
        model_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}"
        model_resp = requests.get(model_url, headers=headers)
        if model_resp.status_code != 200:
            return {"error": f"Could not fetch model info: {model_resp.text}"}
        version_id = model_resp.json()["latest_version"]["id"]
        
        # Step 2: Get the input schema
        version_url = f"https://api.replicate.com/v1/models/{model_owner}/{model_name}/versions/{version_id}"
        version_resp = requests.get(version_url, headers=headers)
        if version_resp.status_code != 200:
            return {"error": f"Could not fetch version info: {version_resp.text}"}
        input_schema = version_resp.json()["openapi_schema"]["components"]["schemas"]["Input"]
        return input_schema
    except Exception as e:
        return {"error": f"Error fetching schema: {str(e)}"}

client = OpenAI()

# Tool definitions for OpenAI function-calling
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_schema",
            "description": "Get the input schema for a Replicate model using the official API. Use this after finding a suitable model through web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "model_owner": {"type": "string", "description": "The owner of the model, e.g. 'black-forest-labs'"},
                    "model_name": {"type": "string", "description": "The name of the model, e.g. 'flux-1.1-pro'"}
                },
                "required": ["model_owner", "model_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information about AI models on Replicate.com. Use this to find suitable models for a given prompt or task."
        }
    }
]

def format_model_info(model_info: Dict[str, Any]) -> str:
    """Format model information in a readable way."""
    if "error" in model_info:
        return f"Error: {model_info['error']}"
    
    formatted_info = "Model Schema Information:\n"
    for key, value in model_info.items():
        if isinstance(value, dict):
            formatted_info += f"\n{key}:\n"
            for subkey, subvalue in value.items():
                formatted_info += f"  {subkey}: {subvalue}\n"
        else:
            formatted_info += f"{key}: {value}\n"
    return formatted_info

def extract_replicate_model_urls(text: str) -> list:
    """Extract Replicate.com model URLs from a block of text."""
    # Replicate model URLs are like https://replicate.com/owner/model
    url_pattern = r"https://replicate.com/([\w-]+)/([\w-]+)"
    return re.findall(url_pattern, text)

# Create the research agent with the specified tools
research_agent = Agent(
    name="Research Agent",
    instructions="""
You are a specialized research agent that helps users find and understand AI models on Replicate.com. Your tasks are:

1. Use web search to find suitable models on Replicate.com that match the user's requirements
2. For each promising model found, use get_schema to retrieve its input schema
3. Present the information in a clear, organized format
4. Save the results using save_results

Guidelines:
- Focus ONLY on models available on Replicate.com
- Use web search to find models that match the user's requirements
- After finding suitable models, use get_schema to get their exact input schemas
- Present the information in a structured, easy-to-understand format
- If no suitable models are found, explain why and suggest alternative approaches

Remember to:
- Be thorough in your research
- Validate model compatibility with the user's requirements
- Provide clear explanations of model capabilities and limitations
""",
    tools=[WebSearchTool(), get_schema, save_results]
)

def run_agent(user_query: str):
    """Run the research agent with the given query."""
    try:
        print("\nStarting research agent...")
        print(f"Query: {user_query}\n")
        
        # Run the agent with the user's query
        result = research_agent.run(user_query)
        
        print("\nResearch completed!")
        print("="*50)
        print(result)
        print("="*50)
        
    except Exception as e:
        print(f"Error during agent execution: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = input("Enter your research query: ")
    run_agent(query) 