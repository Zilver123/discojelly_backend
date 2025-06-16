from openai import OpenAI
from typing import List, Callable, Any
import json

class WebSearchTool:
    """A tool for performing web searches."""
    def __init__(self):
        self.client = OpenAI()
    
    def __call__(self, query: str) -> str:
        """Perform a web search using OpenAI's web search capability."""
        try:
            # If query is a JSON string, parse it
            if isinstance(query, str) and query.strip().startswith('{'):
                query_data = json.loads(query)
                query = query_data.get('query', '')
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a web search assistant. Search for information about AI models on Replicate.com."},
                    {"role": "user", "content": query}
                ],
                tools=[{
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "description": "Search the web for information about AI models on Replicate.com.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The search query"
                                }
                            },
                            "required": ["query"]
                        }
                    }
                }],
                tool_choice="auto"
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error performing web search: {str(e)}"

def function_tool(func: Callable) -> Callable:
    """Decorator to mark a function as a tool."""
    func.is_tool = True
    return func

class Agent:
    """An agent that can use tools to perform tasks."""
    def __init__(self, name: str, instructions: str, tools: List[Any]):
        self.name = name
        self.instructions = instructions
        self.tools = tools
        self.client = OpenAI()
    
    def run(self, query: str) -> str:
        """Run the agent with the given query."""
        messages = [
            {"role": "system", "content": self.instructions},
            {"role": "user", "content": query}
        ]
        
        while True:
            try:
                # Create tool definitions for OpenAI
                tool_definitions = []
                for tool in self.tools:
                    if hasattr(tool, 'is_tool'):
                        # For function tools
                        tool_definitions.append({
                            "type": "function",
                            "function": {
                                "name": tool.__name__,
                                "description": tool.__doc__ or "",
                                "parameters": {
                                    "type": "object",
                                    "properties": {
                                        param: {"type": "string"}
                                        for param in tool.__annotations__
                                        if param != 'return'
                                    },
                                    "required": list(tool.__annotations__.keys())[:-1]
                                }
                            }
                        })
                    else:
                        # For class tools like WebSearchTool
                        tool_definitions.append({
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "description": "Search the web for information about AI models on Replicate.com."
                            }
                        })
                
                print("\nSending request to OpenAI...")
                response = self.client.chat.completions.create(
                    model="gpt-4",
                    messages=messages,
                    tools=tool_definitions,
                    tool_choice="auto"
                )
                
                message = response.choices[0].message
                messages.append(message)
                
                if message.tool_calls:
                    print("\nTool calls detected:")
                    for tool_call in message.tool_calls:
                        print(f"\nUsing tool: {tool_call.function.name}")
                        print(f"Arguments: {tool_call.function.arguments}")
                        
                        tool_response = None
                        if tool_call.function.name == "web_search":
                            for tool in self.tools:
                                if isinstance(tool, WebSearchTool):
                                    print("Performing web search...")
                                    tool_response = tool(tool_call.function.arguments)
                                    break
                        else:
                            # Find and call the matching function tool
                            for tool in self.tools:
                                if hasattr(tool, 'is_tool') and tool.__name__ == tool_call.function.name:
                                    print(f"Calling function tool: {tool.__name__}")
                                    args = json.loads(tool_call.function.arguments)
                                    tool_response = tool(**args)
                                    break
                        
                        if tool_response:
                            print("Tool response received")
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_call.function.name,
                                "content": tool_response
                            })
                else:
                    print("\nFinal response received")
                    return message.content
                    
            except Exception as e:
                return f"Error during agent execution: {str(e)}" 