from typing import Optional, Dict, Any, List
import os
import json
import logging
from functools import lru_cache
import tool_replicate
from models import Tool, AIAgent, get_supabase_client
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get environment variables
KEY = os.getenv('OPENAI_API_KEY')
if KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

app = FastAPI()

# Add CORS middleware with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins during development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
    expose_headers=["*"],
    max_age=3600,  # Cache preflight requests for 1 hour
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Request: {request.method} {request.url}")
    try:
        body = await request.body()
        if body:
            logger.info(f"Request body: {body.decode()}")
    except Exception as e:
        logger.error(f"Error reading request body: {str(e)}")
    
    response = await call_next(request)
    return response

@lru_cache(maxsize=100)
def get_cached_agent(agent_name: str) -> AIAgent:
    """Get an agent from the database with caching."""
    try:
        supabase = get_supabase_client()
        response = supabase.table('ai_agents').select('*').eq('name', agent_name).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        return AIAgent(**response.data[0])
    except Exception as e:
        logger.error(f"Error getting agent {agent_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving agent: {str(e)}")

@lru_cache(maxsize=100)
def get_cached_tool(tool_id: str) -> Tool:
    """Get a tool from the database with caching."""
    try:
        supabase = get_supabase_client()
        response = supabase.table('tools').select('*').eq('id', tool_id).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Tool with ID {tool_id} not found")
        return Tool(**response.data[0])
    except Exception as e:
        logger.error(f"Error getting tool {tool_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error retrieving tool: {str(e)}")

class Service:
    def __init__(self, api_key: str, agent_name: str):
        self.client = OpenAI(api_key=api_key)
        self.supabase = get_supabase_client()
        self.agent = get_cached_agent(agent_name)
        tool_ids = self.agent.tools.get('ids', []) if self.agent.tools else []
        self.tools = self.load_tools(tool_ids)
        self.context = [{"role": "system", "content": self.agent.system_prompt or ""}]
        self.api_handlers: Dict[str, Any] = {
            "Replicate": tool_replicate
        }
    
    def load_tools(self, tool_ids: List[str]) -> List[Dict[str, Any]]:
        tools = []
        for tool_id in tool_ids:
            try:
                tool = get_cached_tool(tool_id)
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.json_schema
                    }
                })
            except Exception as e:
                logger.error(f"Error loading tool {tool_id}: {str(e)}")
                continue
        return tools

    def run_tool(self, name: str, args: Dict[str, Any]) -> Any:
        try:
            # Find the tool in our loaded tools
            tool = next((t for t in self.tools if t["function"]["name"] == name), None)
            if not tool:
                raise ValueError(f"Tool {name} not found in loaded tools")
                
            # Get the tool details from database
            response = self.supabase.table('tools').select('*').eq('name', name).execute()
            if not response.data:
                raise ValueError(f"Tool {name} not found in database")
                
            tool_details = Tool(**response.data[0])
            
            # Get the appropriate handler based on api_name
            handler = self.api_handlers.get(tool_details.api_name)
            if not handler:
                raise ValueError(f"API {tool_details.api_name} not supported")
                
            # Call the handler with the model and args
            return handler.generate(tool_details.model, args)
        except Exception as e:
            logger.error(f"Error running tool {name}: {str(e)}")
            raise

    def call_model(self, message: Optional[str] = None) -> Any:
        if message is not None:
            self.context.append({"role": "user", "content": message})
        
        try:
            completion = self.client.chat.completions.create(
                model=self.agent.model or "gpt-4",
                messages=self.context,
                tools=self.tools,
            )
            return completion
        except Exception as e:
            logger.error(f"Error calling model: {str(e)}")
            raise

    def call_tools(self, payload: Any) -> None:
        try:
            for tool_call in payload.message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                result = self.run_tool(name, args)
                self.context.append(payload.message)
                self.context.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})
        except Exception as e:
            logger.error(f"Error calling tools: {str(e)}")
            raise

    def main(self, message: Optional[str] = None) -> str:
        try:
            completion = self.call_model(message)
            payload = completion.choices[0]

            if payload.message.tool_calls is not None:
                self.call_tools(payload)
                return self.main(None)
            else:
                self.context.append(payload.message)
                return payload.message.content
        except Exception as e:
            logger.error(f"Error in main function: {str(e)}")
            raise

class InputData(BaseModel):
    user_input: str
    agent_name: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/process-input")
async def process_input(data: InputData):
    try:
        logger.info(f"Processing input for agent {data.agent_name}")
        logger.info(f"User input: {data.user_input}")
        
        agent = Service(api_key=KEY, agent_name=data.agent_name)
        response = agent.main(data.user_input)
        
        logger.info(f"Response: {response}")
        
        return JSONResponse(
            content={"response": response},
            status_code=200,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except HTTPException as e:
        logger.error(f"HTTP error processing input: {str(e)}")
        return JSONResponse(
            content={"error": e.detail},
            status_code=e.status_code,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
    except Exception as e:
        logger.error(f"Error processing input: {str(e)}")
        return JSONResponse(
            content={"error": str(e)},
            status_code=500,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "Content-Type",
            }
        )
