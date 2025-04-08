from typing import Optional
import os
import json
import tool_replicate
from models import Tool, AIAgent, get_supabase_client
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

KEY = os.getenv('OPENAI_API_KEY')
if KEY is None:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zilver123.github.io"],  # Allow specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class Service:
    def __init__(self, api_key: str, agent_name: str):
        self.client = OpenAI(api_key=api_key)
        self.supabase = get_supabase_client()
        self.agent = self.load_agent(agent_name)
        self.tools = self.load_tools(self.agent.tool_ids or [])
        self.context = [{"role": "system", "content": self.agent.system_prompt or ""}]
    
    def load_agent(self, agent_name: str) -> AIAgent:
        response = self.supabase.table('ai_agents').select('*').eq('name', agent_name).execute()
        if not response.data:
            raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
        return AIAgent(**response.data[0])
    
    def load_tools(self, tool_ids: list) -> list:
        tools = []
        for tool_id in tool_ids:
            response = self.supabase.table('tools').select('*').eq('id', tool_id).execute()
            if response.data:
                tool = Tool(**response.data[0])
                tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.json_schema
                    }
                })
        return tools

    def run_tool(self, name: str, args: dict):
        if name == "generate_image":
            return tool_replicate.generate("black-forest-labs/flux-1.1-pro", args)
        elif name == "generate_music_v2":
            return tool_replicate.generate("meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb", args)
        elif name == "generate_music":
            return tool_replicate.generate("minimax/music-01", args)

    def call_model(self, message: Optional[str] = None):
        if message is not None:
            self.context.append({"role": "user", "content": message})
        
        completion = self.client.chat.completions.create(
            model=self.agent.model or "gpt-4",
            messages=self.context,
            tools=self.tools,
        )
        return completion

    def call_tools(self, payload):
        for tool_call in payload.message.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = self.run_tool(name, args)
            self.context.append(payload.message)
            self.context.append({"role": "tool", "tool_call_id": tool_call.id, "content": str(result)})

    def main(self, message: Optional[str] = None):
        completion = self.call_model(message)
        payload = completion.choices[0]

        if payload.message.tool_calls is not None:
            self.call_tools(payload)
            return self.main(None)
        else:
            self.context.append(payload.message)
            return payload.message.content

class InputData(BaseModel):
    user_input: str
    agent_name: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/process-input")
async def process_input(data: InputData):
    try:
        agent = Service(api_key=KEY, agent_name=data.agent_name)
        response = agent.main(data.user_input)
        return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



