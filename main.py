from typing import Optional
import os
import json
import tool_replicate
from tools_config import tools

from fastapi import FastAPI
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
    def __init__(self, api_key, config):
        self.client = OpenAI(api_key=api_key)
        self.tools = config["tools"]
        self.model = config["model"]
        self.context = [{"role": "system", "content": config["instructions"]}]
    
    def run_tool(self, name, args):
        if name == "generate_image":
            return tool_replicate.generate("black-forest-labs/flux-1.1-pro", args)
        elif name == "generate_music_v2":
            return tool_replicate.generate("meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb", args)
        elif name == "generate_music":
            return tool_replicate.generate("minimax/music-01", args)

    def call_model(self, message):
        if message is not None:
            self.context.append({"role": "user", "content": message})
        print(self.context)
        completion = self.client.chat.completions.create(
            model=self.model,
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

    def main(self, message):
        completion = self.call_model(message)
        payload = completion.choices[0]

        if payload.message.tool_calls is not None:
            self.call_tools(payload)
            return self.main(None)
        else:
            self.context.append(payload.message)
            return payload.message.content

config = {
    "model": "gpt-4",
    "instructions": "You are a creator tool for content creators. You can generate images, videos, and text.",
    "tools": tools
}

agent = Service(api_key=KEY, config=config)

"""
class DeepSeekAgent:
    def __init__(self, api_key, model_name="deepseek-chat"):
        self.client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        self.model = model_name
        self.context = []

    def process_input(self, user_input):
        # Add input to context
        self.context.append({"role": "user", "content": user_input})
        
        # Generate response
        response = self.client.chat.completions.create(
            model=self.model,
            messages=self.context,
            temperature=0.7,
            max_tokens=1000
        )
        
        # Update context with response
        assistant_response = response.choices[0].message.content
        self.context.append({"role": "assistant", "content": assistant_response})
        
        return assistant_response

# Initialize DeepSeekAgent with the API key from the environment variable
deepseek_api_key = os.getenv('deepseek_api_key')
agent = DeepSeekAgent(api_key=deepseek_api_key)

"""

class InputData(BaseModel):
    user_input: str

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}

@app.post("/process-input")
async def process_input(data: InputData):
    response = agent.main(data.user_input)
    return {"response": response}



