from typing import Optional
import os

from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zilver123.github.io"],  # Allow specific origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

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
    response = agent.process_input(data.user_input)
    return {"response": response}



