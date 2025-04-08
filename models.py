from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime
from supabase import create_client, Client
import os
from dotenv import load_dotenv

load_dotenv()

class Tool(BaseModel):
    id: str
    name: str
    description: str
    json_schema: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class AIAgent(BaseModel):
    id: str
    name: str
    description: Optional[str]
    category: Optional[str]
    model: Optional[str]
    system_prompt: Optional[str]
    template: Optional[str]
    resources: Optional[Dict[str, Any]]
    chat_history: Optional[Dict[str, Any]]
    tool_ids: Optional[List[str]]
    tools: Optional[Dict[str, Any]]
    creator_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    capabilities: Optional[Dict[str, Any]]

def get_supabase_client() -> Client:
    supabase_url = os.getenv('SUPABASE_URL')
    supabase_key = os.getenv('SUPABASE_KEY')
    
    if not supabase_url or not supabase_key:
        raise ValueError("Supabase credentials not found in environment variables")
    
    return create_client(supabase_url, supabase_key) 