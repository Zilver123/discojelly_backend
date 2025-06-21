from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # These variables should be set in a .env file
    # SUPABASE_URL="YOUR_SUPABASE_URL"
    # SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
    supabase_url: Optional[str] = None
    supabase_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    backend_cors_origins: Optional[str] = "*"  # Default to allow all for development

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

settings = Settings() 