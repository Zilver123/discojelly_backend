from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api"
    PROJECT_NAME: str = "DiscoJelly Backend"
    
    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = [
        "https://discojelly-frontend.onrender.com",
        "http://localhost:3000",  # For local development
        "http://localhost:5173",  # For local development with Vite
    ]
    
    # File Settings
    UPLOAD_DIR: str = "uploads"
    FILE_CLEANUP_INTERVAL: int = 900  # 15 minutes
    FILE_MAX_AGE: int = 3600  # 1 hour
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings() 