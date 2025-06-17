from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from config import settings
from tools.pipeline_handler import PipelineHandler
from typing import Optional, Callable

def create_app(lifespan: Optional[Callable] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Set up CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.BACKEND_CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
        max_age=settings.FILE_MAX_AGE,
    )

    # Initialize pipeline handler
    pipeline = PipelineHandler()

    # Mount uploads directory
    app.mount("/uploads", StaticFiles(directory=pipeline.media_processor.file_manager.upload_dir), name="uploads")

    # Store pipeline in app state
    app.state.pipeline = pipeline

    return app 