from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from config import settings
from tools.pipeline_handler import PipelineHandler
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

async def catch_exceptions_middleware(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        logger.error(f"Unhandled error: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

def create_app(lifespan: Optional[Callable] = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title=settings.PROJECT_NAME,
        openapi_url=f"{settings.API_V1_STR}/openapi.json",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Add error handling middleware
    app.middleware("http")(catch_exceptions_middleware)

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