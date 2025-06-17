from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from typing import List, Optional
from pydantic import BaseModel
import logging
from tools.pipeline_handler import PipelineHandler

logger = logging.getLogger(__name__)

router = APIRouter()

class RenderVideoRequest(BaseModel):
    storyboard: str  # JSON string
    media_files: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "storyboard": '{"scenes": [...]}',
                "media_files": ["path/to/image1.jpg", "path/to/image2.jpg"]
            }
        }

async def get_pipeline() -> PipelineHandler:
    """Dependency to get the pipeline handler from app state."""
    request: Request = Request.get_current()
    return request.app.state.pipeline

@router.post("/input")
async def input_phase(
    product_url: Optional[str] = Form(None, description="URL of the product to scrape"),
    creative_prompt: str = Form(..., description="Creative prompt for storyboard generation"),
    media: Optional[List[UploadFile]] = File(None, description="Media files to process"),
    pipeline: PipelineHandler = Depends(get_pipeline)
):
    """
    Process input data and generate a storyboard.
    
    - **product_url**: Optional URL to scrape product information from
    - **creative_prompt**: Required prompt for storyboard generation
    - **media**: Optional list of media files to process
    """
    try:
        logger.info(f"Processing input - URL: {product_url}, Prompt: {creative_prompt}")
        response_data = await pipeline.handle_input_pipeline(
            product_url,
            creative_prompt,
            media
        )
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    except Exception as e:
        logger.error(f"Error in input_phase: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

@router.post("/render_video")
async def render_video_endpoint(
    req: RenderVideoRequest,
    pipeline: PipelineHandler = Depends(get_pipeline)
):
    """
    Render a video from a storyboard and media files.
    
    - **storyboard**: JSON string containing the storyboard data
    - **media_files**: List of paths to media files to use in the video
    """
    try:
        logger.info("Starting video rendering")
        video_path = await pipeline.handle_video_pipeline(req.storyboard)
        return JSONResponse(
            content={"video_path": video_path},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )
    except Exception as e:
        logger.error(f"Error in render_video_endpoint: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": str(e)},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Headers": "*",
            }
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse(
        content={"status": "healthy"},
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        }
    ) 