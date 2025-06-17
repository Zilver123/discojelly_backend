import asyncio
import logging
from contextlib import asynccontextmanager
from .app import create_app
from .routes import router
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    """Lifespan event handler."""
    # Startup
    pipeline = app.state.pipeline
    asyncio.create_task(pipeline.media_processor.cleanup_old_files())
    logger.info("Application startup complete")
    yield
    # Shutdown
    logger.info("Application shutdown")

# Create FastAPI app with lifespan
app = create_app(lifespan=lifespan)

# Include routes
app.include_router(router, prefix=settings.API_V1_STR) 