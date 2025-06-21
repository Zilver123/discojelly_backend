from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from supabase.lib.client_options import ClientOptions
from config import settings
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Supabase client
supabase: Client | None = None

@app.on_event("startup")
async def startup_event():
    global supabase
    if settings.supabase_url and settings.supabase_key:
        try:
            options = ClientOptions()
            # httpx by default picks up proxy settings from env vars.
            # We explicitly disable it here.
            options.postgrest_client_options.proxies = {}
            supabase = create_client(
                settings.supabase_url,
                settings.supabase_key,
                options=options,
            )
            logger.info("Successfully connected to Supabase.")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            supabase = None
    else:
        logger.warning("Supabase URL or Key not provided. Backend will not connect to the database.")

app = FastAPI()

# Configure CORS
origins = [
    "*" # Allow all origins for development
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sample data to be added
sample_services = [
    {
        "name": "Synthesia",
        "description": "Create professional AI videos from text in over 120 languages.",
        "image_url": "https://assets-global.website-files.com/612640a65f42621435237a33/6215316258df634305537553_Primary_lockup_Light.svg",
        "how_to_use": "Type your script, choose an avatar, and generate your video. No actors, cameras, or mics needed.",
        "service_url": "https://www.synthesia.io/",
        "tags": ["video", "generation", "avatar"],
    },
    {
        "name": "Midjourney",
        "description": "An independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.",
        "image_url": "https://media.nextspy.com/uploads/2023/04/midjourney-logo.png",
        "how_to_use": "Use text prompts to create stunningly rich and artistic images.",
        "service_url": "https://www.midjourney.com/",
        "tags": ["image", "generation", "art", "creative"],
    },
    {
        "name": "Canva Magic Studio",
        "description": "All the power of AI, all in one place. Create engaging content and stunning designs, fast.",
        "image_url": "https://static.canva.com/static/images/logo_wordmark_logotype_dark.svg",
        "how_to_use": "Use simple text prompts to generate designs, edit photos, and create presentations.",
        "service_url": "https://www.canva.com/magic-studio/",
        "tags": ["design", "graphics", "social media", "presentation"],
    },
    {
        "name": "Rytr",
        "description": "An AI writing assistant that helps you create high-quality content, in just a few seconds, at a fraction of the cost!",
        "image_url": "https://rytr.me/assets/images/rytr-logo-dark.svg",
        "how_to_use": "Choose a use-case, enter some context, and let Rytr write for you.",
        "service_url": "https://rytr.me/",
        "tags": ["writing", "copywriting", "content", "assistant"],
    },
    {
        "name": "Notion AI",
        "description": "Access the limitless power of AI, right inside Notion. Work faster. Write better. Think bigger.",
        "image_url": "https://www.notion.so/images/meta/default.png",
        "how_to_use": "Use AI to summarize notes, brainstorm ideas, and organize your information within your Notion workspace.",
        "service_url": "https://www.notion.so/product/ai",
        "tags": ["productivity", "notes", "knowledge management", "writing"],
    }
]

@app.get("/")
def read_root():
    return {"message": "DiscoJelly Backend is running!"}

@app.get("/search")
def search_services(query: str):
    """
    Performs a full-text search on the 'name' and 'description' fields.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not connected")

    # The query is tokenized and joined with '|' for OR search
    search_query = " | ".join(query.split())
    
    try:
        response = supabase.table('services').select('*').text_search('fts', search_query, config='english').execute()
        return {"query": query, "results": response.data}
    except Exception as e:
        logger.error(f"Error during search for query '{query}': {e}")
        raise HTTPException(status_code=500, detail="Failed to search for services")


@app.get("/service/{service_id}")
def get_service(service_id: int):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        response = supabase.table('services').select('*').eq('id', service_id).single().execute()
        if not response.data:
             raise HTTPException(status_code=404, detail=f"Service with id {service_id} not found")
        return response.data
    except Exception as e:
        logger.error(f"Error fetching service with id {service_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch service")

@app.get("/seed")
async def seed_database():
    """
    Endpoint to seed the database with sample services.
    This is for demonstration purposes and should not be used in production.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available")
    try:
        # Check if data already exists to avoid duplicates
        response = supabase.table("services").select("id").limit(1).execute()
        if response.data:
            return {"message": "Database has already been seeded."}

        # Insert sample data
        insert_response = supabase.table("services").insert(sample_services).execute()

        return {"message": "Database seeded successfully", "data": insert_response.data}
    except Exception as e:
        logger.error(f"Error seeding database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to seed database: {str(e)}")

# To use text_search, we need a full-text search vector.
# The user should run this SQL in their Supabase dashboard once.
#
# alter table services add column fts tsvector
#   generated always as (to_tsvector('english', name || ' ' || description)) stored;
#
# create index services_fts on services using gin (fts); 