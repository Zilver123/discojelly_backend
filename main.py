from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client, Client
from config import settings
import logging
from datetime import datetime, timedelta
import re
import openai
import json
from typing import Optional
import asyncio
from urllib.parse import urlparse, quote, unquote
import httpx
from fastapi.responses import Response
import base64
from pydantic import BaseModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

# Initialize Supabase client
supabase: Optional[Client] = None

# Initialize OpenAI client
openai_client = None

# Initialize the discovery agent
discovery_agent = None

@app.on_event("startup")
async def startup_event():
    global supabase, openai_client, discovery_agent
    if settings.supabase_url and settings.supabase_key:
        try:
            supabase = create_client(
                settings.supabase_url,
                settings.supabase_key,
            )
            logger.info("Successfully connected to Supabase.")
        except Exception as e:
            logger.error(f"Failed to connect to Supabase: {e}")
            supabase = None
    else:
        logger.warning("Supabase URL or Key not provided. Backend will not connect to the database.")
    
    if settings.openai_api_key:
        try:
            openai_client = openai.OpenAI(api_key=settings.openai_api_key)
            discovery_agent = AIServiceDiscoveryAgent(openai_client)
            logger.info("Successfully connected to OpenAI and initialized discovery agent.")
        except Exception as e:
            logger.error(f"Failed to connect to OpenAI: {e}")
            openai_client = None
            discovery_agent = None
    else:
        logger.warning("OpenAI API Key not provided. Agent discovery will not work.")

# Configure CORS
# Parse CORS origins from environment variable
if settings.backend_cors_origins and settings.backend_cors_origins != "*":
    try:
        origins = json.loads(settings.backend_cors_origins)
    except json.JSONDecodeError:
        origins = ["*"]
else:
    origins = ["*"]  # Allow all origins for development

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple AI Service Discovery Agent
class AIServiceDiscoveryAgent:
    def __init__(self, openai_client):
        self.openai_client = openai_client
        
    async def discover_services(self, query: str) -> list:
        """
        Discover AI services using the agents SDK with web search tools.
        This implementation uses the proper agents library with web search capabilities.
        NO FALLBACK DATA - ONLY REAL WEB SEARCH RESULTS.
        """
        if not self.openai_client:
            logger.warning("OpenAI client not available for discovery")
            return []

        try:
            # Import agents SDK - this is the proper way to use it
            from agents import Agent, Runner, WebSearchTool
            
            # Create agent with web search tool
            agent = Agent(
                name="AI Service Discovery Expert",
                instructions="""You are an expert AI assistant that discovers and analyzes top-tier AI tools and services based on a user's query.

Your main goal is to understand the user's intent and find the most relevant, high-quality, and current AI tools for their specific task.

- If the user asks for a specific type of tool (e.g., "video generation," "image editing," "writing assistant"), focus your search on that category.
- If the user gives a general query (e.g., "help with homework," "marketing tools"), analyze the query to determine the most relevant tool categories and find the best services.
- For each tool, you MUST find the official name, a comprehensive description, how to use it, the official service URL, and relevant tags.

Your response MUST be a valid JSON array of objects, with no other text. Each object should contain:
- name: The official name of the tool.
- description: A detailed description of its features and capabilities.
- how_to_use: Simple steps on how to get started with the tool.
- service_url: The direct URL to the service's website.
- tags: A list of relevant lowercase tags (e.g., ["video", "ai", "writing"]).""",
                tools=[WebSearchTool()]
            )
            
            # Create the search query
            search_query = f"""Find the best AI tools for the following user query: '{query}'

Analyze the user's request and identify the primary task (e.g., video creation, writing, design, productivity).

Search for the top-tier, currently available, and well-reviewed AI services that directly address the user's needs.

For each tool you find, provide comprehensive information including its features, primary use case, and website URL.

Return the results as a valid JSON array with these exact fields: "name", "description", "how_to_use", "service_url", "tags"."""
            
            # Run the agent with web search using the correct Runner.run pattern
            logger.info(f"Starting agents SDK web search for: {query}")
            result = await Runner.run(agent, search_query)
            
            # Extract the response content
            content = result.final_output
            logger.info(f"Agents SDK response received, length: {len(content)}")
            logger.debug(f"Raw agents response: {content[:1000]}...")
            
            # Try to extract JSON from the response
            try:
                # Look for JSON array in the response
                start_idx = content.find('[')
                end_idx = content.rfind(']') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = content[start_idx:end_idx]
                    discovered_services = json.loads(json_str)
                    
                    if isinstance(discovered_services, list):
                        logger.info(f"Successfully discovered {len(discovered_services)} services via agents SDK web search")
                        return discovered_services
                    else:
                        logger.error("Agents SDK response was not a list")
                        return []
                else:
                    logger.error("No JSON array found in agents SDK response")
                    logger.debug(f"Full response: {content}")
                    return []
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse agents SDK response as JSON: {e}")
                logger.debug(f"Raw agents response: {content}")
                return []

        except ImportError as e:
            logger.error(f"Failed to import agents module: {e}")
            return []
        except Exception as e:
            logger.error(f"CRITICAL: Agents SDK discovery failed: {e}", exc_info=True)
            return []

# AI-powered query understanding system
class QueryUnderstanding:
    def __init__(self):
        # Task-to-keywords mapping for common AI use cases
        self.task_mappings = {
            # Image generation tasks
            "product marketing images": ["design", "graphics", "social media", "marketing", "image"],
            "logo creation": ["design", "logo", "branding", "graphics"],
            "social media graphics": ["design", "social media", "graphics", "marketing"],
            "product photos": ["image", "photo", "product", "marketing"],
            
            # Video tasks
            "video creation": ["video", "generation", "content"],
            "marketing videos": ["video", "marketing", "content", "promotional"],
            "explainer videos": ["video", "explanation", "educational"],
            
            # Writing tasks
            "content writing": ["writing", "content", "copywriting"],
            "blog posts": ["writing", "blog", "content"],
            "marketing copy": ["writing", "marketing", "copywriting"],
            "product descriptions": ["writing", "product", "description"],
            
            # Productivity tasks
            "note taking": ["productivity", "notes", "organization"],
            "project management": ["productivity", "management", "organization"],
            "documentation": ["writing", "documentation", "productivity"],
            
            # Creative tasks
            "art creation": ["art", "creative", "design"],
            "illustrations": ["art", "illustration", "design"],
            "creative writing": ["writing", "creative", "storytelling"],
        }
        
        # Common patterns for natural language queries
        self.query_patterns = [
            r"I want to (?:create|generate|make|build) (.+)",
            r"I need (?:to create|to generate|to make|to build) (.+)",
            r"Help me (?:create|generate|make|build) (.+)",
            r"Looking for (?:a tool|an app|a service) to (.+)",
            r"Best (?:AI|tool|service) for (.+)",
        ]
    
    def understand_query(self, query: str) -> dict:
        """
        Understands abstract queries and maps them to relevant search terms.
        Returns enhanced search terms and confidence score.
        """
        query_lower = query.lower().strip()
        
        # Check for exact task matches
        for task, keywords in self.task_mappings.items():
            if task in query_lower:
                return {
                    "original_query": query,
                    "enhanced_terms": keywords,
                    "confidence": 0.9,
                    "task_type": task
                }
        
        # Check for pattern matches
        for pattern in self.query_patterns:
            match = re.search(pattern, query_lower)
            if match:
                extracted_task = match.group(1).strip()
                # Find the best matching task
                best_match = self._find_best_task_match(extracted_task)
                if best_match:
                    return {
                        "original_query": query,
                        "enhanced_terms": self.task_mappings[best_match],
                        "confidence": 0.7,
                        "task_type": best_match
                    }
        
        # Fallback: extract key terms from the query
        key_terms = self._extract_key_terms(query_lower)
        return {
            "original_query": query,
            "enhanced_terms": key_terms,
            "confidence": 0.5,
            "task_type": "general"
        }
    
    def _find_best_task_match(self, extracted_task: str) -> str:
        """Find the best matching task from our mappings."""
        best_match = None
        best_score = 0
        
        for task in self.task_mappings.keys():
            # Simple word overlap scoring
            task_words = set(task.split())
            extracted_words = set(extracted_task.split())
            overlap = len(task_words.intersection(extracted_words))
            score = overlap / max(len(task_words), len(extracted_words))
            
            if score > best_score and score > 0.3:  # Minimum threshold
                best_score = score
                best_match = task
        
        return best_match
    
    def _extract_key_terms(self, query: str) -> list:
        """Extract meaningful terms from a query."""
        # Remove common words
        stop_words = {"i", "want", "to", "create", "generate", "make", "build", "need", "help", "me", "for", "a", "an", "the", "and", "or", "but", "in", "on", "at", "with", "by", "from", "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can"}
        
        words = query.split()
        key_terms = [word for word in words if word.lower() not in stop_words and len(word) > 2]
        
        return key_terms

# Initialize query understanding
query_understanding = QueryUnderstanding()

# Sample data to be added
sample_services = [
    {
        "name": "Synthesia",
        "description": "Create professional AI videos from text in over 120 languages.",
        "image_url": "https://via.placeholder.com/150x150/6366f1/ffffff?text=Synthesia",
        "how_to_use": "Type your script, choose an avatar, and generate your video. No actors, cameras, or mics needed.",
        "service_url": "https://www.synthesia.io/",
        "tags": ["video", "generation", "avatar"],
    },
    {
        "name": "Midjourney",
        "description": "An independent research lab exploring new mediums of thought and expanding the imaginative powers of the human species.",
        "image_url": "https://via.placeholder.com/150x150/10b981/ffffff?text=Midjourney",
        "how_to_use": "Use text prompts to create stunningly rich and artistic images.",
        "service_url": "https://www.midjourney.com/",
        "tags": ["image", "generation", "art", "creative"],
    },
    {
        "name": "Canva Magic Studio",
        "description": "All the power of AI, all in one place. Create engaging content and stunning designs, fast.",
        "image_url": "https://via.placeholder.com/150x150/06b6d4/ffffff?text=Canva",
        "how_to_use": "Use simple text prompts to generate designs, edit photos, and create presentations.",
        "service_url": "https://www.canva.com/magic-studio/",
        "tags": ["design", "graphics", "social media", "presentation"],
    },
    {
        "name": "Rytr",
        "description": "An AI writing assistant that helps you create high-quality content, in just a few seconds, at a fraction of the cost!",
        "how_to_use": "Choose a use-case, enter some context, and let Rytr write for you.",
        "service_url": "https://rytr.me/",
        "tags": ["writing", "copywriting", "content", "assistant"],
    },
    {
        "name": "Notion AI",
        "description": "Access the limitless power of AI, right inside Notion. Work faster. Write better. Think bigger.",
        "how_to_use": "Use AI to summarize notes, brainstorm ideas, and organize your information within your Notion workspace.",
        "service_url": "https://www.notion.so/product/ai",
        "tags": ["productivity", "notes", "knowledge management", "writing"],
    }
]

@app.get("/")
def read_root():
    return {"message": "DiscoJelly Backend is running!"}

@app.get("/search")
async def search_services(query: str):
    """
    Performs AI-powered search that understands abstract queries and maps them to relevant services.
    Automatically triggers agent discovery if no good local matches are found.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not connected")

    # Step 1: Understand the query using AI
    query_analysis = query_understanding.understand_query(query)
    enhanced_terms = query_analysis["enhanced_terms"]
    
    logger.info(f"Query analysis: {query_analysis}")
    
    # Step 2: Try full-text search with enhanced terms
    local_results = []
    try:
        search_query = " | ".join(enhanced_terms)
        response = supabase.table('services').select('*').text_search('fts', search_query).execute()
        local_results = response.data
        search_method = "full_text"
    except Exception as e:
        logger.info(f"Full-text search failed, falling back to LIKE search: {e}")
        try:
            # Fallback to simple LIKE search with enhanced terms
            like_conditions = []
            for term in enhanced_terms:
                like_conditions.append(f'name.ilike.%{term}%,description.ilike.%{term}%')
            
            response = supabase.table('services').select('*').or_(','.join(like_conditions)).execute()
            local_results = response.data
            search_method = "like_search"
        except Exception as fallback_error:
            logger.error(f"LIKE search also failed: {fallback_error}")
            search_method = "fallback"
    
    # Step 3: Check if we need to trigger agent discovery
    discovered_results = []
    
    # ALWAYS trigger agent discovery for video generation queries to get real web search results
    should_trigger_discovery = (
        len(local_results) < 5 or  # If we have fewer than 5 good local matches
        any(term in query.lower() for term in ['video', 'generate', 'create', 'make']) or  # Video-related queries
        query_analysis["task_type"] in ["video creation", "marketing videos", "explainer videos"]  # Video tasks
    )
    
    logger.info(f"Discovery trigger logic: should_trigger_discovery={should_trigger_discovery}, local_results={len(local_results)}, query='{query}', task_type={query_analysis['task_type']}")
    
    if should_trigger_discovery:
        logger.info(f"Triggering agent discovery for query: '{query}' (local results: {len(local_results)})")
        
        if discovery_agent:
            try:
                logger.info("Calling discovery_agent.discover_services...")
                discovered_results = await discovery_agent.discover_services(query)
                logger.info(f"Agent discovered {len(discovered_results)} additional services: {json.dumps(discovered_results)[:500]}")
            except Exception as e:
                logger.error(f"Error during agent discovery: {e}")
        else:
            logger.warning("Discovery agent not available")
    else:
        logger.info(f"Skipping agent discovery - sufficient local results found ({len(local_results)})")
    
    # Step 4: Combine results and process images consistently
    all_results = local_results + discovered_results
    
    async with httpx.AsyncClient() as client:
        for service in all_results:
            if 'service_url' in service and service['service_url'] and not service.get('image_url', '').startswith('data:'):
                try:
                    domain = service['service_url'].split('//')[-1].split('/')[0]
                    favicon_url = f"https://www.google.com/s2/favicons?domain={domain}&sz=128"
                    
                    resp = await client.get(favicon_url, follow_redirects=True, timeout=10.0)
                    resp.raise_for_status()
                    
                    content_type = resp.headers.get('content-type', 'image/png')
                    encoded_string = base64.b64encode(resp.content).decode('utf-8')
                    service['image_url'] = f"data:{content_type};base64,{encoded_string}"
                except Exception as e:
                    logger.warning(f"Could not process image for {service.get('name')}: {e}")
                    service['image_url'] = None # Ensure it's null if fetching fails

    # Step 5: Return final response
    response_data = {
        "query": query,
        "results": all_results,
        "query_analysis": query_analysis,
        "search_method": search_method,
        "local_results_count": len(local_results),
        "discovered_results_count": len(discovered_results),
        "note": f"Found {len(local_results)} local and {len(discovered_results)} discovered services"
    }
    
    logger.info(f"Final response: {json.dumps(response_data)}")
    return response_data

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

class FeedbackSubmission(BaseModel):
    content: str

@app.post("/feedback")
async def submit_feedback(feedback: FeedbackSubmission):
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not connected")
    
    try:
        supabase.table('feedback').insert({
            "content": feedback.content,
        }).execute()
        
        logger.info("New feedback submitted")
        return {"message": "Feedback submitted successfully!"}
        
    except Exception as e:
        logger.error(f"Error submitting feedback: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit feedback.")

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

@app.delete("/clear")
async def clear_database():
    """
    Endpoint to clear all services from the database.
    This is for testing purposes and should not be used in production.
    """
    if not supabase:
        raise HTTPException(status_code=503, detail="Database connection not available")
    try:
        # Delete all services
        response = supabase.table("services").delete().neq("id", 0).execute()
        return {"message": "Database cleared successfully", "deleted_count": len(response.data)}
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {str(e)}")

# To use text_search, we need a full-text search vector.
# The user should run this SQL in their Supabase dashboard once.
#
# alter table services add column fts tsvector
#   generated always as (to_tsvector('english', name || ' ' || description)) stored;
#
# create index services_fts on services using gin (fts); 