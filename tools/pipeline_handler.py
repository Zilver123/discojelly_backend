import logging
from typing import List, Dict, Optional
from fastapi import UploadFile
from tools.scrape_url import scrape_url
from tools.analyze_media import analyze_media
from tools.generate_storyboard import generate_storyboard
from .media_processor import MediaProcessor
from .video_processor import VideoProcessor
import json

logger = logging.getLogger(__name__)

class PipelineHandler:
    def __init__(self):
        self.media_processor = MediaProcessor()
        self.video_processor = VideoProcessor(self.media_processor.file_manager.upload_dir)

    async def handle_input_pipeline(
        self,
        product_url: Optional[str],
        creative_prompt: str,
        media: Optional[List[UploadFile]]
    ) -> Dict:
        """Handle the input pipeline from product URL/media to storyboard."""
        try:
            logger.info(f"Starting input pipeline - URL: {product_url}, Prompt: {creative_prompt}")
            
            # Step 1: Scrape product info if URL provided
            product_data = {}
            if product_url:
                logger.info(f"Scraping URL: {product_url}")
                product_data = scrape_url(product_url)
            
            # Step 2: Process all media
            media_files, media_json = await self._process_all_media(media, product_data)
            
            # Step 3: Generate storyboard
            storyboard_data = await self._generate_storyboard(
                creative_prompt,
                product_data,
                media_files,
                media_json
            )
            
            return storyboard_data
            
        except Exception as e:
            logger.error(f"Error in input pipeline: {str(e)}", exc_info=True)
            raise

    async def handle_video_pipeline(self, storyboard: str) -> str:
        """Handle the video rendering pipeline."""
        try:
            logger.info("Starting video rendering pipeline")
            
            # Step 1: Process media for video
            media_files, temp_files = self.video_processor.process_media_for_video(
                storyboard,
                self.media_processor
            )
            
            # Step 2: Render video
            video_path = self.video_processor.render_video(
                storyboard,
                media_files,
                temp_files
            )
            
            return video_path
            
        except Exception as e:
            logger.error(f"Error in video pipeline: {str(e)}", exc_info=True)
            raise

    async def _process_all_media(
        self,
        media: Optional[List[UploadFile]],
        product_data: Dict
    ) -> tuple[List[str], List[Dict]]:
        """Process all media files (uploaded and scraped)."""
        # Process uploaded media
        media_files, media_json = await self.media_processor.save_uploaded_media(media)
        
        # Process scraped images
        scraped_image_paths, scraped_media_json = [], []
        if product_data.get("images"):
            scraped_image_paths, scraped_media_json = await self.media_processor.download_scraped_images(
                product_data["images"]
            )
        
        # Combine all media
        all_media_json = media_json + scraped_media_json
        all_media_paths = media_files + scraped_image_paths
        
        # Analyze media
        if all_media_paths:
            media_descriptions = analyze_media(all_media_paths)
            all_media_json = self.media_processor.update_media_descriptions(
                all_media_json,
                media_descriptions
            )
        
        # Deduplicate media
        deduped_media = self.media_processor.deduplicate_media(all_media_json)
        
        return all_media_paths, deduped_media

    async def _generate_storyboard(
        self,
        creative_prompt: str,
        product_data: Dict,
        media_files: List[str],
        media_json: List[Dict]
    ) -> Dict:
        """Generate storyboard and prepare response."""
        # Prepare input for storyboard
        input_json = {
            "creative_prompt": creative_prompt,
            "product": {
                "title": product_data.get("title", ""),
                "description": product_data.get("description", "")
            },
            "media": media_json
        }
        logger.info(f"Prepared storyboard input: {json.dumps(input_json, indent=2)}")
        
        # Generate storyboard
        storyboard_json = generate_storyboard(input_json)
        logger.info(f"Generated storyboard: {storyboard_json}")
        
        # Prepare response
        response_data = {
            "product": product_data,
            "creative_prompt": creative_prompt,
            "media_files": media_files,
            "media_descriptions": {},  # media_descriptions are now part of media_json
            "storyboard": storyboard_json
        }
        logger.info(f"Prepared response: {json.dumps(response_data, indent=2)}")
        
        return response_data 