import os
import logging
from typing import List, Dict, Tuple
from fastapi import UploadFile
from datetime import datetime, timedelta
import asyncio
from .file_manager import FileManager

logger = logging.getLogger(__name__)

class MediaProcessor:
    def __init__(self, upload_dir: str = "uploads"):
        self.file_manager = FileManager(upload_dir)

    async def save_uploaded_media(self, media: List[UploadFile]) -> Tuple[List[str], List[Dict]]:
        """Save uploaded media files and return paths and metadata."""
        media_files = []
        media_json = []
        
        if not media:
            return media_files, media_json

        for file in media:
            try:
                file_path = file.filename
                content = await file.read()
                if await self.file_manager.save_uploaded_file(file_path, content):
                    media_files.append(file_path)
                    media_json.append({"path": f"uploads/{file.filename}", "description": ""})
            except Exception as e:
                logger.error(f"Error processing media file {file.filename}: {str(e)}")
                raise

        return media_files, media_json

    async def download_scraped_images(self, image_urls: List[str]) -> Tuple[List[str], List[Dict]]:
        """Download scraped images and return paths and metadata."""
        scraped_image_paths = []
        media_json = []

        for i, img_url in enumerate(image_urls):
            try:
                file_path = f"scraped_{i}.jpg"
                if await self.file_manager.download_file(img_url, file_path):
                    scraped_image_paths.append(file_path)
                    media_json.append({"path": img_url, "description": ""})
            except Exception as e:
                logger.error(f"Error downloading image {img_url}: {str(e)}")
                continue

        return scraped_image_paths, media_json

    def validate_image(self, image_path: str) -> bool:
        """Validate if a file is a valid image."""
        return self.file_manager.validate_image(image_path)

    async def cleanup_old_files(self):
        """Clean up files older than 1 hour."""
        while True:
            try:
                now = datetime.now()
                for filename in os.listdir(self.file_manager.upload_dir):
                    filepath = os.path.join(self.file_manager.upload_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if now - file_time > timedelta(hours=1):
                            try:
                                os.remove(filepath)
                                logger.info(f"Cleaned up old file: {filename}")
                            except Exception as e:
                                logger.error(f"Error cleaning up {filename}: {e}")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(900)  # Run cleanup every 15 minutes

    def deduplicate_media(self, media_json: List[Dict]) -> List[Dict]:
        """Remove duplicate media entries."""
        seen = set()
        deduped_media = []
        for m in media_json:
            if m["path"] not in seen:
                deduped_media.append(m)
                seen.add(m["path"])
        return deduped_media

    def update_media_descriptions(self, media_json: List[Dict], media_descriptions: Dict[str, str]) -> List[Dict]:
        """Update media descriptions based on analysis results."""
        for m in media_json:
            desc = ""
            for k, v in media_descriptions.items():
                if m["path"].endswith(os.path.basename(k)) or os.path.basename(m["path"]) in k:
                    desc = v
                    if m["path"].startswith("http"):
                        media_descriptions[m["path"]] = v
                    break
            m["description"] = desc or "Image"
        return media_json 