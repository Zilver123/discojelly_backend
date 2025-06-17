import os
import logging
import tempfile
import requests
from typing import Tuple
from datetime import datetime, timedelta
import asyncio
import cv2

logger = logging.getLogger(__name__)

class FileManager:
    def __init__(self, upload_dir: str = "uploads"):
        self.upload_dir = upload_dir
        os.makedirs(upload_dir, exist_ok=True)

    async def save_uploaded_file(self, file_path: str, content: bytes) -> bool:
        """Save a file to the upload directory."""
        try:
            full_path = os.path.join(self.upload_dir, file_path)
            with open(full_path, "wb") as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error saving file {file_path}: {e}")
            return False

    async def download_file(self, url: str, file_path: str) -> bool:
        """Download a file from a URL."""
        try:
            response = requests.get(url, timeout=10)
            return await self.save_uploaded_file(file_path, response.content)
        except Exception as e:
            logger.error(f"Error downloading file from {url}: {e}")
            return False

    def validate_image(self, image_path: str) -> bool:
        """Validate if a file is a valid image using OpenCV."""
        try:
            img = cv2.imread(image_path)
            return img is not None
        except Exception as e:
            logger.error(f"Error validating image {image_path}: {e}")
            return False

    async def cleanup_old_files(self, max_age_hours: int = 1):
        """Clean up files older than max_age_hours."""
        while True:
            try:
                now = datetime.now()
                for filename in os.listdir(self.upload_dir):
                    filepath = os.path.join(self.upload_dir, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if now - file_time > timedelta(hours=max_age_hours):
                            try:
                                os.remove(filepath)
                                logger.info(f"Cleaned up old file: {filename}")
                            except Exception as e:
                                logger.error(f"Error cleaning up {filename}: {e}")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            await asyncio.sleep(900)  # Run cleanup every 15 minutes

    def create_temp_file(self, suffix: str = ".jpg") -> Tuple[str, str]:
        """Create a temporary file and return its path and name."""
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=self.upload_dir)
        return temp_file.name, os.path.basename(temp_file.name)

    def get_file_extension(self, url: str) -> str:
        """Get file extension from URL, defaulting to .jpg."""
        return os.path.splitext(url.split("?")[0])[-1] or ".jpg"

    def ensure_directory(self, path: str):
        """Ensure a directory exists."""
        os.makedirs(path, exist_ok=True) 