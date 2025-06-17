import json
import logging
import tempfile
import os
import requests
from typing import List, Tuple
from tools.render_video import render_video

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, upload_dir: str):
        self.upload_dir = upload_dir

    def process_media_for_video(
        self,
        storyboard: str,
        media_processor
    ) -> Tuple[List[str], List[str]]:
        """Process media files for video rendering."""
        sb = json.loads(storyboard)
        media_files = []
        temp_files = []
        
        logger.info("Checking media files for video rendering...")
        for item in sb.get("media", []):
            media_path = item["file"]
            if media_path.startswith(("http://", "https://")):
                # Download to temp file
                ext = os.path.splitext(media_path.split("?")[0])[-1] or ".jpg"
                with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=self.upload_dir) as tmp:
                    try:
                        r = requests.get(media_path, timeout=10)
                        tmp.write(r.content)
                        tmp.flush()
                        tmp_path = tmp.name
                        if media_processor.validate_image(tmp_path):
                            media_files.append(tmp_path)
                            temp_files.append(tmp_path)
                        else:
                            logger.warning(f"Downloaded file is not a valid image: {media_path}")
                            os.remove(tmp_path)
                    except Exception as e:
                        logger.error(f"Failed to download {media_path}: {e}")
            else:
                # Local file
                if media_processor.validate_image(media_path):
                    media_files.append(media_path)
                else:
                    logger.warning(f"Local file is not a valid image: {media_path}")
        
        logger.info(f"Final media_files for video: {media_files}")
        return media_files, temp_files

    def render_video(
        self,
        storyboard: str,
        media_files: List[str],
        temp_files: List[str]
    ) -> str:
        """Render video from storyboard and media files."""
        try:
            output_path = os.path.join(self.upload_dir, "output.mp4")
            video_path = render_video(storyboard, media_files, output_path)
            
            # Cleanup temp files
            for f in temp_files:
                try:
                    os.remove(f)
                except Exception as e:
                    logger.error(f"Error cleaning up temp file {f}: {e}")
            
            return video_path
        except Exception as e:
            logger.error(f"Error rendering video: {str(e)}")
            raise 