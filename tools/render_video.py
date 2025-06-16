from typing import List
import re
import os
import logging
import json
from PIL import Image
import cv2
import numpy as np
import subprocess

# Placeholder for agent tool registration
def function_tool(func):
    return func

@function_tool
def render_video(storyboard: str, media_files: List[str], output_path: str) -> str:
    """Render a video from storyboard (JSON) and media files using OpenCV."""
    try:
        print(f"[render_video] Storyboard (JSON):\n{storyboard}")
        print(f"[render_video] Media files: {media_files}")
        data = json.loads(storyboard)
        media_list = data.get("media", [])
        if not media_files or not media_list:
            raise ValueError("No media files provided")
        # --- Set output video size to 720x1280 (9:16) ---
        width, height = 720, 1280
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        # Calculate total duration from storyboard
        if media_list:
            last_end = media_list[-1]["end"]
            total_duration = int(last_end.split(":")[0])*60 + int(last_end.split(":")[1])
        else:
            total_duration = 0
        fps = 1
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        current_frame = 0
        # Build a mapping from storyboard file path to local file path
        storyboard_files = [item["file"] for item in media_list]
        file_map = {storyboard_files[i]: media_files[i] for i in range(len(media_files))}
        for idx, item in enumerate(media_list):
            media_path = file_map.get(item["file"], item["file"])
            start = item["start"]
            end = item["end"]
            start_s = int(start.split(":")[0])*60 + int(start.split(":")[1])
            end_s = int(end.split(":")[0])*60 + int(end.split(":")[1])
            duration = max(1, end_s - start_s)
            print(f"[render_video] Adding {media_path} for {duration} seconds")
            frame = cv2.imread(media_path)
            if frame is None:
                print(f"[render_video] Failed to read {media_path}, skipping")
                continue
            # --- Resize and pad to 720x1280 (centered, black bars) ---
            h, w = frame.shape[:2]
            scale = min(width / w, height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))
            padded = np.zeros((height, width, 3), dtype=np.uint8)
            y_off = (height - new_h) // 2
            x_off = (width - new_w) // 2
            padded[y_off:y_off+new_h, x_off:x_off+new_w] = resized
            # Pad with black frames if there's a gap
            while current_frame < start_s:
                out.write(np.zeros((height, width, 3), dtype=np.uint8))
                current_frame += 1
            for _ in range(duration):
                out.write(padded)
                current_frame += 1
        # Pad to total duration with black frames if needed
        while current_frame < total_duration:
            out.write(np.zeros((height, width, 3), dtype=np.uint8))
            current_frame += 1
        out.release()
        print(f"[render_video] Video written to {output_path}")

        # --- Post-process: re-encode to H.264 for browser compatibility ---
        h264_path = output_path.replace('.mp4', '_h264.mp4')
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-i', output_path,
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
            h264_path
        ]
        try:
            print(f"[render_video] Running ffmpeg: {' '.join(ffmpeg_cmd)}")
            subprocess.run(ffmpeg_cmd, check=True)
            # Replace original with h264 version
            os.replace(h264_path, output_path)
            print(f"[render_video] Re-encoded video to H.264 at {output_path}")
        except Exception as e:
            print(f"[render_video] ffmpeg re-encode failed: {e}")
        return output_path
    except Exception as e:
        print(f"[render_video] Exception: {e}")
        with open(output_path, "wb") as f:
            f.write(b"00")
        return output_path 