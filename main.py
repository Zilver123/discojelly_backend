from fastapi import FastAPI, UploadFile, File, Form, Body
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
from pydantic import BaseModel
from .tools.scrape_url import scrape_url
from .tools.analyze_media import analyze_media
from .tools.generate_storyboard import generate_storyboard
from .tools.render_video import render_video
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import requests
import tempfile
import shutil
import json
import cv2

app = FastAPI()

# Mount uploads directory for static file serving
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev, allow all. For prod, restrict this.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/input")
async def input_phase(
    product_url: Optional[str] = Form(None),
    creative_prompt: str = Form(...),
    media: Optional[List[UploadFile]] = File(None)
):
    # Scrape product info if URL provided
    product_data = scrape_url(product_url) if product_url else {}
    # Save uploaded media
    media_files = []
    media_json = []
    if media:
        for file in media:
            file_path = os.path.join(UPLOAD_DIR, file.filename)
            with open(file_path, "wb") as f:
                f.write(await file.read())
            media_files.append(file_path)
            media_json.append({"path": f"uploads/{file.filename}", "description": ""})
    # Add scraped images to media_json (use local download for analysis)
    scraped_image_paths = []
    if product_data.get("images"):
        for i, img_url in enumerate(product_data["images"]):
            # Download image locally for analysis
            local_path = os.path.join(UPLOAD_DIR, f"scraped_{i}.jpg")
            try:
                r = requests.get(img_url, timeout=5)
                with open(local_path, "wb") as f:
                    f.write(r.content)
                scraped_image_paths.append(local_path)
                media_json.append({"path": img_url, "description": ""})
            except Exception:
                pass
    # Analyze all media (uploaded + scraped)
    all_media_paths = media_files + scraped_image_paths
    media_descriptions = analyze_media(all_media_paths) if all_media_paths else {}
    # Fill in media descriptions
    for m in media_json:
        # Try to match by filename or URL ending
        desc = ""
        for k, v in media_descriptions.items():
            if m["path"].endswith(os.path.basename(k)) or os.path.basename(m["path"]) in k:
                desc = v
                # Also map the URL to the description if it's a URL
                if m["path"].startswith("http"):
                    media_descriptions[m["path"]] = v
                break
        m["description"] = desc or "Image"
    # After filling in media descriptions, map each scraped URL to its local file's description
    if product_data.get("images"):
        for i, img_url in enumerate(product_data["images"]):
            local_path = os.path.join(UPLOAD_DIR, f"scraped_{i}.jpg")
            if local_path in media_descriptions:
                media_descriptions[img_url] = media_descriptions[local_path]
    # Remove duplicates by path
    seen = set()
    deduped_media = []
    for m in media_json:
        if m["path"] not in seen:
            deduped_media.append(m)
            seen.add(m["path"])
    # Build input for storyboard (no images field)
    input_json = {
        "creative_prompt": creative_prompt,
        "product": {
            "title": product_data.get("title", ""),
            "description": product_data.get("description", "")
        },
        "media": deduped_media
    }
    # Generate storyboard (strict JSON)
    storyboard_json = generate_storyboard(input_json)
    # --- Combine uploaded and scraped image URLs for media_files ---
    uploaded_files = media_files if media_files else []
    scraped_urls = [m["path"] for m in deduped_media if m["path"].startswith("http")]
    all_media_files = uploaded_files + scraped_urls
    return JSONResponse({
        "product": product_data,
        "creative_prompt": creative_prompt,
        "media_files": all_media_files,
        "media_descriptions": media_descriptions,
        "storyboard": storyboard_json
    })

class RenderVideoRequest(BaseModel):
    storyboard: str  # JSON string
    media_files: List[str]

@app.post("/api/render_video")
async def render_video_endpoint(req: RenderVideoRequest):
    # Parse storyboard JSON
    sb = json.loads(req.storyboard)
    media_files = []
    temp_files = []
    print("[render_video_endpoint] Checking media files for video rendering...")
    for item in sb.get("media", []):
        media_path = item["file"]
        if media_path.startswith("http://") or media_path.startswith("https://"):
            # Download to temp file, strip query params from extension
            ext = os.path.splitext(media_path.split("?")[0])[-1] or ".jpg"
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=UPLOAD_DIR) as tmp:
                try:
                    r = requests.get(media_path, timeout=10)
                    tmp.write(r.content)
                    tmp.flush()
                    # Validate with OpenCV
                    tmp_path = tmp.name
                    img = cv2.imread(tmp_path)
                    print(f"[render_video_endpoint] Downloaded {media_path} -> {tmp_path}, cv2.imread: {img.shape if img is not None else None}")
                    if img is not None:
                        media_files.append(tmp_path)
                        temp_files.append(tmp_path)
                    else:
                        print(f"[render_video_endpoint] Downloaded file is not a valid image: {media_path}")
                        os.remove(tmp_path)
                except Exception as e:
                    print(f"[render_video_endpoint] Failed to download {media_path}: {e}")
        else:
            # Local file
            img = cv2.imread(media_path)
            print(f"[render_video_endpoint] Local file {media_path}, cv2.imread: {img.shape if img is not None else None}")
            if img is not None:
                media_files.append(media_path)
            else:
                print(f"[render_video_endpoint] Local file is not a valid image: {media_path}")
    print(f"[render_video_endpoint] Final media_files for video: {media_files}")
    output_path = os.path.join(UPLOAD_DIR, "output.mp4")
    video_path = render_video(req.storyboard, media_files, output_path)
    # Optionally: cleanup temp files after rendering
    for f in temp_files:
        try:
            os.remove(f)
        except Exception:
            pass
    return JSONResponse({"video_path": video_path}) 