# DiscoJelly_backend

An AI-powered video generation system that creates TikTok-style marketing videos from product information and media.

## Features

- Product information scraping from URLs
- AI-powered media analysis
- Automatic storyboard generation
- Video rendering with transitions
- Support for both uploaded and scraped media
- TikTok-style 9:16 aspect ratio output

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create an uploads directory:
   ```bash
   mkdir uploads
   ```
4. Set up your OpenAI API key:
   ```bash
   export OPENAI_API_KEY='your-api-key-here'
   ```

## Running the Server

```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /api/input
Process product information and media to generate a storyboard.

Request body (multipart/form-data):
- `product_url` (optional): URL of the product to scrape
- `creative_prompt` (required): Creative direction for the video
- `media` (optional): List of media files to include

Response:
```json
{
  "product": {
    "title": "Product Title",
    "description": "Product Description",
    "images": ["image_url1", "image_url2"]
  },
  "creative_prompt": "Your creative prompt",
  "media_files": ["file1.jpg", "file2.jpg"],
  "media_descriptions": {
    "file1.jpg": "AI-generated description",
    "file2.jpg": "AI-generated description"
  },
  "storyboard": {
    "script": "Video script",
    "media": [
      {
        "start": "00:00",
        "end": "00:05",
        "file": "file1.jpg"
      }
    ]
  }
}
```

### POST /api/render_video
Render a video from a storyboard and media files.

Request body:
```json
{
  "storyboard": "JSON string of storyboard",
  "media_files": ["file1.jpg", "file2.jpg"]
}
```

Response:
```json
{
  "video_path": "path/to/output.mp4"
}
```

## Technical Details

- Built with FastAPI for high-performance API endpoints
- Uses OpenAI's GPT-4 Vision for media analysis
- OpenCV for video rendering
- BeautifulSoup4 for web scraping
- Supports various image formats
- Outputs H.264 encoded MP4 files for broad compatibility

## Error Handling

The API will return appropriate HTTP status codes:
- 400: Invalid input
- 500: Server error

## Notes

- The system automatically handles image resizing and aspect ratio maintenance
- Videos are rendered in 720x1280 (9:16) format
- Media files are temporarily stored in the `uploads` directory
- The system supports both local file uploads and remote URLs