from typing import List, Dict
import os
import openai
import base64

# Placeholder for agent tool registration
def function_tool(func):
    return func

@function_tool
def analyze_media(media_paths: List[str]) -> Dict[str, str]:
    """Analyze each media file and return a description using OpenAI Vision (gpt-4o)."""
    client = openai.OpenAI()
    results = {}
    for path in media_paths:
        try:
            with open(path, "rb") as f:
                img_bytes = f.read()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "user", "content": [
                            {"type": "text", "text": "Describe this image for a marketing video."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                        ]}
                    ],
                    max_tokens=256
                )
                desc = response.choices[0].message.content
                results[path] = desc
        except Exception as e:
            results[path] = f"Error analyzing {path}: {e}"
    return results 