from typing import Dict
import openai
import os
import re
import json

# Placeholder for agent tool registration
def function_tool(func):
    return func

@function_tool
def generate_storyboard(input_json: Dict) -> str:
    print("[generate_storyboard] input_json:", json.dumps(input_json, indent=2))
    """Generate a storyboard as strict JSON from product info, media, and creative prompt."""
    try:
        client = openai.OpenAI()
        # Compose a strict prompt for the LLM
        prompt = f'''
You are an expert short-form video marketer. Given the following JSON input, generate a TikTok-style video storyboard as a JSON object with only these fields:
- 'script': Write a short, punchy, conversion-focused TikTok ad script (1-3 sentences max). Use a fun, engaging, and persuasive tone. Do NOT copy or paraphrase the product description. Write as if you are a TikTok influencer trying to sell this product in 10 seconds. Highlight unique features and benefits, and include a call to action. Do not include directions, overlays, or music cues.
- 'media': an ordered list of objects with 'start', 'end', and 'file' referencing the input media. Use all provided images in a logical sequence to match the script. Do not omit any images unless there are more than 10; in that case, use the 10 most relevant.
Do not include any explanation or extra text. Output only valid JSON.
Example input:\n{{"creative_prompt": "10 sec vid", "product": {{"title": "Test Product", "description": "A great product for testing. Soft, cuddly, and perfect for all ages."}}, "media": [{{"path": "/uploads/test1.jpg", "description": "A red plush toy on a white background."}}, {{"path": "/uploads/test2.jpg", "description": "A close-up of the plush toy's face."}}]}}
Example output:\n{{"script": "Meet the Test Product! Soft, cuddly, and perfect for all ages. Grab yours now and snuggle up!", "media": [{{"start": "00:00", "end": "00:05", "file": "/uploads/test1.jpg"}}, {{"start": "00:05", "end": "00:10", "file": "/uploads/test2.jpg"}}]}}
Input:\n{json.dumps(input_json)}
'''
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512
        )
        content = response.choices[0].message.content
        # Try to parse as JSON
        try:
            sb = json.loads(content)
        except Exception:
            sb = None
        # Fallback: catchy TikTok-style script
        product = input_json.get("product", {})
        title = product.get("title", "")
        desc = product.get("description", "")
        fallback_script = f"Meet {title.split(':')[-1].strip()}! {desc.split('.')[0]}. Preorder now and join the cozy revolution!"
        media = input_json.get("media", [])
        n = len(media)
        duration = 10
        per = duration // n if n else 10
        fallback_media = [
            {"start": f"00:{str(i*per).zfill(2)}", "end": f"00:{str((i+1)*per).zfill(2)}", "file": m["path"]}
            for i, m in enumerate(media)
        ]
        if not sb:
            sb = {"script": fallback_script, "media": fallback_media}
        # Post-process: if script is too similar to description or too long, use fallback
        script = sb.get("script", "")
        if (desc.strip() and desc.strip() in script) or len(script) > 220:
            sb["script"] = fallback_script
        # Fallback: if media is missing or empty, use all images
        if not sb.get("media"):
            sb["media"] = fallback_media
        # Fallback: if script is missing, use catchy fallback
        if not sb.get("script"):
            sb["script"] = fallback_script

        # --- Enforce sequential, non-overlapping timings for media ---
        media_list = sb.get("media", [])
        n = len(media_list)
        total_duration = 10
        if n > 0:
            per = total_duration / n
            times = [round(i * per) for i in range(n)] + [total_duration]
            for i, item in enumerate(media_list):
                item["start"] = f"00:{str(times[i]).zfill(2)}"
                item["end"] = f"00:{str(times[i+1]).zfill(2)}"
        sb["media"] = media_list
        # --- End timing enforcement ---

        return json.dumps(sb)
    except Exception as e:
        return json.dumps({"script": f"Error: {e}", "media": []}) 