from service_scaffold import ServiceScaffold
from typing import Any, Dict, List
import json

class InstagramPostService(ServiceScaffold):
    def __init__(self):
        super().__init__(
            name="Instagram Post Generator",
            description="Generates 5 Instagram posts for a cafe with captions and hashtags"
        )
    
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        required_fields = ["cafe_name", "cafe_description", "target_audience"]
        return all(field in input_data for field in required_fields)
    
    def validate_output(self, output_data: List[Dict[str, str]]) -> bool:
        if not isinstance(output_data, list) or len(output_data) != 5:
            return False
        required_fields = ["caption", "hashtags"]
        return all(
            isinstance(post, dict) and 
            all(field in post for field in required_fields)
            for post in output_data
        )
    
    def process(self, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        prompt = f"""
        Generate 5 Instagram posts for a cafe with the following details:
        Cafe Name: {input_data['cafe_name']}
        Description: {input_data['cafe_description']}
        Target Audience: {input_data['target_audience']}
        
        For each post, provide:
        1. A creative caption
        2. Relevant hashtags
        
        Format the response as a JSON array of objects with 'caption' and 'hashtags' fields.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a social media expert specializing in cafe marketing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        try:
            posts = json.loads(response.choices[0].message.content)
            return posts
        except json.JSONDecodeError:
            # Fallback in case the response isn't valid JSON
            return [
                {
                    "caption": "Error generating posts. Please try again.",
                    "hashtags": "#error #tryagain"
                }
            ] * 5 