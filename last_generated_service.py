from service_scaffold import ServiceScaffold
from typing import Any, Dict, List

class GeneratedService(ServiceScaffold):
    def __init__(self):
        super().__init__(
            name="Instagram Post Generator",
            description="Generates creative Instagram posts for cafes"
        )
        
    def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validates the input data.
        The input must contain 'cafe_name', 'cafe_description', and 'target_audience' fields.
        """
        return all(key in input_data for key in ['cafe_name', 'cafe_description', 'target_audience'])
        
    def validate_output(self, output_data: Any) -> bool:
        """
        Validates the output data.
        The output must be a list of 5 dicts, each with 'caption' and 'hashtags' fields.
        """
        if not isinstance(output_data, list) or len(output_data) != 5:
            return False
        for post in output_data:
            if not isinstance(post, dict) or 'caption' not in post or 'hashtags' not in post:
                return False
        return True
        
    def process(self, input_data: Dict[str, Any]) -> List[Dict[str, str]]:
        """
        Generates 5 Instagram posts for the cafe.
        Uses the input fields to create creative captions and relevant hashtags for each post.
        """
        posts = []
        for i in range(5):
            caption = f"Come and enjoy at {input_data['cafe_name']}! {input_data['cafe_description']}"
            hashtags = f"#cafe #coffee #instacafe #{input_data['cafe_name'].replace(' ', '')} #{input_data['target_audience']}"
            posts.append({'caption': caption, 'hashtags': hashtags})
        return posts