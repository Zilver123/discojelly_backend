from service_scaffold import ServiceScaffold
import openai
import json
import os
from typing import Dict, Any

class ServiceBuilder:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    def build_service(self, query: str) -> ServiceScaffold:
        """
        Build a service based on the user query using OpenAI.
        """
        prompt = f"""
        Create a Python service class that inherits from ServiceScaffold to handle the following query: "{query}"
        
        Requirements:
        1. The class must be named 'GeneratedService'
        2. Must implement all required methods: __init__, validate_input, validate_output, and process
        3. Must include proper type hints and docstrings
        4. The process method must return a list of 5 dicts, each with 'caption' and 'hashtags' fields (both strings)
        5. The validate_output method must check that the output is a list of 5 dicts, each with 'caption' and 'hashtags' fields (both strings)
        6. The validate_input method must check for 'cafe_name', 'cafe_description', and 'target_audience' in the input dict
        7. The process method should use the input fields to generate creative captions and relevant hashtags for each post
        
        Example format:
        ```python
        from service_scaffold import ServiceScaffold
        from typing import Any, Dict, List
        
        class GeneratedService(ServiceScaffold):
            def __init__(self):
                super().__init__(
                    name="Service Name",
                    description="Service Description"
                )
            
            def validate_input(self, input_data: Dict[str, Any]) -> bool:
                # Implementation
                pass
            
            def validate_output(self, output_data: Any) -> bool:
                # Implementation
                pass
            
            def process(self, input_data: Dict[str, Any]) -> Any:
                # Implementation
                pass
        ```
        
        Return only the Python code, no explanations or markdown formatting.
        """
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an expert Python developer specializing in creating AI-powered services. Return only valid Python code."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        
        # Extract the generated code and clean it up
        generated_code = response.choices[0].message.content.strip()
        if generated_code.startswith("```python"):
            generated_code = generated_code[9:]
        if generated_code.endswith("```"):
            generated_code = generated_code[:-3]
        generated_code = generated_code.strip()
        
        # Always save the generated code for debugging
        with open("last_generated_service.py", "w") as debug_f:
            debug_f.write(generated_code)
        
        # Create a temporary file to store the generated service
        temp_file = "temp_service.py"
        with open(temp_file, "w") as f:
            f.write(generated_code)
        
        # Import the generated service
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("temp_service", temp_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            service = module.GeneratedService()
            return service
        except Exception as e:
            print(f"Error creating service: {e}")
            print("Generated code:")
            print(generated_code)
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file):
                os.remove(temp_file) 