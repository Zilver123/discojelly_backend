import replicate
from models import get_supabase_client
from functools import lru_cache
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

@lru_cache(maxsize=100)
def get_tool_schema(function_name: str) -> Optional[Dict[str, Any]]:
    """Get the tool schema from the database with caching."""
    supabase = get_supabase_client()
    try:
        response = supabase.table('tools').select('json_schema').eq('name', function_name).execute()
        if not response.data:
            return None
        return response.data[0]['json_schema']
    except Exception as e:
        print(f"Error fetching tool schema: {str(e)}")
        return None

def get_parameter_type(function_name: str, param_name: str) -> Optional[str]:
    """Get the expected type for a parameter from the tools configuration."""
    supabase = get_supabase_client()
    response = supabase.table('tools').select('json_schema').eq('name', function_name).execute()
    if not response.data:
        return None
        
    json_schema = response.data[0]['json_schema']
    properties = json_schema.get('properties', {})
    if param_name in properties:
        return properties[param_name].get('type')
    return None

def convert_value(value: Any, expected_type: str) -> Any:
    """Convert a value to the expected type with improved error handling."""
    if value is None:
        return value
    
    try:
        if expected_type == "integer":
            return int(value)
        elif expected_type == "number":
            return float(value)
        elif expected_type == "boolean":
            if isinstance(value, str):
                return value.lower() == 'true'
            return bool(value)
        elif expected_type == "string":
            return str(value)
        return value
    except (ValueError, TypeError) as e:
        print(f"Error converting value {value} to type {expected_type}: {str(e)}")
        return value

def generate(model: str, args: Dict[str, Any]) -> Any:
    """Generate output using the Replicate API with improved error handling."""
    try:
        # Get the function name from the database
        supabase = get_supabase_client()
        response = supabase.table('tools').select('name').eq('model', model).execute()
        
        if not response.data:
            raise ValueError(f"No tool found for model {model}")
            
        function_name = response.data[0]['name']
        
        # Validate required parameters
        json_schema = get_tool_schema(function_name)
        if json_schema:
            required_params = json_schema.get('required', [])
            missing_params = [param for param in required_params if param not in args]
            if missing_params:
                raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")
        
        # Convert all arguments to their expected types
        for key, value in args.items():
            expected_type = get_parameter_type(function_name, key)
            if expected_type:
                args[key] = convert_value(value, expected_type)
        
        # For image generation, make image_prompt optional if not provided
        if function_name == 'generate_image' and 'image_prompt' not in args:
            args['image_prompt'] = ''
        
        logger.info(f"Running {function_name} with args: {args}")
        output = replicate.run(
            model,
            input=args
        )
        return output
    except Exception as e:
        logger.error(f"Error in generate function: {str(e)}")
        raise