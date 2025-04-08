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
        logger.error(f"Error fetching tool schema: {str(e)}")
        return None

def get_parameter_type(function_name: str, param_name: str) -> Optional[str]:
    """Get the expected type for a parameter from the tools configuration."""
    json_schema = get_tool_schema(function_name)
    if not json_schema:
        return None
        
    properties = json_schema.get('properties', {})
    if param_name in properties:
        return properties[param_name].get('type')
    return None

def get_parameter_format(json_schema: Dict[str, Any], param_name: str) -> Optional[str]:
    """Get the format of a parameter from its JSON schema."""
    properties = json_schema.get('properties', {})
    if param_name in properties:
        return properties[param_name].get('format')
    return None

def get_parameter_default(json_schema: Dict[str, Any], param_name: str, function_name: str) -> Any:
    """Get the default value for a parameter from its JSON schema."""
    properties = json_schema.get('properties', {})
    if param_name in properties:
        param_schema = properties[param_name]
        
        # Check for explicit default value
        if 'default' in param_schema:
            return param_schema['default']
            
        # Set sensible defaults based on type and constraints
        param_type = param_schema.get('type')
        if param_type == 'integer':
            if 'minimum' in param_schema:
                return param_schema['minimum']
            return 0
        elif param_type == 'number':
            if 'minimum' in param_schema:
                return param_schema['minimum']
            return 0.0
        elif param_type == 'boolean':
            return False
        elif param_type == 'string':
            if 'enum' in param_schema and param_schema['enum']:
                return param_schema['enum'][0]
            return ''
    return None

def convert_value(value: Any, expected_type: str, format: Optional[str] = None) -> Any:
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
            if format == 'uri' and value == '':
                return None  # Don't convert empty strings to URIs
            return str(value)
        return value
    except (ValueError, TypeError) as e:
        logger.error(f"Error converting value {value} to type {expected_type}: {str(e)}")
        return value

def generate(model: str, args: Dict[str, Any]) -> Any:
    """Generate output using the Replicate API with improved error handling."""
    try:
        # Get the tool details from the database
        supabase = get_supabase_client()
        response = supabase.table('tools').select('name, json_schema').eq('model', model).execute()
        
        if not response.data:
            raise ValueError(f"No tool found for model {model}")
            
        tool_data = response.data[0]
        function_name = tool_data['name']
        json_schema = tool_data['json_schema']
        
        # Log the schema for debugging
        logger.info(f"Tool schema for {function_name}: {json_schema}")
        
        # Direct handling for known tools
        if function_name == "generate_image":
            # For image generation, we only need prompt and optional parameters
            if 'prompt' not in args:
                raise ValueError("Prompt is required for image generation")
            # Set defaults for optional parameters
            args.setdefault('width', 512)
            args.setdefault('height', 512)
            args.setdefault('seed', 42)
            args.setdefault('aspect_ratio', '1:1')
            args.setdefault('output_format', 'png')
            args.setdefault('output_quality', 100)
            args.setdefault('safety_tolerance', 3)
            args.setdefault('prompt_upsampling', True)
            # Remove image_prompt if not provided
            if 'image_prompt' not in args:
                args.pop('image_prompt', None)
                
        elif function_name == "generate_music_v2":
            # For music generation, we need prompt and optional parameters
            if 'prompt' not in args:
                raise ValueError("Prompt is required for music generation")
            # Set defaults for optional parameters
            args.setdefault('duration', 8)
            args.setdefault('temperature', 1.0)
            args.setdefault('top_k', 250)
            args.setdefault('top_p', 0)
            args.setdefault('seed', 42)
            args.setdefault('model_version', 'stereo-melody-large')
            args.setdefault('output_format', 'wav')
            args.setdefault('continuation', False)
            args.setdefault('multi_band_diffusion', False)
            args.setdefault('normalization_strategy', 'loudness')
            args.setdefault('classifier_free_guidance', 3)
            # Remove input_audio if not provided
            if 'input_audio' not in args:
                args.pop('input_audio', None)
                args.pop('continuation_start', None)
                args.pop('continuation_end', None)
                
        else:
            # For unknown tools, use schema validation
            required_params = json_schema.get('required', [])
            properties = json_schema.get('properties', {})
            
            # Handle conditional requirements
            for param in list(required_params):
                param_schema = properties.get(param, {})
                if param_schema.get('format') == 'uri' and param not in args:
                    required_params.remove(param)
                elif param_schema.get('type') == 'boolean' and param not in args:
                    args[param] = False
                    required_params.remove(param)
            
            # Set defaults for required parameters
            for param in required_params:
                if param not in args:
                    default_value = get_parameter_default(json_schema, param, function_name)
                    if default_value is not None:
                        args[param] = default_value
                    else:
                        raise ValueError(f"Missing required parameter: {param}")
        
        # Convert all arguments to their expected types and formats
        for key, value in args.items():
            expected_type = get_parameter_type(function_name, key)
            expected_format = get_parameter_format(json_schema, key)
            if expected_type:
                args[key] = convert_value(value, expected_type, expected_format)
        
        # Remove None values from args
        args = {k: v for k, v in args.items() if v is not None}
        
        logger.info(f"Running {function_name} with args: {args}")
        output = replicate.run(
            model,
            input=args
        )
        return output
    except Exception as e:
        logger.error(f"Error in generate function: {str(e)}")
        raise