import replicate
from tools_config import tools

def get_parameter_type(function_name, param_name):
    """Get the expected type for a parameter from the tools configuration."""
    for tool in tools:
        if tool["function"]["name"] == function_name:
            properties = tool["function"]["parameters"]["properties"]
            if param_name in properties:
                return properties[param_name]["type"]
    return None

def convert_value(value, expected_type):
    """Convert a value to the expected type."""
    if value is None:
        return value
    
    if expected_type == "integer":
        return int(value)
    elif expected_type == "number":
        return float(value)
    elif expected_type == "boolean":
        return bool(value)
    elif expected_type == "string":
        return str(value)
    return value

def generate(model, args):
    # Get the function name from the model
    function_name = None
    if "black-forest-labs/flux-1.1-pro" in model:
        function_name = "generate_image"
    elif "meta/musicgen" in model:
        function_name = "generate_music_v2"
    elif "minimax/music-01" in model:
        function_name = "generate_music"
    
    if function_name:
        # Convert all arguments to their expected types
        for key, value in args.items():
            expected_type = get_parameter_type(function_name, key)
            if expected_type:
                args[key] = convert_value(value, expected_type)
    
    print(args)
    output = replicate.run(
        model,
        input=args
    )
    return output