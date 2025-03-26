tools = [{
    "type": "function",
    "function": {
        "name": "generate_image",
        "description": "Generate an image based on text prompt and optional parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Text prompt for image generation"
                },
                "seed": {
                    "type": "integer",
                    "description": "Random seed. Set for reproducible generation"
                },
                "width": {
                    "type": "integer",
                    "description": "Width of the generated image in text-to-image mode. Must be between 256 and 1440, multiple of 32."
                },
                "height": {
                    "type": "integer",
                    "description": "Height of the generated image in text-to-image mode. Must be between 256 and 1440, multiple of 32."
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "Aspect ratio for the generated image",
                    "enum": ["custom", "1:1", "16:9", "3:2", "2:3", "4:5", "5:4", "9:16", "3:4", "4:3"]
                },
                "image_prompt": {
                    "type": "string",
                    "description": "Image to use with Flux Redux. Must be jpeg, png, gif, or webp."
                },
                "output_format": {
                    "type": "string",
                    "description": "Format of the output images.",
                    "enum": ["webp", "jpg", "png"]
                },
                "output_quality": {
                    "type": "integer",
                    "description": "Quality when saving the output images, from 0 to 100."
                },
                "safety_tolerance": {
                    "type": "integer",
                    "description": "Safety tolerance, 1 is most strict and 6 is most permissive"
                },
                "prompt_upsampling": {
                    "type": "boolean",
                    "description": "Automatically modify the prompt for more creative generation"
                }
            },
            "required": ["prompt", "seed", "width", "height", "aspect_ratio", "image_prompt", "output_format", "output_quality", "safety_tolerance", "prompt_upsampling"],
            "additionalProperties": False
        },
        "strict": True
    }
}, {
    "type": "function",
    "function": {
        "name": "generate_music",
        "description": "Generate music using provided lyrics, voice, and instrumental references.",
        "parameters": {
            "type": "object",
            "properties": {
                "lyrics": {
                    "type": "string",
                    "description": "Lyrics with optional formatting. You can use a newline to separate each line of lyrics. You can use two newlines to add a pause between lines. You can use double hash marks (##) at the beginning and end of the lyrics to add accompaniment. Maximum 350 to 400 characters. (default: '')"
                },
                "bitrate": {
                    "type": "integer",
                    "enum": [32000, 64000, 128000, 256000],
                    "description": "Bitrate for the generated music. (default: 256000)"
                },
                "voice_id": {
                    "type": "string",
                    "description": "Reuse a previously uploaded voice ID."
                },
                "song_file": {
                    "type": "string",
                    "description": "Reference song, should contain music and vocals. Must be a .wav or .mp3 file longer than 15 seconds. (format: uri)"
                },
                "voice_file": {
                    "type": "string",
                    "description": "Voice reference. Must be a .wav or .mp3 file longer than 15 seconds. If only a voice reference is given, an a cappella vocal hum will be generated. (format: uri)"
                },
                "sample_rate": {
                    "type": "integer",
                    "enum": [16000, 24000, 32000, 44100],
                    "description": "Sample rate for the generated music. (default: 44100)"
                },
                "instrumental_id": {
                    "type": "string",
                    "description": "Reuse a previously uploaded instrumental ID."
                },
                "instrumental_file": {
                    "type": "string",
                    "description": "Instrumental reference. Must be a .wav or .mp3 file longer than 15 seconds. If only an instrumental reference is given, a track without vocals will be generated. (format: uri)"
                }
            },
            "required": [
                "lyrics", "bitrate", "voice_id", "song_file", "voice_file", "sample_rate", "instrumental_id", "instrumental_file"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}, {
    "type": "function",
    "function": {
        "name": "generate_music_v2",
        "description": "Generate music based on a given prompt, input audio, and various parameters.",
        "parameters": {
            "type": "object",
            "properties": {
                "seed": {
                    "type": "integer",
                    "description": "Seed for random number generator. If None or -1, a random seed will be used. (x-order: 14)"
                },
                "top_k": {
                    "type": "integer",
                    "description": "Reduces sampling to the k most likely tokens. Default: 250 (x-order: 9)"
                },
                "top_p": {
                    "type": "number",
                    "description": "Reduces sampling to tokens with cumulative probability of p. When set to 0 (default), top_k sampling is used. (x-order: 10)"
                },
                "prompt": {
                    "type": "string",
                    "description": "A description of the music you want to generate. (x-order: 1)"
                },
                "duration": {
                    "type": "integer",
                    "description": "Duration of the generated audio in seconds. Default: 8 (x-order: 3)"
                },
                "input_audio": {
                    "type": "string",
                    "description": "An audio file (URI) that will influence the generated music. If 'continuation' is True, the generated music will be a continuation of the audio file. Otherwise, the generated music will mimic the audio file's melody. (x-order: 2)"
                },
                "temperature": {
                    "type": "number",
                    "description": "Controls the 'conservativeness' of the sampling process. Higher temperature means more diversity. Default: 1 (x-order: 11)"
                },
                "continuation": {
                    "type": "boolean",
                    "description": "If True, generated music will continue from 'input_audio'. Otherwise, generated music will mimic 'input_audio's melody. Default: False (x-order: 4)"
                },
                "model_version": {
                    "type": "string",
                    "enum": ["stereo-melody-large", "stereo-large", "melody-large", "large"],
                    "description": "Model to use for generation. Default: stereo-melody-large (x-order: 0)"
                },
                "output_format": {
                    "type": "string",
                    "enum": ["wav", "mp3"],
                    "description": "Output format for generated audio. Default: wav (x-order: 13)"
                },
                "continuation_end": {
                    "type": "integer",
                    "description": "End time of the audio file to use for continuation. If -1 or None, will default to the end of the audio clip. Minimum: 0 (x-order: 6)"
                },
                "continuation_start": {
                    "type": "integer",
                    "description": "Start time of the audio file to use for continuation. Default: 0, Minimum: 0 (x-order: 5)"
                },
                "multi_band_diffusion": {
                    "type": "boolean",
                    "description": "If True, the EnCodec tokens will be decoded with MultiBand Diffusion. Only works with non-stereo models. Default: False (x-order: 7)"
                },
                "normalization_strategy": {
                    "type": "string",
                    "enum": ["loudness", "clip", "peak", "rms"],
                    "description": "Strategy for normalizing audio. Default: loudness (x-order: 8)"
                },
                "classifier_free_guidance": {
                    "type": "integer",
                    "description": "Increases the influence of inputs on the output. Higher values produce lower-variance outputs that adhere more closely to inputs. Default: 3 (x-order: 12)"
                }
            },
            "required": [
                "seed", "top_k", "top_p", "prompt", "duration", "input_audio", "temperature",
                "continuation", "model_version", "output_format", "continuation_end",
                "continuation_start", "multi_band_diffusion", "normalization_strategy", "classifier_free_guidance"
            ],
            "additionalProperties": False
        },
        "strict": True
    }
}] 