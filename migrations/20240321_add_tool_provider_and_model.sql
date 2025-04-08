-- Add api_name and model columns to tools table
ALTER TABLE public.tools
ADD COLUMN api_name text,
ADD COLUMN model text;

-- Update existing tools with Replicate api_name and their respective models
UPDATE public.tools
SET 
    api_name = 'Replicate',
    model = CASE 
        WHEN name = 'generate_image' THEN 'black-forest-labs/flux-1.1-pro'
        WHEN name = 'generate_music_v2' THEN 'meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb'
        WHEN name = 'generate_music' THEN 'minimax/music-01'
        ELSE NULL
    END
WHERE name IN ('generate_image', 'generate_music_v2', 'generate_music');

-- Add NOT NULL constraint after updating existing data
ALTER TABLE public.tools
ALTER COLUMN api_name SET NOT NULL,
ALTER COLUMN model SET NOT NULL; 