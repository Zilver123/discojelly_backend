# DiscoJelly Backend

This directory contains the Python backend for the DiscoJelly application, built with FastAPI.

## Setup

1.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Create a `.env` file:**
    Create a file named `.env` in this directory (`backend/`) and add your Supabase credentials:
    ```
    SUPABASE_URL="YOUR_SUPABASE_URL"
    SUPABASE_KEY="YOUR_SUPABASE_ANON_KEY"
    ```

3.  **Define the database table in Supabase:**
    Log in to your Supabase project and run the following SQL query to create the `services` table:
    ```sql
    CREATE TABLE services (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        name TEXT NOT NULL,
        description TEXT,
        image_url TEXT,
        how_to_use TEXT,
        service_url TEXT,
        tags TEXT[],
        created_at TIMESTAMPTZ DEFAULT NOW()
    );
    ```

## Running the server

Once the setup is complete, you can run the development server:

```bash
uvicorn main:app --reload
```

The server will be available at `http://127.0.0.1:8000`. 