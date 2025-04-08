# DiscoJelly Backend

A dynamic AI agent system that loads tools and configurations from Supabase.

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your credentials:
   ```bash
   cp .env.example .env
   ```
4. Update the `.env` file with your:
   - OpenAI API Key
   - Supabase URL
   - Supabase Key

## Database Setup

The system uses two main tables in Supabase:

### Tools Table
```sql
create table public.tools (
  id uuid not null default extensions.uuid_generate_v4 (),
  name text not null,
  description text not null,
  json_schema jsonb not null,
  created_at timestamp with time zone not null default now(),
  updated_at timestamp with time zone not null default now(),
  constraint tools_pkey primary key (id)
) TABLESPACE pg_default;
```

### AI Agents Table
```sql
create table public.ai_agents (
  id uuid not null default extensions.uuid_generate_v4 (),
  name text not null,
  description text null,
  category text null,
  model text null,
  system_prompt text null,
  template text null,
  resources jsonb null default '[]'::jsonb,
  chat_history jsonb null default '[]'::jsonb,
  tool_ids text[] null default '{}'::text[],
  tools jsonb null default '[]'::jsonb,
  creator_id uuid null,
  created_at timestamp with time zone null default now(),
  updated_at timestamp with time zone not null default now(),
  capabilities jsonb null default '{}'::jsonb,
  constraint ai_agents_pkey primary key (id),
  constraint ai_agents_creator_id_fkey foreign KEY (creator_id) references profiles (id)
) TABLESPACE pg_default;
```

## Running the Server

```bash
uvicorn main:app --reload
```

## API Endpoints

### POST /process-input
Process user input with a specific agent.

Request body:
```json
{
  "user_input": "Your message here",
  "agent_name": "name_of_agent"
}
```

Response:
```json
{
  "response": "Agent's response"
}
```

## Error Handling

The API will return appropriate HTTP status codes:
- 404: Agent not found
- 500: Server error