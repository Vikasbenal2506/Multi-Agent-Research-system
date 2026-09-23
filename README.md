# Multi Agent Research System

A research workflow that combines search, scraping, and LLM-powered writing/critique to generate a structured report on a topic.

## Features

- Search agent for finding recent sources
- Reader agent for scraping selected URLs
- Writer chain to generate a polished report
- Critic chain to evaluate the report
- Streamlit frontend for local interaction

## Project Structure

- `agents.py` – agent and chain definitions
- `tools.py` – web search and scraping tools
- `pipeline.py` – terminal research pipeline
- `app.py` – Streamlit UI
- `.env.example` – template for local environment variables
- `.env` – local environment variables (not committed)
- `requirements.lock` – optional pinned dependencies when available

## Setup

1. Create and activate a virtual environment:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

3. Copy the sample environment file and fill in your keys:

   ```powershell
   Copy-Item .env.example .env
   ```

   Then update `.env`:

   ```env
   TAVILY_API_KEY=your_tavily_key
   GOOGLE_API_KEY=your_google_key
   GROQ_API_KEY=your_groq_key
   ```

4. Run the app:

   ```powershell
   streamlit run app.py
   ```

5. Or run the CLI pipeline:

   ```powershell
   python pipeline.py
   ```

## Notes

- Keep `.env` local and do not commit it.
- If the model is unavailable or access is denied, update the model name in `agents.py` to a model available on your provider account.
- The project uses LangChain and provider-specific integrations for search, scraping, and generation.

## License

This project is for educational and research use.
