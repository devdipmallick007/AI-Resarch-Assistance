# Multi-Agent Research Assistant

A no-Docker Python research assistant for uploading documents, building a local searchable knowledge base, and answering questions with citations through a multi-agent RAG workflow.

The system supports:

- Streamlit web UI for document upload and research questions
- FastAPI backend for programmatic ingestion and querying
- Local JSON vector index under `data/faiss_index`
- Gemini or OpenAI for final answer generation
- Local fallback embeddings and extractive answers when no LLM key is available
- Evidence-grounded citations such as `[Fundamentals_of_Marketing.pdf#213]`

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Setup And Usage](docs/USAGE.md)
- [API Reference](docs/API.md)

## Quick Start

```powershell
cd D:\ai-support-automation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Configure `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash

VECTOR_STORE_PATH=data/faiss_index
UPLOAD_DIR=data/uploaded_docs
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

Run Streamlit:

```powershell
streamlit run frontend\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

Run FastAPI:

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## Project Layout

```text
app/
  agents/       Query, retrieval, research, and validation agents
  api/          FastAPI routes
  llm/          Gemini/OpenAI client wrapper and prompts
  rag/          Document loading, chunking, embeddings, vector store, retriever
  utils/        Logging, validators, shared dataclasses
  workflow/     End-to-end research workflow
frontend/       Streamlit interface
data/           Uploaded documents and local index
docs/           Project documentation
tests/          RAG tests
```

## Testing

```powershell
pytest tests
```

If `pytest` is not installed:

```powershell
pip install -r requirements.txt
```
