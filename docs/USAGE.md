# Setup And Usage

## Requirements

- Python 3.10+
- No Docker required
- Optional Gemini or OpenAI API key

## Install

```powershell
cd D:\ai-support-automation
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Environment Configuration

Create or edit `.env`.

Gemini:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key
GEMINI_MODEL=gemini-2.5-flash
```

OpenAI:

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=your-openai-api-key
OPENAI_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
```

Shared settings:

```env
VECTOR_STORE_PATH=data/faiss_index
UPLOAD_DIR=data/uploaded_docs
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

## Run Streamlit

```powershell
streamlit run frontend\streamlit_app.py
```

Open:

```text
http://localhost:8501
```

### Streamlit Workflow

1. Upload one or more supported files.
2. The app automatically indexes new uploads.
3. Ask a research question.
4. Adjust **Search depth** only when needed.
5. Use **Rebuild search index** if files already exist in `data/uploaded_docs` but the index was cleared.

## Run FastAPI

```powershell
uvicorn app.main:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Run CLI

Ask a question against the existing index:

```powershell
python -m app.main "What is integrated marketing communications?"
```

Ingest files and ask:

```powershell
python -m app.main --ingest data\uploaded_docs\Fundamentals_of_Marketing.pdf "What is IMC?"
```

## Rebuild The Index

From Streamlit:

1. Open the sidebar.
2. Click **Rebuild search index**.

From Python:

```powershell
python -c "from pathlib import Path; from app.workflow.langgraph_flow import ResearchWorkflow; w=ResearchWorkflow(); w.vector_store.clear(); [w.ingest_document(p) for p in Path('data/uploaded_docs').glob('*') if p.suffix.lower() in {'.txt','.md','.csv','.json','.pdf'}]; print(w.vector_store.count())"
```

## Logs

Logs are written to:

```text
logs/app.log
```

Control logging with:

```env
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
```

The app writes system-level logs for startup, environment summary, workflow initialization, indexing, retrieval, and answer validation. API keys are never printed; logs only show whether a provider key is configured.

## Testing

```powershell
pytest tests
```

If tests cannot start because `pytest` is missing:

```powershell
pip install -r requirements.txt
```

## Troubleshooting

### FastAPI shows 404 on `/`

This is expected. Use:

```text
http://127.0.0.1:8000/docs
```

or:

```text
http://127.0.0.1:8000/health
```

### Answers say no relevant information

Possible causes:

- The index is empty.
- The uploaded document was not indexed.
- The query terms do not appear in the indexed documents.

Fix:

- Click **Rebuild search index** in Streamlit.
- Confirm `data/faiss_index/store.json` contains records.

### Output only shows fallback answers

Possible causes:

- Missing API key.
- LLM quota exceeded.
- Gemini/OpenAI SDK not installed.

Fix:

```powershell
pip install -r requirements.txt
```

Then restart Streamlit or FastAPI.
