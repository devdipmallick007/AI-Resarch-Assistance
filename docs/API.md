# API Reference

Base URL:

```text
http://127.0.0.1:8000
```

Interactive docs:

```text
http://127.0.0.1:8000/docs
```

## GET /health

Returns application status and indexed section count.

Example:

```powershell
curl http://127.0.0.1:8000/health
```

Response:

```json
{
  "status": "ok",
  "indexed_chunks": 99
}
```

## POST /ingest

Uploads and indexes a document.

Supported file types:

- `.txt`
- `.md`
- `.csv`
- `.json`
- `.pdf`

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/ingest" -F "file=@data/uploaded_docs/langchain.pdf"
```

Response:

```json
{
  "file": "langchain.pdf",
  "chunks": 54
}
```

## POST /query

Runs a research query against the current index.

Request body:

```json
{
  "query": "What is LangChain?",
  "k": 5
}
```

Constraints:

- `query`: required, minimum length 1
- `k`: 1 to 10

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/query" ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"What is LangChain?\",\"k\":5}"
```

Response:

```json
{
  "query": "What is LangChain?",
  "refined_query": "What is LangChain?",
  "answer": "**Answer**\nLangChain is ... [langchain.pdf#0]",
  "citations": ["langchain.pdf#0"],
  "warnings": [],
  "grounded": true
}
```

## POST /reset

Clears the local vector index.

Example:

```powershell
curl -X POST "http://127.0.0.1:8000/reset"
```

Response:

```json
{
  "status": "cleared"
}
```

## Error Handling

Invalid upload:

```json
{
  "detail": "Unsupported document type: .exe"
}
```

Invalid query:

```json
{
  "detail": [
    {
      "type": "greater_than_equal",
      "loc": ["body", "k"],
      "msg": "Input should be greater than or equal to 1"
    }
  ]
}
```
