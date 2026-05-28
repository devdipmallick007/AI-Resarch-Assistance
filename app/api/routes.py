from __future__ import annotations

import os
import shutil
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.workflow.langgraph_flow import ResearchWorkflow
from app.utils.logging_config import configure_logging, log_system_context
from app.utils.validators import safe_upload_filename

load_dotenv()
configure_logging()
logger = logging.getLogger(__name__)

app = FastAPI(title="Multi-Agent Research Assistant", version="0.1.0")
workflow = ResearchWorkflow()
log_system_context("FastAPI")


@app.on_event("startup")
def on_startup() -> None:
    logger.info("FastAPI application startup complete | indexed_chunks=%s", workflow.vector_store.count())


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=10)


@app.get("/health")
def health() -> dict:
    logger.info("Health check requested")
    return {"status": "ok", "indexed_chunks": workflow.vector_store.count()}


@app.post("/ingest")
def ingest(file: UploadFile = File(...)) -> dict:
    upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploaded_docs"))
    upload_dir.mkdir(parents=True, exist_ok=True)
    try:
        filename = safe_upload_filename(file.filename)
        logger.info("Upload received | filename=%s | content_type=%s", filename, file.content_type)
        target = upload_dir / filename
        with target.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        chunks = workflow.ingest_document(target)
        logger.info("Ingested uploaded file %s with %s chunks", filename, chunks)
    except Exception as exc:
        logger.exception("Failed to ingest uploaded file %s", file.filename)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"file": filename, "chunks": chunks}


@app.post("/query")
def query(request: QueryRequest) -> dict:
    try:
        logger.info("API query received | chars=%s | k=%s", len(request.query), request.k)
        result = dict(workflow.run(request.query, k=request.k))
        logger.info("Query completed with grounded=%s", result.get("grounded"))
        return result
    except ValueError as exc:
        logger.warning("Rejected invalid query: %s", exc)
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/reset")
def reset() -> dict:
    workflow.vector_store.clear()
    logger.info("Vector store cleared")
    return {"status": "cleared"}
