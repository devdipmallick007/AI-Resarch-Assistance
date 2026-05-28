from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.workflow.langgraph_flow import ResearchWorkflow
from app.rag.document_loader import DocumentLoader
from app.utils.logging_config import configure_logging, log_system_context

configure_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Research Assistant", layout="wide")
st.title("Multi-Agent Research Assistant")

if not st.session_state.get("system_logged"):
    log_system_context("Streamlit")
    st.session_state.system_logged = True

workflow = st.session_state.get("workflow")
if workflow is None:
    workflow = ResearchWorkflow()
    st.session_state.workflow = workflow
    logger.info("Streamlit workflow created | indexed_sections=%s", workflow.vector_store.count())

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=["txt", "md", "csv", "json", "pdf"],
    )
    indexed_uploads = st.session_state.setdefault("indexed_uploads", set())
    pending_files = [
        uploaded_file
        for uploaded_file in uploaded_files or []
        if (uploaded_file.name, uploaded_file.size) not in indexed_uploads
    ]
    if pending_files:
        upload_dir = ROOT / "data" / "uploaded_docs"
        upload_dir.mkdir(parents=True, exist_ok=True)
        total = 0
        with st.spinner("Indexing uploaded documents..."):
            for uploaded_file in pending_files:
                target = upload_dir / uploaded_file.name
                target.write_bytes(uploaded_file.getbuffer())
                logger.info("Streamlit upload saved | filename=%s | bytes=%s", uploaded_file.name, uploaded_file.size)
                total += workflow.ingest_document(target)
                indexed_uploads.add((uploaded_file.name, uploaded_file.size))
        logger.info("Streamlit upload indexing complete | files=%s | chunks=%s", len(pending_files), total)
        st.success(f"Ready: indexed {total} document sections")

    if st.button("Clear index"):
        workflow.vector_store.clear()
        st.session_state.indexed_uploads = set()
        logger.info("Streamlit clear index requested")
        st.success("Index cleared")

    st.metric("Indexed sections", workflow.vector_store.count())
    existing_docs = [
        path
        for path in (ROOT / "data" / "uploaded_docs").glob("*")
        if path.suffix.lower() in DocumentLoader.supported_extensions
    ]
    if existing_docs and st.button("Rebuild search index"):
        workflow.vector_store.clear()
        total = 0
        with st.spinner("Rebuilding search index..."):
            for path in existing_docs:
                total += workflow.ingest_document(path)
        st.session_state.indexed_uploads = {(path.name, path.stat().st_size) for path in existing_docs}
        logger.info("Streamlit rebuild index complete | files=%s | chunks=%s", len(existing_docs), total)
        st.success(f"Rebuilt index with {total} document sections")

    with st.expander("Search settings", expanded=False):
        k = st.slider(
            "Search depth",
            1,
            10,
            5,
            help="Number of document sections reviewed before answering. Higher values add context but can introduce noise.",
        )

query = st.text_area("Research question", height=110, placeholder="Ask a question about your uploaded documents")

if st.button("Research", type="primary", disabled=not query.strip()):
    with st.spinner("Coordinating agents..."):
        logger.info("Streamlit research requested | chars=%s | k=%s", len(query), k)
        result = workflow.run(query, k=k)
        logger.info(
            "Streamlit research completed | grounded=%s | citations=%s",
            result.get("grounded"),
            len(result.get("citations", [])),
        )
    st.markdown(result["answer"]) # type: ignore
    with st.expander("Sources used", expanded=True):
        for citation in result.get("citations", []):
            st.write(f"- {citation}")
    for warning in result.get("warnings", []):
        st.warning(warning)
