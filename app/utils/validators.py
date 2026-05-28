from __future__ import annotations

from pathlib import Path

from app.rag.document_loader import DocumentLoader


def validate_upload_path(path: str | Path) -> Path:
    file_path = Path(path)
    if file_path.suffix.lower() not in DocumentLoader.supported_extensions:
        raise ValueError(f"Unsupported document type: {file_path.suffix}")
    return file_path


def safe_upload_filename(filename: str | None) -> str:
    if not filename:
        raise ValueError("Uploaded file must have a filename.")
    safe_name = Path(filename).name
    if safe_name in {"", ".", ".."}:
        raise ValueError("Uploaded file has an invalid filename.")
    validate_upload_path(safe_name)
    return safe_name
