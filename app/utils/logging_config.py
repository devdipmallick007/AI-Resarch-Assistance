from __future__ import annotations

import logging
import os
import platform
import sys
from pathlib import Path

from dotenv import load_dotenv


def configure_logging() -> None:
    load_dotenv()
    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    log_file = os.getenv("LOG_FILE", "logs/app.log")

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(path, encoding="utf-8"))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("watchfiles").setLevel(logging.WARNING)


def log_system_context(app_name: str) -> None:
    load_dotenv()
    logger = logging.getLogger("system")
    provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
    gemini_configured = bool(os.getenv("GEMINI_API_KEY", "").strip())
    openai_configured = bool(os.getenv("OPENAI_API_KEY", "").strip())
    vector_store_path = os.getenv("VECTOR_STORE_PATH", "data/faiss_index")
    upload_dir = os.getenv("UPLOAD_DIR", "data/uploaded_docs")

    logger.info(
        "%s startup | python=%s | platform=%s | cwd=%s",
        app_name,
        sys.version.split()[0],
        platform.platform(),
        Path.cwd(),
    )
    logger.info(
        "%s config | provider=%s | gemini_key_configured=%s | openai_key_configured=%s | vector_store=%s | upload_dir=%s",
        app_name,
        provider,
        gemini_configured,
        openai_configured,
        vector_store_path,
        upload_dir,
    )
