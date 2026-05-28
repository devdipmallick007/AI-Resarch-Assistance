from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from app.utils.helpers import Document


class DocumentLoader:
    supported_extensions = {".txt", ".md", ".csv", ".json", ".pdf"}

    def load(self, path: str | Path) -> list[Document]:
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")
        suffix = file_path.suffix.lower()
        if suffix not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {suffix}")

        if suffix in {".txt", ".md"}:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
        elif suffix == ".csv":
            text = pd.read_csv(file_path).to_csv(index=False)
        elif suffix == ".json":
            text = json.dumps(json.loads(file_path.read_text(encoding="utf-8")), indent=2)
        else:
            text = self._load_pdf(file_path)

        return [Document(text=text, metadata={"source": file_path.name, "path": str(file_path)})]

    def _load_pdf(self, path: Path) -> str:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ImportError("Install pypdf to load PDF documents.") from exc

        reader = PdfReader(str(path))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(pages)
