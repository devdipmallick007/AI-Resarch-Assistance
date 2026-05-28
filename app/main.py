from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from app.utils.logging_config import configure_logging
from app.workflow.langgraph_flow import ResearchWorkflow

configure_logging()

try:
    from app.api.routes import app
except ModuleNotFoundError as exc:
    if exc.name not in {"fastapi", "pydantic", "python_multipart"}:
        raise
    app = None


def run_cli() -> None:
    load_dotenv()
    configure_logging()
    parser = argparse.ArgumentParser(description="Run a research query against the local index.")
    parser.add_argument("query", nargs="?", help="Question to answer")
    parser.add_argument("--ingest", nargs="*", default=[], help="Document paths to add before querying")
    args = parser.parse_args()

    workflow = ResearchWorkflow()
    for file_path in args.ingest:
        count = workflow.ingest_document(Path(file_path))
        print(f"Ingested {count} chunks from {file_path}")

    if args.query:
        result = workflow.run(args.query)
        print(result["answer"])
        if result.get("citations"):
            print("\nSources:")
            for citation in result["citations"]:
                print(f"- {citation}")
    elif not args.ingest:
        parser.print_help()


if __name__ == "__main__":
    run_cli()
