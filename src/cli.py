from __future__ import annotations

import argparse
from pathlib import Path
import sys

from src.config import load_settings
from src.rag_pipeline import RAGPipeline


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="CLI RAG for a single Jane Austen book PDF")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="Build or rebuild the index from a PDF")
    index_parser.add_argument("--pdf", required=True, help="Path to PDF file")
    index_parser.add_argument("--title", required=False, help="Optional document title")

    ask_parser = subparsers.add_parser("ask", help="Ask a question against the indexed content")
    ask_parser.add_argument("question", help='Question string, e.g. ask "Who is Darcy?"')
    ask_parser.add_argument("--verbose", action="store_true", help="Print retrieved chunk scores")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        settings = load_settings()
        pipeline = RAGPipeline(settings)
    except Exception as exc:
        print(f"Configuration error: {exc}")
        return 1

    if args.command == "index":
        pdf_path = Path(args.pdf)
        try:
            count = pipeline.build_index(pdf_path=pdf_path, title=args.title)
        except Exception as exc:
            print(f"Indexing failed: {exc}")
            return 1
        print(f"Indexed {count} chunks from {pdf_path}.")
        return 0

    if args.command == "ask":
        question = args.question.strip()
        if not question:
            print("Question cannot be empty.")
            return 1

        try:
            answer, retrieved = pipeline.answer_question(question)
        except Exception as exc:
            print(f"Question answering failed: {exc}")
            return 1

        print(answer)
        if args.verbose:
            print("\nTop retrieved chunks:")
            for chunk in retrieved:
                preview = chunk.text.replace("\n", " ")[:120]
                print(f"- score={chunk.score:.3f} chunk={chunk.chunk_index} text={preview}...")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
