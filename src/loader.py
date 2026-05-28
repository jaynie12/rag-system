from __future__ import annotations

from pathlib import Path

from src.pdfToMd import normalize_text_to_markdown, pdf_to_markdown


def load_document_text(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Document not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return pdf_to_markdown(file_path)
    if suffix in {".txt", ".md"}:
        return normalize_text_to_markdown(file_path.read_text(encoding="utf-8"))
    raise ValueError(f"Unsupported file type: {suffix}")
