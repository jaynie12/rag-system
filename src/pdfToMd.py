from __future__ import annotations

import re
from pathlib import Path

from pypdf import PdfReader


def extract_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    reader = PdfReader(str(pdf_path))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def normalize_text_to_markdown(text: str) -> str:
    """Normalize spacing and paragraph boundaries for downstream chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def pdf_to_markdown(pdf_path: Path) -> str:
    raw_text = extract_pdf_text(pdf_path)
    return normalize_text_to_markdown(raw_text)
