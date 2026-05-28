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


def _is_chapter_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return bool(
        re.match(r"^(chapter|book|volume)\s+([0-9]+|[ivxlcdm]+)\b", stripped, flags=re.IGNORECASE)
    )


def _is_page_or_footer_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    # Keep actual headings even if they are uppercase/short.
    if _is_chapter_heading(stripped):
        return False

    # Common page markers.
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    if re.fullmatch(r"page\s+\d{1,4}", stripped, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"-\s*\d{1,4}\s*-", stripped):
        return True

    # Frequent ebook/pdf footer/header artifacts.
    if re.search(r"project gutenberg|www\.", stripped, flags=re.IGNORECASE):
        return True
    if re.search(r"copyright|all rights reserved", stripped, flags=re.IGNORECASE):
        return True

    return False


def normalize_text_to_markdown(text: str) -> str:
    """Normalize text and remove page/footer artifacts for downstream chunking."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
    cleaned_lines: list[str] = []

    for line in lines:
        if _is_page_or_footer_noise(line):
            continue
        cleaned_lines.append(line)

    # Rebuild paragraphs while preserving intentional blank lines and headings.
    output: list[str] = []
    previous_blank = False
    for line in cleaned_lines:
        if not line:
            if not previous_blank:
                output.append("")
            previous_blank = True
            continue

        output.append(line)
        previous_blank = False

    normalized = "\n".join(output)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def pdf_to_markdown(pdf_path: Path) -> str:
    raw_text = extract_pdf_text(pdf_path)
    return normalize_text_to_markdown(raw_text)
