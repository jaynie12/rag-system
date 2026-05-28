# rag-system

CLI Retrieval-Augmented Generation (RAG) system for a single Jane Austen book PDF.

## Features

- PDF ingestion to normalized text
- Sentence-aware chunking with max chunk size (`CHUNK_MAX_TOKENS`, default 800)
- OpenAI embeddings for chunk and query vectors
- SQLite persistence for documents, chunks, and vectors
- Cosine similarity retrieval (`top-k`)
- Grounded answer generation:
  - answers only from retrieved context
  - returns exactly `I don't know` when evidence is missing or weak

## Project Structure

```text
rag-system/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   └── .gitkeep
├── storage/
│   └── .gitkeep
├── src/
│   ├── __init__.py
│   ├── cli.py
│   ├── chunker.py
│   ├── config.py
│   ├── embeddings.py
│   ├── generator.py
│   ├── loader.py
│   ├── pdfToMd.py
│   ├── rag_pipeline.py
│   ├── retriever.py
│   └── vector_store.py
└── tests/
    ├── test_chunker.py
    └── test_retriever.py
```

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and set your OpenAI key:
   - `OPENAI_API_KEY`
   - optional model/threshold settings
4. Put your Jane Austen PDF into `data/`.

## CLI Usage

Build index:

```bash
python -m src.cli index --pdf data/pride_and_prejudice.pdf --title "Pride and Prejudice"
```

Ask question:

```bash
python -m src.cli ask "Who insults Elizabeth at the dance?"
```

Ask with retrieval debug output:

```bash
python -m src.cli ask "Who insults Elizabeth at the dance?" --verbose
```

## Test

```bash
pytest
```

## Notes

- MVP scope intentionally excludes query expansion and optimization.
- This project is designed for local non-production use.
