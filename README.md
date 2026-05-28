# rag-system

CLI Retrieval-Augmented Generation (RAG) system for a single Jane Austen book PDF.

## Features

# Design

**Chucking strategy**

- A combination of size limit and paragraph, keep going until you reach a paragraph, but if the next paragraph will split, then stop
- This way tokens are consistent but also we don’t lose context by breaking off paragraphs halfway
- I will need to convert to a txt or md file as well to make it easier to process

Embeddings

- BERT model might reduce latency, but that is a lower priority for me here, as this RAG system is not time sensitive (just looking up book)
- I don’t have an PII data as well so its ok for data to be exposed online
- OPEN API Embeddings API is quicker for a POC and learnings

Storage:

- SQLite DB  is quick and easy to set up for a non-prod usage
- Can implement vector storage as well with this

Retrieval:

- I care about the meaning not the distance between them so Cosine similarity is being used

Question generation technique's:

- Query Expansion: Expands the query by adding synonyms, related terms, or domain-specific keywords to improve recall.
- Query Reformulation: Rewrites the query using LLM-based paraphrasing to improve structure and clarity.

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
