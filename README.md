# rag-system MVP

CLI Retrieval-Augmented Generation (RAG) system for a single Jane Austen book PDF.

# Design


# Design

**Chucking strategy**

- A combination of size limit and paragraph, keep going until you reach a paragraph, but if the next paragraph will split, then stop
- This way tokens are consistent but also we don’t lose context by breaking off paragraphs halfway
- I will need to convert to a txt or md file as well to make it easier to process

Embeddings

| Model | Pros | Cons |
| --- | --- | --- |
| BERT |   • Bi-directional 
  • Captures contextual relationships between words
  • Lower latency |   • Requires GPU costs
  • Doesn’t handle a larger input text
  • Considered more of an architecture that needs tuning than  a service |
| Embeddings API |   • Larger token window
  • Ready to use service
  •  |   • Risk of PII data being input
  • API costs could be larger |

TradeOff made: Prioritised a ready-to-use service over custom infrastructure. The meaning of the words are not actually useful as I am not doing any NLP;  rather with the embeddings API a RAG system can be created without extensive fine tuning and can scale much quicker

**Storage**:

| DB | Pros | Cons |
| --- | --- | --- |
| SQL LIte |   • Supports vector stoage
  • Simply to set up |   • Not scalable at all |
| **Chroma** |   • Integrates well with AI code packages such as LangChain
  •  |   • Not scalable as well |
|  |  |  |

Design decision:

- Go with SQL Lite for the POC to focus on creating the RAG system
- Then for large scale usage and to allow for multiple documents being stored I would use vespa because if the retrieval process also becomes multi-layered with Cross - Encoder models, this database can handle this and has low latency

Retrieval:

| Method | Pros | Cons |
| --- | --- | --- |
|  |  |  |
|  |  |  |

TradeOff: Prioritised Semantic meaning over exact matching terms as the questions being asked won’t always be in the language of the documents. 

# Tables required

1. Document Storage Table
2. Embeddings storage
3. History of all the questions 
4. Feedback/ User rankings 

# How to scale long term

1. Have an ETL mechanism to load in various formats of data and standardise them into markdowns
2. Implement the hybrid B25 and Cosine retrieval method, adding second layers if we see issues with ordering of the embeddings
3. Add output guardrails as well (I currently have input)
4. Add authentication to control usage of the RAG
5. Rate limiting per users is required to prevent spamming 
6. Monitoring of hallucinations, or messages with ‘I don’t know’

## User Guardrails

`ask` questions are scanned for structured PII **before** retrieval or LLM calls, using LangChain email, credit card, IP address, MAC address, URL, API keys/tokens, phone numbers, UK postcodes, US-style addresses, and US SSN-style IDs.




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
│   ├── user_guardrails.py
│   └── vector_store.py
└── tests/
    ├── test_chunker.py
    ├── test_pdf_to_md.py
    ├── test_retriever.py
    └── test_user_guardrails.py
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
   - `PII_GUARDRAILS_ENABLED` (default `true`)
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
