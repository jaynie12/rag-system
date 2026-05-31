# rag-system MVP

CLI Retrieval-Augmented Generation (RAG) system for a single Jane Austen book PDF.

## Features

# Design

# AI Systems Crash Guide (LLM + RAG + MCP + Agents)

---

## Goal

Understand how modern AI systems are built by combining:

- LLM → reasoning
- RAG → knowledge ( Retrieval Augmented Generation )
- Tools / MCP → actions
- Agent loop → decision-making

---

## Core Concepts

### 1. LLM (the “brain” , the generator)

- Takes input → generates output
- Good at reasoning, not reliable for factual accuracy

Key idea:

> LLMs should reason, not store truth
> 

---

### 2. RAG (Retrieval-Augmented Generation)

```
User → retrieve documents → LLM → generates answer
```

Use when:

- you need correctness
- data changes frequently
- domain knowledge matters

Key idea:

> RAG gives the model access to external knowledge
> 

---

### 3. Tools (Function Calling / MCP)

Tools allow the model to take actions.

Examples:

- get_weather
- search_hotels
- get_metrics

Key idea:

> Tools provide capabilities, not knowledge
> 

---

### 4. MCP (Model Context Protocol)

Standard way to expose tools and resources to LLMs

```
MCP server → exposes tools
Host → connects to server
LLM → decides when to use tools
```

Key idea:

> MCP standardizes how tools are exposed and invoked
> 

---

### 5. Agents (the loop)

```python
while not done:
    think → act → observe → repeat
```

Key idea:

> Agent = LLM + tools + loop + state
> 

---

## Putting it all together

```
User
 ↓
LLM (reasoning)
 ↓
RAG (knowledge)
 ↓
Tools (actions via MCP)
 ↓
Agent loop (multi-step)
```

---

## Must-Read Resources

RAG:

https://platform.openai.com/docs/guides/retrieval

https://galileo.ai/blog/mastering-rag-how-to-architect-an-enterprise-rag-system

OpenAI Agents:

https://platform.openai.com/docs/guides/agents

Core agent design:

https://www.anthropic.com/research/building-effective-agents

MCP:

https://modelcontextprotocol.io/docs/getting-started/intro

# Build a RAG System

## Objective

Build and Design a working Retrieval-Augmented Generation (RAG) system that answers questions using external documents.

---

## Requirements

### Step 1 — Data ingestion

- Choose a dataset:
    - 3–10 text files, OR
    - a single long document (e.g., a book chapter, blog posts, or technical docs)
- Preprocess:
    - split into chunks (200–500 words each)
    - store chunks in memory or file

---

### Step 2 — Embeddings

- Generate embeddings for each chunk
- Store:
    - chunk text
    - embedding vector

You can use:

- OpenAI embeddings API
- or any embedding model
- LLM Embeddings Explained: A Visual and Intuitive Guide - a Hugging Face Space by hesamation

---

### Step 3 — Retrieval

Given a user query:

- embed the query
- compute similarity (cosine similarity)
- retrieve top-k chunks (k = 3–5)

---

### Step 4 — Generation

Construct a prompt:

```
Answer the question using ONLY the context below.

Context:
[retrieved chunks]

Question:
[user question]
```

- Call LLM
- return answer

---

### Step 5 — Constraints

The system must:

- only answer from retrieved content
- say “I don’t know” if answer is not found
- not hallucinate

## Homework structure learning RAG

```python
rag-homework/
├── README.md
├── requirements.txt
├── .env ( you place your open api key here ) 
├── data/
│   ├── doc1.txt
│   ├── doc2.txt
│   └── doc3.txt
├── storage/
│   └──  ( it can be sqlite files or generated artifacts i.e embeddings, etc ) 
├── src/
│   ├── __init__.py
│   ├── config.py ( reads the .env file maybe with dotenv lib , model, chunk size etc ) 
│   ├── loader.py  ( loads documents , reads txt files for eample ) 
│   ├── chunker.py
│   ├── embeddings.py
│   ├── pdfToMd.py
│   ├── vector_store.py ( you can use a vector db like faiss) 
│   ├── retriever.py ( searches the index / vector db and finds docs ) 
│   ├── generator.py ( uses an LLM model to generate answer ) 
│   ├── rag_pipeline.py ( two main methods builds index and generates answer ) 
│   └── cli.py
└── tests/
    ├── test_chunker.py
    └── test_retriever.py
```

## requirements

openai

python-dotenv

numpy

faiss-cpu

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
Choose Cosine similarity

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
