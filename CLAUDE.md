# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python 3.12-based RAG (Retrieval-Augmented Generation) chatbot using LangChain, OpenAI, ChromaDB for vector storage, and Whisper for audio transcription of course materials.

## Setup

**Virtual Environment:**

```bash
python3 -m venv .venv
source .venv/bin/activate  # Linux/MacOS
.venv\Scripts\activate     # Windows
```

**Install Dependencies:**

```bash
pip install -r requirements.txt
```

**Environment Configuration:**

- Copy `.env.example` to `.env` and add your OpenAI API key
- Required: `OPENAI_API_KEY=sk-...`

## Architecture

**Modular RAG Pipeline:**

1. **src/data_loader.py** - Document ingestion (loads PDFs and transcribes MP4 audio via Whisper)
2. **src/embeddings.py** - Vector embedding generation and management
3. **src/retriever.py** - Similarity search over vector database
4. **src/chatbot.py** - Orchestrates the RAG pipeline and conversation flow

**Data Flow:**

```
PDFs (data/) → text extraction → chunks
MP4s (data/) → audio extraction → Whisper transcription → text → chunks
All chunks → embeddings → ChromaDB (embeddings/)
User query → embeddings → retriever → relevant chunks → chatbot + OpenAI → response
```

**Storage:**

- `data/` - Source documents (PDFs and MP4 files) for processing
- `embeddings/` - ChromaDB vector database (created at runtime)

**Key Dependencies:**

- `langchain` - LLM application framework
- `openai` - API client for completions
- `chromadb` - Vector database
- `pypdf` - PDF processing
- `openai-whisper` - Audio transcription

## Development Notes

Project is in early development stage with module stubs in place. Core architecture follows strict separation: load → embed → retrieve → generate.
