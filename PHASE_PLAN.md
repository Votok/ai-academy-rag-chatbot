# AI Academy RAG Chatbot – Implementation Plan

This plan describes how to implement a RAG chatbot for homework/learning purposes. The project ingests course materials from PDFs and MP4 audio files, creates a searchable knowledge base, and answers questions using GPT-4.

**Core Tech Stack:**
- Python 3.12 with `venv`
- OpenAI APIs (GPT-4, Whisper, Embeddings)
- LangChain
- ChromaDB (local vector store)

**Knowledge Sources:**
- PDF documents (text extraction)
- MP4 files (audio extraction → Whisper transcription → text)

---

## Phase 1 – Environment Setup & Dependency Management

**Status:** ✅ **MOSTLY COMPLETE** (venv exists, basic deps installed)

**Goal**
Create a reproducible Python 3.12 development environment with all required dependencies.

**What's Already Done:**
- ✅ Virtual environment (`.venv`) created
- ✅ Basic dependencies installed: `langchain`, `openai`, `chromadb`, `openai-whisper`, `pypdf`, `tiktoken`, `python-dotenv`
- ✅ Module stubs created in `src/`: `__init__.py`, `chatbot.py`, `data_loader.py`, `embeddings.py`, `retriever.py`
- ✅ `.gitignore` configured
- ✅ `.env.example` template exists

**Remaining Tasks:**
1. Update `requirements.txt` with missing packages:
   - `langchain-community` – additional LangChain integrations
   - `langchain-openai` – LangChain OpenAI wrapper
   - `typer` – modern CLI framework
   - `pydantic` – data validation
   - `tqdm` – progress bars
   - `ffmpeg-python` – MP4 audio extraction (requires system `ffmpeg` installed)
2. Install updated dependencies: `pip install -r requirements.txt`
3. Verify system has `ffmpeg` installed (for audio extraction from MP4)
   - macOS: `brew install ffmpeg`
   - Linux: `apt-get install ffmpeg` or `yum install ffmpeg`
   - Windows: Download from ffmpeg.org

**Required Accounts/Credentials:**
- None yet (OpenAI key comes in Phase 2)

**Validation Steps:**
```bash
source .venv/bin/activate
python -c "import openai, langchain, chromadb, whisper, typer, pydantic"
ffmpeg -version  # Verify ffmpeg is installed
```

---

## Phase 2 – Configuration & Secrets Management

**Goal**
Centralize configuration (API keys, model names, paths) using environment variables and a config module.

**Files Involved:**
- New: `src/config.py`
- `.env` (local only, not committed)
- Update: `README.md` with configuration instructions

**Concrete Tasks:**
1. Create `src/config.py` to centralize all configuration:
   - Load environment variables using `python-dotenv`
   - Define configuration parameters:
     - `OPENAI_API_KEY` (required)
     - Model names: `GPT_MODEL` (e.g., "gpt-4"), `EMBEDDING_MODEL` (e.g., "text-embedding-3-small"), `WHISPER_MODEL` (e.g., "whisper-1")
     - Paths: `DATA_DIR` (default: "./data"), `CHROMA_DB_DIR` (default: "./embeddings")
     - Chunking: `CHUNK_SIZE` (default: 1000), `CHUNK_OVERLAP` (default: 200)
     - Retrieval: `TOP_K` (default: 5)
   - Provide clear error messages if required variables are missing
2. Create `.env` file from `.env.example` and add your OpenAI API key
3. Ensure `config.py` is the **only** module that reads `os.environ` directly
4. Update `README.md` with configuration instructions

**Required Accounts/Credentials:**
- **OpenAI API key** (required): Get from https://platform.openai.com/api-keys
  - Add to `.env` as: `OPENAI_API_KEY=sk-...`

**Validation Steps:**
```bash
# Create .env file
cp .env.example .env
# Edit .env and add your real OpenAI API key

# Test config loading
python -c "from src.config import OPENAI_API_KEY, GPT_MODEL; print(f'Using model: {GPT_MODEL}')"
```

---

## Phase 3 – Data Ingestion, Audio Transcription & Chunking

**Goal**
Implement a robust data pipeline that:
1. Loads PDFs and extracts text
2. Extracts audio from MP4 files and transcribes using Whisper
3. Chunks all text into semantically meaningful pieces for embedding

**Files Involved:**
- `src/data_loader.py` – main data ingestion and chunking logic
- `data/` – input files (PDFs and MP4s)
- Update: `README.md` with data format documentation

**Concrete Tasks:**

### Part A: PDF Processing
1. Implement PDF text extraction using `pypdf` or LangChain's `PyPDFLoader`
2. Handle multiple PDFs in `data/` directory
3. Extract metadata: filename, page numbers

### Part B: MP4 Audio Transcription
1. Implement MP4 audio extraction using `ffmpeg-python`:
   - Extract audio track from MP4 → temporary WAV/MP3 file
2. Transcribe audio using OpenAI Whisper API:
   - Use `openai.audio.transcriptions.create()` with model "whisper-1"
   - Handle API rate limits and retries
3. Clean up temporary audio files after transcription
4. Store metadata: filename, transcription timestamp

### Part C: Text Chunking
1. Choose chunking strategy:
   - **Recommended**: LangChain's `RecursiveCharacterTextSplitter`
   - Use `CHUNK_SIZE` and `CHUNK_OVERLAP` from config
2. Implement chunking that:
   - Splits text at natural boundaries (paragraphs, sentences)
   - Retains metadata (source file, page/timestamp)
   - Assigns unique IDs to each chunk
3. Create a `Document` data structure (can use LangChain's `Document` or custom class):
   - `text`: chunk content
   - `metadata`: `{"source": "filename.pdf", "page": 5, "chunk_id": "..."}`

### Part D: Pipeline Orchestration
1. Create a main function `load_and_chunk_documents()` that:
   - Scans `data/` directory for PDFs and MP4s
   - Processes each file type appropriately
   - Returns a list of chunked documents ready for embedding
2. Add progress indicators using `tqdm`
3. Add error handling for corrupted files, API failures

**Required Accounts/Credentials:**
- **OpenAI API key** (for Whisper transcription)

**Validation Steps:**
```bash
# Test data loading and chunking
python -c "
from src.data_loader import load_and_chunk_documents
docs = load_and_chunk_documents()
print(f'Loaded {len(docs)} chunks')
print(f'Sample chunk: {docs[0].text[:200]}...')
print(f'Metadata: {docs[0].metadata}')
"
```

**Expected Output:**
- Multiple chunks from PDFs (text-based)
- Multiple chunks from MP4 transcriptions (audio-based)
- Each chunk has source metadata

---

## Phase 4 – Embeddings & ChromaDB Integration

**Goal**
Generate vector embeddings for all document chunks and store them in ChromaDB for similarity search.

**Files Involved:**
- `src/embeddings.py` – embedding generation logic
- `src/retriever.py` – ChromaDB integration and storage
- `src/config.py` – embedding model configuration

**Concrete Tasks:**

### Part A: Embeddings Module (`src/embeddings.py`)
1. Create a function to initialize OpenAI embeddings client:
   - Use LangChain's `OpenAIEmbeddings` or direct OpenAI API
   - Use `EMBEDDING_MODEL` from config (e.g., "text-embedding-3-small")
2. Implement `embed_texts(texts: list[str]) -> list[list[float]]`:
   - Batch embed multiple texts efficiently
   - Handle API rate limits and retries
   - Return list of embedding vectors
3. Add error handling for API failures

### Part B: ChromaDB Setup (`src/retriever.py`)
1. Initialize ChromaDB client with persistent storage:
   - Use `CHROMA_DB_DIR` from config
   - Create/get a collection (e.g., "course_materials")
2. Implement `index_documents(documents: list[Document])`:
   - Generate embeddings for all document chunks
   - Store in ChromaDB with:
     - `ids`: unique chunk identifiers
     - `embeddings`: vector representations
     - `documents`: original text
     - `metadatas`: source, page/timestamp info
   - Show progress with `tqdm`
3. Handle incremental updates (add new documents without reprocessing all)

### Part C: Index Building Script
1. Create a simple script or CLI command to build the index:
   ```python
   # Can be in src/build_index.py or part of CLI
   from src.data_loader import load_and_chunk_documents
   from src.retriever import index_documents

   docs = load_and_chunk_documents()
   index_documents(docs)
   ```
2. Make it idempotent (safe to run multiple times)

**Required Accounts/Credentials:**
- **OpenAI API key** (for embeddings API)

**Validation Steps:**
```bash
# Build the index
python -c "
from src.data_loader import load_and_chunk_documents
from src.retriever import index_documents

docs = load_and_chunk_documents()
print(f'Processing {len(docs)} chunks...')
index_documents(docs)
print('Index built successfully!')
"

# Verify ChromaDB directory created
ls -la embeddings/
```

**Expected Output:**
- ChromaDB directory created at `./embeddings/`
- All chunks embedded and stored
- Progress indicators during processing

---

## Phase 5 – Retrieval Pipeline

**Goal**
Implement the query retrieval pipeline that finds relevant chunks given a user's question.

**Files Involved:**
- `src/retriever.py` – add similarity search functions
- `src/embeddings.py` – embed queries

**Concrete Tasks:**
1. Implement `retrieve_relevant_chunks(query: str, top_k: int = None) -> list[Document]`:
   - Use `top_k` from config if not provided
   - Embed the query using the embeddings module
   - Perform similarity search in ChromaDB
   - Return top-k most relevant chunks with metadata and similarity scores
2. Add optional score filtering (only return results above a threshold)
3. Format retrieved chunks for easy consumption by LLM:
   - Include source attribution
   - Optionally deduplicate chunks from the same source
4. Add logging to track retrieved documents and scores

**Required Accounts/Credentials:**
- **OpenAI API key** (to embed queries)

**Validation Steps:**
```bash
# Test retrieval
python -c "
from src.retriever import retrieve_relevant_chunks

query = 'What is RAG?'
results = retrieve_relevant_chunks(query, top_k=3)

print(f'Query: {query}')
print(f'Found {len(results)} relevant chunks:\n')
for i, doc in enumerate(results, 1):
    print(f'{i}. Source: {doc.metadata[\"source\"]}')
    print(f'   Text: {doc.text[:200]}...\n')
"
```

**Expected Output:**
- Relevant chunks retrieved for test query
- Source metadata included
- Chunks ranked by relevance

---

## Phase 6 – LLM Integration & Prompt Engineering

**Status:** ✅ **COMPLETE**

**Goal**
Integrate GPT-4 to generate answers using retrieved context, with carefully designed prompts.

**Files Involved:**
- `src/chatbot.py` – main RAG orchestration
- New (optional): `src/prompts.py` – prompt templates
- `src/config.py` – LLM model configuration

**Concrete Tasks:**

### Part A: Prompt Design
1. Create system prompt template that:
   - Instructs the model to be helpful and accurate
   - Emphasizes grounding answers in provided context
   - Requests citations of sources when possible
   - Handles cases where context doesn't contain the answer
2. Create user prompt template that includes:
   - Retrieved context chunks (formatted clearly)
   - User's question
   - Clear separation between context and question

Example structure:
```
System: You are a helpful course assistant. Answer questions based on the provided context.
If the context doesn't contain enough information, say so. Always cite sources.

User:
Context:
---
[Chunk 1 from lecture_notes.pdf, page 5]
RAG stands for Retrieval-Augmented Generation...

[Chunk 2 from video_transcript.txt, timestamp 3:45]
The key benefit of RAG is reduced hallucination...
---

Question: What is RAG and why is it useful?
```

### Part B: LLM Integration
1. In `src/chatbot.py`, implement `generate_answer(query: str) -> dict`:
   - Call retrieval pipeline to get relevant chunks
   - Build prompt using templates
   - Call OpenAI Chat API with GPT-4:
     - Use `GPT_MODEL` from config
     - Set appropriate temperature (e.g., 0.7)
   - Parse response and extract answer
   - Return dict with:
     - `answer`: generated text
     - `sources`: list of source documents used
     - `retrieved_chunks`: the actual context used
2. Add error handling:
   - No relevant context found
   - API rate limits, timeouts
   - Invalid responses
3. Add token counting to track costs
4. Optionally support streaming responses

**Required Accounts/Credentials:**
- **OpenAI API key** (for GPT-4 completions)

**Validation Steps:**
```bash
# Test end-to-end RAG
python -c "
from src.chatbot import generate_answer

query = 'What is RAG?'
result = generate_answer(query)

print(f'Question: {query}')
print(f'Answer: {result[\"answer\"]}')
print(f'\nSources used:')
for source in result['sources']:
    print(f'  - {source}')
"
```

**Expected Output:**
- Coherent answer grounded in context
- Source citations included
- Graceful handling of edge cases

---

## Phase 7 – CLI Interface & Conversation Loop

**Goal**
Create a user-friendly command-line interface for interacting with the RAG chatbot.

**Files Involved:**
- `src/chatbot.py` – extend with CLI logic
- New: `src/cli.py` (optional, for cleaner separation)
- Update: `README.md` with usage instructions

**Concrete Tasks:**

### Part A: CLI Framework
1. Use `typer` to create CLI commands:
   - `build-index`: Rebuild the vector database from scratch
   - `chat`: Start interactive chat session
   - `query`: Single question mode (non-interactive)
2. Add CLI options:
   - `--verbose`: Show retrieved chunks and debug info
   - `--top-k`: Override number of chunks to retrieve
   - `--rebuild`: Force rebuild index before chatting

### Part B: Interactive Chat Loop
1. Implement REPL-style conversation:
   ```
   RAG Chatbot Ready! Type 'exit' to quit, 'help' for commands.

   You: What is RAG?
   Assistant: RAG stands for...
   Sources: lecture_notes.pdf (page 3), video_transcript.txt

   You: How does it work?
   Assistant: ...
   ```
2. Add special commands:
   - `/help`: Show available commands
   - `/sources`: Show all source documents indexed
   - `/stats`: Show index statistics (# chunks, # sources)
   - `/verbose`: Toggle verbose mode
   - `/exit` or `Ctrl+C`: Exit gracefully
3. Handle conversation context:
   - For this homework, simple single-turn Q&A is sufficient
   - Optionally maintain short conversation history (last 2-3 exchanges)

### Part C: Error Handling & UX
1. Pretty print responses (consider using `rich` library for formatting)
2. Show loading indicators during retrieval and generation
3. Handle empty index gracefully (prompt user to run `build-index`)
4. Validate OpenAI API key on startup
5. Display helpful error messages for common issues

**Required Accounts/Credentials:**
- **OpenAI API key** (for embeddings and completions)

**Validation Steps:**
```bash
# Build index
python -m src.chatbot build-index

# Single query
python -m src.chatbot query "What is RAG?"

# Interactive chat
python -m src.chatbot chat
```

**Expected Output:**
- Clean, intuitive CLI interface
- Responsive interaction
- Clear error messages
- Easy to use for testing

---

## Phase 8 – Documentation & Manual Testing

**Goal**
Polish documentation and create a manual testing guide to validate the system through actual usage.

**Files Involved:**
- Update: `README.md` – comprehensive usage guide
- Update: `PHASE_PLAN.md` – mark phases as complete
- New (optional): `TESTING.md` – manual test scenarios

**Concrete Tasks:**

### Part A: README Documentation
1. Expand `README.md` with:
   - **Project Overview**: What it does, what it's for
   - **Prerequisites**: Python 3.12, ffmpeg, OpenAI API key
   - **Setup Instructions**:
     - Clone repo
     - Create venv
     - Install dependencies
     - Configure `.env`
   - **Data Preparation**:
     - Add PDFs and MP4s to `data/` directory
     - Supported formats
   - **Usage**:
     - Build index: `python -m src.chatbot build-index`
     - Ask questions: `python -m src.chatbot query "..."`
     - Interactive mode: `python -m src.chatbot chat`
   - **Architecture Overview**: Brief diagram or description
   - **Troubleshooting**: Common issues and solutions
     - Missing OpenAI API key
     - ffmpeg not installed
     - Empty index
     - Rate limit errors

### Part B: Manual Testing Guide
1. Create test scenarios (can be in README or separate TESTING.md):
   - **Test 1: PDF Content Retrieval**
     - Question: "[Something from your PDFs]"
     - Expected: Accurate answer with PDF source citation
   - **Test 2: Audio/MP4 Content Retrieval**
     - Question: "[Something mentioned in video]"
     - Expected: Answer from transcribed audio, cites MP4 file
   - **Test 3: Cross-Source Answer**
     - Question: "[Topic covered in both PDF and video]"
     - Expected: Synthesizes info from multiple sources
   - **Test 4: Unknown Topic**
     - Question: "What is quantum computing?" (if not in your data)
     - Expected: Admits it's not in the context
   - **Test 5: Verbose Mode**
     - Run with `--verbose` flag
     - Expected: Shows retrieved chunks and debug info
2. Document edge cases to test:
   - Very long questions
   - Ambiguous questions
   - Follow-up questions
   - Typos and informal language

### Part C: Code Cleanup
1. Add docstrings to all functions
2. Add type hints throughout
3. Remove debug print statements
4. Ensure consistent code style
5. Check all error messages are user-friendly

### Part D: Final Validation
1. Test on a fresh environment (if possible):
   - Clone repo to new location
   - Follow README from scratch
   - Verify everything works
2. Run through all test scenarios
3. Check that costs are reasonable (track API usage)
4. Verify all source files are properly cited

**Required Accounts/Credentials:**
- **OpenAI API key** (for final testing)

**Validation Steps:**
```bash
# Follow your own README instructions from scratch
# Run all test scenarios from testing guide
# Verify outputs match expectations
```

**Completion Criteria:**
- ✅ README is comprehensive and accurate
- ✅ Can set up project from scratch following README
- ✅ All manual test scenarios pass
- ✅ Code is clean and well-documented
- ✅ RAG chatbot successfully answers questions about course materials
- ✅ Both PDF and MP4 sources are properly indexed and retrieved

---

## Summary

This 8-phase plan delivers a working RAG chatbot optimized for homework completion:

1. ✅ **Phase 1**: Environment (mostly done)
2. ✅ **Phase 2**: Configuration
3. ✅ **Phase 3**: Data Loading + Chunking (PDF + MP4/Whisper)
4. ✅ **Phase 4**: Embeddings + ChromaDB
5. ✅ **Phase 5**: Retrieval
6. ✅ **Phase 6**: LLM Integration
7. **Phase 7**: CLI
8. **Phase 8**: Documentation + Testing

**Estimated Timeline**: 6-8 focused work sessions

**Core Technologies**: Python 3.12, OpenAI (GPT-4 + Whisper + Embeddings), LangChain, ChromaDB

**Key Features**:
- Multi-format ingestion (PDF + MP4 audio)
- Semantic search with ChromaDB
- GPT-4 powered answers with source citations
- User-friendly CLI interface
- Manual testing for validation

**Intentionally Excluded** (to focus on core homework requirements):
- Formal unit tests (pytest)
- Qdrant Cloud integration
- User audio questions (Whisper for data ingestion only)
- Web UI
- Advanced features (reranking, summarization, etc.)
