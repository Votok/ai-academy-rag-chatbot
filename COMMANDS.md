# RAG Chatbot - Commands Cheat Sheet

This comprehensive guide covers all available commands, when to use them, and what they do.

---

## 📋 Quick Start Workflow

```bash
# 1. Setup (one-time)
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 2. Build the index (first time or when data changes)
python -m src.build_index build

# 3. Query the chatbot
python -m src.chatbot  # Single test query
```

---

## 🔧 Environment & Setup Commands

| Command | When to Use | What It Does |
|---------|-------------|--------------|
| `source .venv/bin/activate` | Every new terminal session | Activates Python virtual environment |
| `pip install -r requirements.txt` | After cloning or updating deps | Installs all Python dependencies |
| `ffmpeg -version` | One-time setup check | Verifies ffmpeg is installed for MP4 audio extraction |

---

## 📊 Index Management Commands

### Building the Index

#### Standard Build (Incremental)
```bash
python -m src.build_index build
```

**When to use:**
- First time setup
- After adding new PDF/MP4 files to `data/` directory
- After modifying existing files in `data/`

**What it does:**
- Loads PDFs and extracts text
- Transcribes MP4 audio (or uses cached transcripts)
- Chunks text into embeddings-ready pieces
- Generates embeddings via OpenAI API
- Stores in ChromaDB vector database
- **Skips already-indexed documents** (fast incremental updates)

**Example output:**
```
Found 1 PDF(s) and 1 MP4(s)
Loading 1 PDF file(s)...
✓ Created 51 chunks from 21 documents
✓ Successfully indexed 51/51 new documents
```

---

#### Rebuild from Scratch
```bash
python -m src.build_index build --rebuild
```

**When to use:**
- Index is corrupted
- You changed chunking parameters (CHUNK_SIZE, CHUNK_OVERLAP)
- You want to start fresh
- You removed files from `data/` directory

**What it does:**
- Deletes all existing indexed documents
- Re-processes all documents from scratch
- Regenerates all embeddings

⚠️ **Warning:** Deletes all existing indexed documents

---

### Checking Index Status

#### Show Statistics
```bash
python -m src.build_index stats
```

**When to use:**
- Before querying (to verify index exists)
- To check how many documents are indexed
- To see storage location
- After building to verify success

**Example output:**
```
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Metric          ┃ Value                      ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Collection Name │ course_materials           │
│ Document Count  │ 51                         │
│ Storage Path    │ /path/to/embeddings        │
└─────────────────┴────────────────────────────┘
```

---

#### Alternative Stats Command
```bash
python -m src.build_index build --stats
```
Same as `stats` command above.

---

### Clearing the Index

```bash
python -m src.build_index clear
```

**When to use:**
- Before doing a complete rebuild
- Cleaning up after testing
- Starting over with new data

**What it does:**
- Prompts for confirmation
- Deletes all documents from ChromaDB
- Recreates empty collection

⚠️ **Warning:** Irreversible operation!

---

## 🧪 Testing & Validation Commands

### Test Individual Modules

#### Test Embeddings Module
```bash
python -m src.embeddings
```

**When to use:**
- Testing OpenAI API key works
- Verifying embedding generation
- Debugging embedding issues

**What it tests:**
- Single query embedding
- Batch text embedding
- Embedding dimensions

---

#### Test Data Loader
```bash
python -m src.data_loader
```

**When to use:**
- Testing if your PDF/MP4 files load correctly
- Checking chunking output
- Debugging transcription issues

**What it tests:**
- PDF text extraction
- MP4 audio transcription
- Text chunking
- Metadata preservation

---

#### Test Retriever
```bash
python -m src.retriever
```

**When to use:**
- Testing ChromaDB connection
- Verifying similarity search works
- Debugging retrieval issues

**What it tests:**
- Document indexing
- Similarity search
- Result formatting

⚠️ **Note:** Creates test documents - run clear_index() after testing

---

### Custom Test Scripts

#### Test Retrieval Quality
```bash
python -c "
from src.retriever import retrieve_relevant_chunks
results = retrieve_relevant_chunks('Your question here', top_k=3)
for doc in results:
    print(f'Source: {doc.metadata[\"source\"]}, Score: {doc.metadata[\"score\"]:.3f}')
    print(f'Text: {doc.page_content[:200]}...\n')
"
```

**When to use:** Testing retrieval quality on your actual indexed data

---

#### Test Configuration
```bash
python -c "from src.config import print_config; print_config()"
```

**When to use:**
- Verifying your .env settings are loaded correctly
- Checking which models are configured
- Debugging configuration issues

---

## 📁 Data Management Commands

### Check Data Directory
```bash
# List files in data directory
ls -lah data/

# Check for cached transcripts
ls -la data/transcripts/
```

**When to use:**
- Verifying your source files are present
- Checking file sizes
- Seeing what's been processed

---

### Manage Transcripts Cache

```bash
# View cached transcripts
cat data/transcripts/your-video.txt

# Remove cached transcripts (force re-transcription)
rm -rf data/transcripts/

# Remove specific transcript
rm data/transcripts/your-video.txt
```

**When to use:**
- MP4 transcription failed and you want to retry
- Transcript quality is poor and you want to regenerate
- Transcript cache is outdated

---

## ⚙️ Configuration Commands

### View Current Configuration
```bash
python -c "from src.config import print_config; print_config()"
```

**Output example:**
```
=== RAG Chatbot Configuration ===
OpenAI API Key: ********************QhsA
GPT Model: gpt-4
Embedding Model: text-embedding-3-small
Whisper Model: whisper-1
Data Directory: /path/to/data
ChromaDB Directory: /path/to/embeddings
Chunk Size: 1000
Chunk Overlap: 200
Top K Results: 5
========================================
```

---

### Verify OpenAI API Connection
```bash
python -c "from src.embeddings import embed_query; print('✓ API key valid, embedding dim:', len(embed_query('test')))"
```

**When to use:**
- Testing API key is valid
- Checking OpenAI connectivity
- Troubleshooting authentication errors

---

### Check ChromaDB Storage
```bash
# View storage directory
ls -la embeddings/

# Check disk space usage
du -sh embeddings/

# View ChromaDB files
find embeddings/ -type f
```

**When to use:**
- Verifying index was created
- Checking disk space usage
- Troubleshooting persistence issues

---

## 🔬 Environment Variable Overrides

You can temporarily override configuration parameters without editing `.env`:

### Different Chunk Size
```bash
CHUNK_SIZE=500 CHUNK_OVERLAP=100 python -m src.build_index build --rebuild
```

**When to use:** Testing optimal chunk sizes for your documents

---

### Different Top-K Results
```bash
TOP_K=10 python -c "from src.retriever import retrieve_relevant_chunks; ..."
```

**When to use:** Experimenting with retrieval result counts

---

### Different Embedding Model
```bash
EMBEDDING_MODEL=text-embedding-3-large python -m src.build_index build --rebuild
```

**When to use:** Testing different embedding models for quality/cost tradeoffs

⚠️ **Note:** Changing `CHUNK_SIZE` or `EMBEDDING_MODEL` requires `--rebuild` to reprocess all documents

---

## 🚨 Troubleshooting Commands

### Problem: API Rate Limit Errors

**Symptoms:**
- "Rate limit exceeded" errors
- 429 status codes

**Solution:**
```bash
# Wait and retry - automatic exponential backoff is built-in
python -m src.build_index build
```

The system automatically retries with exponential backoff (up to 5 attempts).

---

### Problem: MP4 Transcription Fails

**Symptoms:**
- "Whisper API error" messages
- Empty transcripts

**Debug steps:**
```bash
# 1. Check ffmpeg installation
ffmpeg -version

# 2. Verify MP4 file is valid
ffmpeg -i data/your-file.mp4

# 3. Check MP4 file isn't too large
ls -lh data/your-file.mp4

# 4. Force re-transcription
rm -rf data/transcripts/
python -m src.build_index build --rebuild
```

**Common causes:**
- ffmpeg not installed
- Corrupted MP4 file
- File over 25MB (requires chunking)
- OpenAI API timeout (retry automatically handles this)

---

### Problem: Index Not Found

**Symptoms:**
- "Index is empty" errors
- "Collection not found" errors

**Solution:**
```bash
# Check if index exists
python -m src.build_index stats

# If empty or missing, build it
python -m src.build_index build
```

---

### Problem: Out-of-Date Index

**Symptoms:**
- Query results don't include new documents
- Old document versions being returned

**Solution:**
```bash
# Incremental update (fast)
python -m src.build_index build

# Or full rebuild if index seems corrupted
python -m src.build_index build --rebuild
```

---

### Problem: Poor Retrieval Quality

**Symptoms:**
- Irrelevant results
- Missing relevant content

**Debug & Fix:**
```bash
# 1. Test query manually
python -c "
from src.retriever import retrieve_relevant_chunks, format_retrieved_chunks
results = retrieve_relevant_chunks('your query', top_k=10)
print(format_retrieved_chunks(results, include_scores=True))
"

# 2. Try more results
TOP_K=10 python -c "..."

# 3. Adjust chunk size (requires rebuild)
# Edit .env: CHUNK_SIZE=1500
python -m src.build_index build --rebuild

# 4. Try different embedding model
EMBEDDING_MODEL=text-embedding-3-large python -m src.build_index build --rebuild
```

---

## 📖 Common Workflows

### Workflow 1: First Time Setup

```bash
# 1. Setup environment
cp .env.example .env
# Edit .env file with your OPENAI_API_KEY=sk-...

# 2. Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Verify setup
python -c "from src.config import print_config; print_config()"

# 4. Add data files to data/ directory
# Copy your PDFs and MP4s here

# 5. Build index
python -m src.build_index build

# 6. Verify index
python -m src.build_index stats

# 7. Test retrieval
python -c "
from src.retriever import retrieve_relevant_chunks
results = retrieve_relevant_chunks('test query', top_k=3)
print(f'Found {len(results)} results')
"
```

---

### Workflow 2: Adding New Documents

```bash
# 1. Add new PDF/MP4 files to data/ directory
cp ~/Downloads/new-document.pdf data/

# 2. Run incremental build (fast - skips existing)
python -m src.build_index build

# 3. Verify new docs added
python -m src.build_index stats
```

---

### Workflow 3: Changed Parameters (Chunk Size, etc.)

```bash
# 1. Edit .env file
# Set CHUNK_SIZE=1500
# Set CHUNK_OVERLAP=300

# 2. Rebuild from scratch (required for parameter changes)
python -m src.build_index build --rebuild

# 3. Verify
python -m src.build_index stats

# 4. Test new chunk size effectiveness
python -c "
from src.retriever import retrieve_relevant_chunks
results = retrieve_relevant_chunks('test query', top_k=3)
for r in results:
    print(f'Chunk length: {len(r.page_content)} chars')
"
```

---

### Workflow 4: Debugging Retrieval Quality

```bash
# 1. Test specific query with details
python -c "
from src.retriever import retrieve_relevant_chunks, format_retrieved_chunks
results = retrieve_relevant_chunks('What is a vector database?', top_k=5)
print(format_retrieved_chunks(results, include_scores=True))
"

# 2. Analyze score distribution
python -c "
from src.retriever import retrieve_relevant_chunks
results = retrieve_relevant_chunks('your query', top_k=20)
scores = [r.metadata['score'] for r in results]
print(f'Score range: {min(scores):.3f} - {max(scores):.3f}')
print(f'Average: {sum(scores)/len(scores):.3f}')
"

# 3. If results poor, try more results
TOP_K=10 python -c "..."

# 4. Or adjust chunk size and rebuild
# Edit .env: CHUNK_SIZE=1500
python -m src.build_index build --rebuild
```

---

### Workflow 5: Fresh Start (Clean Slate)

```bash
# 1. Clear all indexed data
python -m src.build_index clear

# 2. Remove cached transcripts (optional)
rm -rf data/transcripts/

# 3. Rebuild everything from scratch
python -m src.build_index build

# 4. Verify
python -m src.build_index stats
```

---

## 🔐 Security & Cleanup

### Never Commit Sensitive Files
```bash
# Verify .gitignore is protecting sensitive files
cat .gitignore

# Should include:
# .env
# embeddings/
# data/transcripts/
```

---

### Clean Up Development/Test Data
```bash
# Clear indexed documents
python -m src.build_index clear

# Remove transcript cache
rm -rf data/transcripts/

# Remove ChromaDB storage (complete wipe)
rm -rf embeddings/
```

---

### Verify No Secrets in Repository
```bash
# Check for accidentally committed secrets
git status
git diff

# NEVER commit:
# - .env file
# - API keys
# - embeddings/ directory
# - Large data files
```

---

## 📝 Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│  ESSENTIAL COMMANDS                                         │
├─────────────────────────────────────────────────────────────┤
│  Build index:           python -m src.build_index build     │
│  Show stats:            python -m src.build_index stats     │
│  Rebuild from scratch:  python -m src.build_index build -r  │
│  Clear everything:      python -m src.build_index clear     │
│                                                              │
│  Test embeddings:       python -m src.embeddings            │
│  Test data loader:      python -m src.data_loader           │
│  Test retriever:        python -m src.retriever             │
│                                                              │
│  Show config:           python -c "from src.config import \ │
│                         print_config; print_config()"       │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Command Categories Summary

| Category | Commands | Purpose |
|----------|----------|---------|
| **Setup** | `cp .env.example .env`, `pip install` | Initial configuration |
| **Indexing** | `build`, `build --rebuild` | Create/update search index |
| **Monitoring** | `stats`, `print_config()` | Check system status |
| **Testing** | `python -m src.*` modules | Validate components |
| **Maintenance** | `clear`, `rm -rf transcripts/` | Clean up data |
| **Debugging** | Custom Python scripts | Troubleshoot issues |

---

## 📚 Additional Resources

- **Project Plan:** See `PHASE_PLAN.md` for complete implementation roadmap
- **Configuration:** See `.env.example` for all available settings
- **Code Documentation:** All modules have docstrings - use `help(module)`

---

## 💬 Chatbot Commands (Phase 6+)

### Test Chatbot Module
```bash
python -m src.chatbot
```

**When to use:**
- Quick test to verify chatbot is working
- Tests with a sample query ("What is RAG?")
- Validates end-to-end RAG pipeline

**What it does:**
- Retrieves relevant chunks for test query
- Generates answer using GPT-4
- Shows answer, sources, and metadata

**Example output:**
```
============================================================
Testing RAG Chatbot - generate_answer()
============================================================

Query: What is RAG?

Answer:
------------------------------------------------------------
RAG (Retrieval-Augmented Generation) is a technique that...
------------------------------------------------------------

Sources used: lecture_notes.pdf, video_transcript.txt
Model: gpt-4
Retrieved chunks: 3

✓ Test successful!
```

---

### Query Programmatically (Python)

#### Simple Query
```python
from src.chatbot import generate_answer

result = generate_answer("What is RAG?")
print(result["answer"])
print("Sources:", result["sources"])
```

**When to use:**
- Integrating chatbot into your own scripts
- Batch processing multiple queries
- Custom post-processing of results

---

#### Advanced Query with Options
```python
from src.chatbot import generate_answer

result = generate_answer(
    query="Explain vector embeddings",
    top_k=10,           # Retrieve more chunks
    min_score=0.7,      # Only use high-quality matches
    temperature=0.5     # More focused/deterministic answers
)

print(result["answer"])

# Access retrieved chunks
for chunk in result["retrieved_chunks"]:
    print(f"- {chunk.metadata['source']}: {chunk.metadata['score']:.3f}")
```

**Parameters:**
- `query` (required): Your question as a string
- `top_k` (optional): Number of chunks to retrieve (default: 5)
- `min_score` (optional): Minimum similarity score (0.0-1.0)
- `temperature` (optional): LLM creativity (0.0=focused, 1.0=creative, default: 0.7)

**Returns dict with:**
- `answer`: Generated response text
- `sources`: List of source filenames used
- `retrieved_chunks`: Full Document objects with metadata
- `query`: Your original query
- `model`: Model name used (e.g., "gpt-4")

---

### Using the RAGChatbot Class

For maintaining state or custom configuration:

```python
from src.chatbot import RAGChatbot

# Initialize once
chatbot = RAGChatbot(
    model="gpt-4",     # or "gpt-3.5-turbo" for faster/cheaper
    top_k=5
)

# Ask multiple questions (reuses config)
result1 = chatbot.generate_answer("What is RAG?")
result2 = chatbot.generate_answer("How do vector databases work?")
result3 = chatbot.generate_answer("Explain embeddings", top_k=10)
```

**When to use:**
- Multiple queries in a session
- Custom model configuration
- Building your own chat interface

---

### Common Query Patterns

#### Basic Question Answering
```python
from src.chatbot import generate_answer

questions = [
    "What is RAG?",
    "How do embeddings work?",
    "What is ChromaDB?",
]

for q in questions:
    result = generate_answer(q)
    print(f"Q: {q}")
    print(f"A: {result['answer']}\n")
```

---

#### Detailed Investigation (More Context)
```python
# Retrieve more chunks for complex questions
result = generate_answer(
    "Compare and contrast RAG vs fine-tuning",
    top_k=10  # Get more context
)
```

---

#### High-Precision Answers (Strict Matching)
```python
# Only use very relevant chunks
result = generate_answer(
    "What is the exact formula for cosine similarity?",
    min_score=0.8,      # High threshold
    temperature=0.3     # Low temperature for precision
)
```

---

#### Creative Explanations
```python
# More creative/elaborative answers
result = generate_answer(
    "Explain RAG to a beginner",
    temperature=0.9  # Higher temperature for creativity
)
```

---

### Inspecting Retrieved Context

```python
from src.chatbot import generate_answer

result = generate_answer("What is RAG?", top_k=5)

# See which sources were used
print("Sources:", result["sources"])

# Inspect retrieved chunks
for i, chunk in enumerate(result["retrieved_chunks"], 1):
    meta = chunk.metadata
    print(f"\n{i}. {meta['source']} (score: {meta['score']:.3f})")
    print(f"   {chunk.page_content[:150]}...")
```

**When to use:**
- Debugging why an answer was generated
- Verifying source quality
- Understanding what context was provided to LLM

---

### Error Handling

```python
from src.chatbot import generate_answer

try:
    result = generate_answer("Your question")
    print(result["answer"])
except ValueError as e:
    print(f"Query error: {e}")
except Exception as e:
    print(f"API error: {e}")
```

**Common errors:**
- `ValueError`: Empty query or retrieval failed
- `Exception`: OpenAI API failure (rate limit, network, invalid key)

---

## 🧪 Testing Chatbot

### Verify Phase 6 Complete
```bash
# Test all Phase 6 components
python -m src.prompts   # Should have no output (import test)
python -m src.chatbot   # Should generate answer to test query
```

---

### Custom Test Queries
```bash
python -c "
from src.chatbot import generate_answer
result = generate_answer('Your custom question here')
print(result['answer'])
print('\\nSources:', ', '.join(result['sources']))
"
```

---

## 📊 Quick Reference Update

```
┌─────────────────────────────────────────────────────────────┐
│  ESSENTIAL COMMANDS                                         │
├─────────────────────────────────────────────────────────────┤
│  Build index:           python -m src.build_index build     │
│  Show stats:            python -m src.build_index stats     │
│  Rebuild from scratch:  python -m src.build_index build -r  │
│  Clear everything:      python -m src.build_index clear     │
│                                                              │
│  Test chatbot:          python -m src.chatbot               │
│  Test embeddings:       python -m src.embeddings            │
│  Test data loader:      python -m src.data_loader           │
│  Test retriever:        python -m src.retriever             │
│                                                              │
│  Show config:           python -c "from src.config import \ │
│                         print_config; print_config()"       │
└─────────────────────────────────────────────────────────────┘
```

---

**Last Updated:** Phase 6 Complete (LLM Integration & Prompt Engineering)
