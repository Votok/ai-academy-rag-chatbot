# AI Academy RAG Chatbot – Multi-Phase Implementation Plan

This plan describes how to implement the full RAG chatbot in clearly separated phases. Each phase is intended to be executed as a separate Claude Code session, focusing on a limited, coherent set of changes.

Core tech stack:
- Python 3.12 with `venv`
- OpenAI APIs (GPT-4, Whisper, Embeddings)
- LangChain
- ChromaDB (local vector store)
- Optional: Qdrant Cloud

---

## Phase 0 – Repository Familiarization & High-Level Design

**Goal**  
Understand the existing skeleton, define high-level architecture, and confirm project scope and conventions.

**Files Involved**
- `README.md`
- `PHASE_PLAN.md` (this file)
- `src/__init__.py`
- `src/chatbot.py`
- `src/data_loader.py`
- `src/embeddings.py`
- `src/retriever.py`
- `data/` (structure and sample content)

**Concrete Tasks**
1. Review the current repo structure and any course requirements (outside this repo) to align terminology and milestones.
2. Decide on the architectural style:
   - Clear separation between **infrastructure** (API clients, vector DB) and **domain logic** (retrieval, conversation orchestration).
   - Plan for configuration via environment variables and/or a central config module.
3. Define the main runtime entry point (likely `src/chatbot.py`) and how it will be used (CLI script vs. notebook vs. web endpoint).
4. Sketch the main modules and their responsibilities (in prose, not code):
   - `data_loader` – responsible for loading and preprocessing raw data files.
   - `embeddings` – responsible for embedding model selection and encoding operations.
   - `retriever` – responsible for chunk storage, vector DB integration, and retrieval logic.
   - `chatbot` – responsible for conversation loop, RAG orchestration and LLM calls.
5. Decide what you want to support for the homework:
   - Text documents only vs. also audio (Whisper) and/or PDFs.
   - Single-user local CLI vs. optional web interface.

**Required Accounts / Credentials**
- None required in this phase, but identify upcoming needs:
  - OpenAI API key (mandatory later).
  - Optional Qdrant Cloud account.

**Validation Steps**
- Confirm that the planned module responsibilities are documented in `README.md` or a short design section in this `PHASE_PLAN.md` (no implementation yet).
- Ensure you can run basic Python commands in the environment and import `src` without errors (if there is existing code):
  ```bash
  python -c "import src"
  ```

---

## Phase 1 – Environment Setup & Dependency Management

**Goal**  
Create a reproducible Python 3.12 development environment with all core dependencies installed and version-pinned.

**Files Involved**
- `requirements.txt`
- (Optional) `README.md` – add setup instructions.

**Concrete Tasks**
1. Create and activate a Python 3.12 virtual environment:
   - Decide on a name (e.g., `.venv`).
2. Update `requirements.txt` with necessary packages (exact versions to be chosen later during the implementation phase):
   - `openai` (or `openai>=1.x` client library).
   - `langchain`, `langchain-community`, `langchain-openai`.
   - `chromadb`.
   - `qdrant-client` (optional for Qdrant integration).
   - Utility packages: `python-dotenv`, `pydantic`, `typer` or `click` (if you want CLI), `tqdm`, etc.
   - Testing: `pytest` (and optionally `pytest-asyncio`).
3. Install dependencies in the venv using `pip`.
4. Configure `.gitignore` (if not already in repo) to exclude `.venv/`, local databases, and environment files (e.g., `.env`).
5. Document environment setup steps in `README.md` (commands to create venv, install deps, run basic scripts).

**Required Accounts / Credentials**
- None required yet (OpenAI key not yet used), but optionally:
  - Create an OpenAI account and obtain an API key ready for later use.

**Validation Steps**
- Confirm venv activation and package importability:
  ```bash
  source .venv/bin/activate
  python -c "import openai, langchain, chromadb"
  ```
- Ensure `requirements.txt` and `README.md` are consistent with the environment setup.

---

## Phase 2 – Configuration & Secrets Management

**Goal**  
Centralize configuration (API keys, model names, paths) and secrets management using environment variables and/or a `.env` file.

**Files Involved**
- New: `src/config.py`
- `.env` (not committed; local only)
- `README.md` – add configuration instructions.

**Concrete Tasks**
1. Define configuration parameters in `src/config.py` (no implementation code here in the plan, only describe):
   - `OPENAI_API_KEY` (required).
   - Default model names: `GPT_4_MODEL`, `EMBEDDING_MODEL`.
   - Paths: `DATA_DIR`, `CHROMA_DB_DIR`.
   - Feature flags: `USE_QDRANT`, `QDRANT_URL`, `QDRANT_API_KEY`, `COLLECTION_NAME`.
2. Decide configuration strategy:
   - Use `python-dotenv` to load `.env` in development.
   - Fallback to standard environment variables in production.
3. Document how to create `.env` in `README.md`:
   - Example variable names (no real keys).
4. Ensure `src/config.py` is the **only** module that directly reads environment variables; other modules should import configuration values from it.

**Required Accounts / Credentials**
- **OpenAI API key** (to be stored in `.env` and environment):
  - `OPENAI_API_KEY`.
- **Optional Qdrant Cloud**:
  - `QDRANT_URL` and `QDRANT_API_KEY` if using hosted Qdrant instead of local Chroma.

**Validation Steps**
- Create a local `.env` file (do not commit) and populate it with dummy or real values.
- Verify environment loading in an interactive Python session, e.g. (pseudocode, no full implementation here):
  - Import `config` and print out a known value to confirm `.env` is loaded.
- Confirm that no other module directly accesses `os.environ` (by design, after implementation).

---

## Phase 3 – Data Ingestion & Preprocessing

**Goal**  
Implement a robust data ingestion pipeline that reads raw course/material files from `data/`, normalizes them to text, and cleans them for downstream chunking.

**Files Involved**
- `data/` – input files (Markdown, text, PDF, etc.).
- `src/data_loader.py` – main data ingestion logic.
- (Optional) New: `src/types.py` for shared data structures (e.g., `Document` type alias or simple dataclass).
- `README.md` – document expected data formats and locations.

**Concrete Tasks**
1. Inventory data formats you plan to support (likely at least `.md`, `.txt`, maybe `.pdf`).
2. In `src/data_loader.py`, design functions (names only at this phase) for:
   - Loading all documents from `data/` recursively.
   - Normalizing paths and metadata (source filename, section, etc.).
   - Optional separate loaders per file type (e.g., `load_markdown`, `load_pdf`).
3. Define a simple internal document representation (e.g., `Content`, `Metadata`) that other modules can use.
4. Decide whether to use LangChain document loaders vs. custom I/O logic.
5. Document how new data should be added to `data/` to be picked up by the pipeline.

**Required Accounts / Credentials**
- None (local data only).

**Validation Steps**
- Run a small script or REPL session to:
  - Load all data via `data_loader` and count the number of documents.
  - Print out a sample document’s text length and metadata.
- Confirm robust handling of missing or empty files (e.g., clear errors or warnings).

---

## Phase 4 – Text Chunking Strategy

**Goal**  
Define and implement a chunking strategy that splits documents into semantically useful chunks for embeddings and retrieval.

**Files Involved**
- `src/data_loader.py` – may be extended to include or call chunking functions.
- New (optional): `src/chunking.py` – if you want to isolate chunking logic.
- `README.md` – brief description of chunking choice.

**Concrete Tasks**
1. Choose chunking strategy:
   - Simple fixed-size window (e.g., N characters or tokens) with overlap.
   - Or a more advanced approach (e.g., based on headings, paragraphs, LangChain text splitters).
2. Design functions in `chunking` (or extended `data_loader`) for:
   - Converting loaded documents into chunks.
   - Retaining metadata (e.g., source, section, page) at the chunk level.
3. Decide on configuration parameters for chunking (to be stored in `config`):
   - `CHUNK_SIZE`, `CHUNK_OVERLAP`, optional `SPLITTER_TYPE`.
4. Decide how chunking fits into the pipeline:
   - A separate script/step that transforms raw documents into chunks stored in memory only.
   - Or persist chunked documents (e.g., JSON) for reuse.
5. Add to `README.md` a short rationale for your chunking parameters and how to adjust them.

**Required Accounts / Credentials**
- None.

**Validation Steps**
- Execute the chunking pipeline and verify:
  - Total number of chunks.
  - Example chunk text and metadata look coherent (no mid-word splits, etc., unless acceptable).
- Experiment with adjusting `CHUNK_SIZE` and `CHUNK_OVERLAP` and confirm it changes the output as expected.

---

## Phase 5 – Embeddings Module Design (OpenAI)

**Goal**  
Design and implement a reusable embeddings module using OpenAI’s embedding API via LangChain or direct client calls.

**Files Involved**
- `src/embeddings.py` – core embedding logic.
- `src/config.py` – embedding model name and related configuration.

**Concrete Tasks**
1. Select the primary embedding model (e.g., OpenAI’s `text-embedding-3-small` or similar) and record that in `config`.
2. Define a clean interface in `embeddings` for:
   - Initializing the embedding client (OpenAI or LangChain wrapper).
   - A function to embed a list of texts and return vectors.
3. Decide whether to use LangChain’s `OpenAIEmbeddings` or a direct OpenAI client call, and document the choice.
4. Ensure the embeddings module does **not** depend on the vector store implementation details (Chroma/Qdrant) – it should only produce vectors.
5. Consider batching logic and rate limiting strategies at design time.

**Required Accounts / Credentials**
- **OpenAI API key** – required to call the embedding API.

**Validation Steps**
- With real (or limited) OpenAI credentials, run a small test:
  - Embed a few sample strings and verify the returned vector dimensions.
  - Measure approximate latency per batch to understand performance.
- Check that errors (e.g., missing API key, rate limits) are surfaced clearly and not swallowed silently.

---

## Phase 6 – Local Vector Store with ChromaDB

**Goal**  
Integrate ChromaDB as the primary local vector store for storing and querying embeddings.

**Files Involved**
- `src/retriever.py` – vector store integration and retrieval logic.
- `src/embeddings.py` – used by the vector store but should remain independent.
- `src/config.py` – Chroma configuration (db directory, collection name).

**Concrete Tasks**
1. Decide how to persist Chroma:
   - Directory path under project root (e.g., `./chroma_db/`) configured in `config`.
2. In `retriever`, define responsibilities:
   - Initialize a Chroma collection (create if not exists).
   - Add documents/chunks with embeddings and metadata.
   - Perform similarity search (top-k retrieval) for a query text.
3. Decide on how IDs and metadata are structured in the vector store (e.g., using hashed IDs or incremental IDs, storing `source`, `chunk_index`, etc.).
4. Design a process for (re)building the Chroma index:
   - A dedicated ingestion script or CLI command that:
     - Loads and chunks documents.
     - Embeds chunks.
     - Upserts to Chroma.
5. Make sure Chroma-specific code is wrapped behind an abstraction so that Qdrant can be swapped in later without rewriting the chatbot logic.

**Required Accounts / Credentials**
- None for local ChromaDB (no external service).

**Validation Steps**
- Build a small index using a subset of documents.
- Run similarity search for a known query and manually inspect the top results.
- Delete and rebuild the Chroma directory to confirm reproducibility.

---

## Phase 7 – Optional Qdrant Cloud Integration

**Goal**  
Add an optional alternative vector store backend using Qdrant Cloud, configurable via `config`.

**Files Involved**
- `src/retriever.py` – extend to support Qdrant adapter.
- `src/config.py` – Qdrant configuration values.
- (Optional) New: `src/vector_stores.py` – abstract base class / interface for vector stores if you want cleaner separation.

**Concrete Tasks**
1. Model the vector store integration around an abstract interface:
   - Methods like `add_documents`, `similarity_search`, etc.
2. Implement a Qdrant-based adapter using `qdrant-client` (or LangChain Qdrant integration), separate from the Chroma implementation.
3. Add configuration to toggle between `CHROMA` and `QDRANT` backends via `config`.
4. Ensure both backends share the same document/chunk metadata schema.
5. Document the trade-offs and how to switch between vector stores in `README.md`.

**Required Accounts / Credentials**
- **Qdrant Cloud** (optional):
  - Qdrant Cloud account.
  - `QDRANT_URL`, `QDRANT_API_KEY` variables set in `.env`.

**Validation Steps**
- With Qdrant credentials, create a test collection and index a small set of chunks.
- Run a test query and verify that it returns results comparable to Chroma for the same data.
- Toggle between backends via configuration and confirm behavior without code changes elsewhere.

---

## Phase 8 – RAG Retrieval Pipeline & Query Orchestration

**Goal**  
Implement the end-to-end retrieval pipeline that, given a user’s natural language question, retrieves relevant context chunks from the vector store.

**Files Involved**
- `src/retriever.py` – finalize retrieval functions.
- `src/embeddings.py` – used to embed queries.
- `src/data_loader.py` / `src/chunking.py` – as part of ingestion/rebuild steps.
- New (optional): `src/pipeline.py` – high-level orchestration utilities.

**Concrete Tasks**
1. Define a function that:
   - Accepts a user query.
   - Embeds the query.
   - Calls the vector store to retrieve top-k chunks.
   - Returns a structured object containing retrieved texts and metadata.
2. Decide on retrieval hyperparameters:
   - `TOP_K`, optional score threshold.
3. Implement a simple pipeline function that combines ingestion/chunking/embedding/indexing into a reproducible flow (even if executed as separate commands).
4. Decide logging/observability strategy for retrieval (e.g., log selected documents, scores).
5. Update `README.md` with a description of the retrieval pipeline and how to run a “retrieve-only” test (no LLM generation yet).

**Required Accounts / Credentials**
- **OpenAI API key** (to embed the query at retrieval time).
- Optional **Qdrant Cloud** credentials if using Qdrant backend.

**Validation Steps**
- For a known query, inspect retrieved chunks and confirm they contain relevant context.
- Test multiple queries and note failure modes (e.g., missing topics, noisy matches).
- Optionally add a simple CLI test command that prints retrieved contexts for a user-provided question.

---

## Phase 9 – LLM Answer Generation (GPT-4) & Prompting Strategy

**Goal**  
Use OpenAI GPT-4 (or a similar model) to generate answers using retrieved context, with a carefully designed prompt template.

**Files Involved**
- `src/chatbot.py` – main RAG orchestration and conversation logic.
- `src/config.py` – LLM model selection and temperature/top_p parameters.
- New (optional): `src/prompts.py` – store system and user prompt templates.

**Concrete Tasks**
1. Choose the primary chat/completion model (e.g., `gpt-4.1`, `gpt-4o`) and configure it in `config`.
2. Design prompt templates that:
   - Provide system-level instructions (e.g., be helpful, grounded in context, avoid fabrication, etc.).
   - Insert retrieved context into a dedicated section.
   - Include user question clearly.
3. In `chatbot`, define a high-level flow (in design terms):
   - Receive user query.
   - Call retrieval pipeline to get context.
   - Build prompt using templates.
   - Call OpenAI chat API.
   - Return an answer plus optional metadata (e.g., which sources were used).
4. Decide error handling and fallbacks:
   - What happens if no relevant context is found.
   - Handling OpenAI API errors, timeouts, rate limits.
5. Optionally design support for streaming responses to the terminal or a UI.

**Required Accounts / Credentials**
- **OpenAI API key** – required for GPT-4 chat completion.

**Validation Steps**
- With a small knowledge base, ask a few test questions and verify:
   - Answers are grounded in the provided context.
   - The model cites or references source documents (if designed to do so).
- Check behavior when context is missing or irrelevant.
- Inspect prompts and responses to refine your prompt template.

---

## Phase 10 – Conversation Management & CLI Interface

**Goal**  
Implement a user-friendly CLI or interactive loop to chat with the RAG system, maintaining conversational context when appropriate.

**Files Involved**
- `src/chatbot.py` – conversation loop / CLI entry.
- New (optional): `src/cli.py` – if separating CLI concerns from core chatbot logic.
- `README.md` – usage instructions.

**Concrete Tasks**
1. Decide on the interface style:
   - Simple REPL in terminal.
   - Typer/Click-based CLI with commands (e.g., `ingest`, `chat`).
2. Implement conversation state management design:
   - Single-turn Q&A vs. multi-turn chat with conversation history.
   - If multi-turn, decide how much history to send with each request.
3. Add CLI options for:
   - Selecting vector store backend (Chroma/Qdrant).
   - Rebuilding index vs. using existing index.
   - Verbose mode to print retrieved chunks.
4. Document how to run the chatbot from the terminal with example commands.

**Required Accounts / Credentials**
- **OpenAI API key** – for both embeddings and completion.
- Optional **Qdrant Cloud** credentials if that backend is selected.

**Validation Steps**
- Run the CLI and perform an end-to-end chat session.
- Verify that multiple questions in the same session behave as expected.
- Confirm that errors are reported clearly (no cryptic stack traces for common user mistakes).

---

## Phase 11 – Audio Input (Optional – Whisper Integration)

**Goal**  
Optionally enable users to ask questions via audio using OpenAI Whisper, converting speech to text before passing it through the RAG pipeline.

**Files Involved**
- New: `src/audio_input.py` – audio recording and Whisper integration.
- `src/chatbot.py` / `src/cli.py` – integrate audio mode.
- `src/config.py` – Whisper-related configuration (model name, audio settings).

**Concrete Tasks**
1. Decide how audio will be captured:
   - Pre-recorded audio files vs. live microphone recording (e.g., via `pyaudio` or other library).
2. Define functions in `audio_input` for:
   - Accepting audio input and sending it to Whisper.
   - Returning transcribed text.
3. Integrate audio mode into CLI:
   - Add a flag or command to accept audio instead of text input.
4. Consider language settings and transcription accuracy improvements.
5. Update `README.md` with audio usage instructions and any additional dependencies.

**Required Accounts / Credentials**
- **OpenAI API key** – for Whisper API.

**Validation Steps**
- Test transcription on short sample audio clips.
- Confirm that transcribed text flows correctly into the RAG pipeline and yields sensible answers.
- Handle typical errors (invalid audio format, network issues).

---

## Phase 12 – Testing, Evaluation & Quality Assurance

**Goal**  
Add automated tests and basic evaluation routines to ensure correctness, robustness, and maintainability.

**Files Involved**
- New: `tests/` directory with multiple test files:
  - `tests/test_data_loader.py`
  - `tests/test_chunking.py`
  - `tests/test_embeddings.py`
  - `tests/test_retriever.py`
  - `tests/test_chatbot.py`
- `requirements.txt` – ensure test dependencies are listed.
- `README.md` – testing instructions.

**Concrete Tasks**
1. Set up `pytest` as the main test runner.
2. Add unit tests for:
   - Data loading and chunking logic (no external APIs).
   - Vector store integration (ideally with local Chroma only, possibly using a small test database in a temporary directory).
3. Add integration tests for:
   - Retrieval pipeline (may use mock embeddings or minimal real calls).
   - Chatbot end-to-end using mocked OpenAI responses where feasible.
4. Optionally add simple evaluation scripts:
   - A small set of question–answer pairs to check if the RAG system returns reasonably accurate answers.
5. Configure a lightweight test configuration (e.g., using environment variables or a separate test config module).

**Required Accounts / Credentials**
- Optional: OpenAI API key for tests that hit real endpoints (but prefer mocks for repeatability and cost control).

**Validation Steps**
- Run `pytest` and ensure all tests pass:
  ```bash
  pytest
  ```
- Confirm tests are deterministic and fast enough to run frequently.

---

## Phase 13 – Logging, Monitoring & Error Handling

**Goal**  
Improve observability and robustness by adding structured logging, meaningful error messages, and graceful fallbacks.

**Files Involved**
- `src/chatbot.py`
- `src/retriever.py`
- `src/embeddings.py`
- `src/data_loader.py`
- New (optional): `src/logging_config.py` – centralized logging configuration.

**Concrete Tasks**
1. Decide on logging approach:
   - Use the standard `logging` module with a consistent format.
2. Add logs for key steps:
   - Data ingestion, chunking, indexing, retrieval, and LLM calls.
3. Ensure exceptions are caught and either:
   - Re-raised with clearer messages, or
   - Handled gracefully with user-friendly CLI messages.
4. Add configuration flags for log verbosity levels (debug/info/warn).
5. Update `README.md` with guidance on enabling debug logs for troubleshooting.

**Required Accounts / Credentials**
- Same as previous phases; no additional credentials.

**Validation Steps**
- Trigger common error scenarios (e.g., missing `.env`, no documents, empty index, invalid API key) and confirm logs and messages are helpful.
- Ensure logs do not leak sensitive information such as full API keys.

---

## Phase 14 – Documentation & Packaging

**Goal**  
Polish user-facing documentation and optionally prepare the project for distribution or easier reuse.

**Files Involved**
- `README.md` – comprehensive usage and architecture description.
- `PHASE_PLAN.md` – keep in sync with any scope changes.
- (Optional) `pyproject.toml` or `setup.cfg` for packaging if desired.

**Concrete Tasks**
1. Expand `README.md` to include:
   - Project overview and goals.
   - Architecture summary and diagrams (optional, described in text if not using images).
   - Setup instructions (environment, `.env`, data preparation).
   - How to run ingestion, index building, and chat.
   - Troubleshooting section.
2. Ensure `PHASE_PLAN.md` reflects the final implemented state (note any deviations from original plan).
3. Optionally add minimal packaging metadata (if you plan to install it as a package locally).
4. Optionally describe future extensions (web UI, multi-tenant support, analytics, etc.).

**Required Accounts / Credentials**
- None beyond what has already been used.

**Validation Steps**
- Follow the `README.md` instructions from scratch on a clean environment and verify you can:
   - Set up the project.
   - Ingest data and build the index.
   - Chat with the RAG system.
- Fix any documentation gaps discovered during this dry run.

---

## Phase 15 – Optional Enhancements & Experiments

**Goal**  
Implement optional advanced features and experiments beyond the core homework requirements.

**Files Involved**
- Varies depending on selected enhancements; may include new modules.

**Example Enhancements**
1. **Reranking**: Add a reranking step using another model (e.g., cross-encoder or LLM) to improve retrieved context quality.
2. **Summarization**: Add utilities to summarize long context or multi-document answers.
3. **Feedback Loop**: Log user feedback on answers and use it to refine prompts or evaluate retrieval.
4. **Web UI**: Wrap the chatbot in a simple web interface (e.g., FastAPI + React, or Streamlit) while keeping RAG core intact.
5. **Multi-datasource RAG**: Support multiple collections (e.g., course notes + external docs) and allow users to select the data source.

**Required Accounts / Credentials**
- Depends on chosen features (may reuse OpenAI and Qdrant credentials).

**Validation Steps**
- Define small, concrete experimental goals and evaluate each enhancement against them (e.g., accuracy improvements on a test question set).
- Ensure that optional features don’t break the core RAG pipeline when disabled.
