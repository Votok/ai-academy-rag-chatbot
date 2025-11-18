"""
Chatbot module for handling conversations.

This module orchestrates the complete RAG pipeline:
1. Retrieve relevant chunks based on user query
2. Format context and build prompts
3. Call OpenAI GPT to generate answer
4. Return answer with source citations
"""

import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import typer
from openai import OpenAI

from src.config import OPENAI_API_KEY, GPT_MODEL, TOP_K, DATA_DIR, CHROMA_DB_DIR
from src.retriever import retrieve_relevant_chunks
from src.prompts import build_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create typer app
app = typer.Typer(
    help="RAG Chatbot - Answer questions using course materials",
    add_completion=False,
)


class RAGChatbot:
    """
    RAG Chatbot that answers questions using retrieved context.

    This class handles the complete RAG workflow:
    - Query understanding and retrieval
    - Context formatting
    - LLM-based answer generation
    - Source attribution
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        top_k: Optional[int] = None,
    ):
        """
        Initialize the RAG chatbot.

        Args:
            api_key: OpenAI API key (defaults to config.OPENAI_API_KEY)
            model: GPT model name (defaults to config.GPT_MODEL)
            top_k: Number of chunks to retrieve (defaults to config.TOP_K)
        """
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model or GPT_MODEL
        self.top_k = top_k or TOP_K

        # Initialize OpenAI client
        self.client = OpenAI(api_key=self.api_key)

        logger.info(f"Initialized RAG Chatbot with model={self.model}, top_k={self.top_k}")

    def generate_answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        min_score: Optional[float] = None,
        temperature: float = 0.7,
    ) -> Dict:
        """
        Generate an answer to a user query using RAG.

        This is the main entry point for the RAG pipeline. It:
        1. Retrieves relevant document chunks
        2. Formats them into a prompt
        3. Calls GPT to generate an answer
        4. Returns structured response with sources

        Args:
            query: User's question
            top_k: Number of chunks to retrieve (overrides instance default)
            min_score: Minimum similarity score for retrieved chunks
            temperature: LLM temperature (0.0-1.0, higher = more creative)

        Returns:
            Dict containing:
                - answer: Generated response text
                - sources: List of unique source filenames used
                - retrieved_chunks: List of Document objects used as context
                - query: The original query
                - model: Model used for generation

        Raises:
            ValueError: If query is empty or retrieval fails
            Exception: If OpenAI API call fails
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")

        logger.info(f"Processing query: '{query}'")

        # Step 1: Retrieve relevant chunks
        try:
            k = top_k or self.top_k
            chunks = retrieve_relevant_chunks(query, top_k=k, min_score=min_score)

            if not chunks:
                logger.warning("No relevant chunks found for query")
                return {
                    "answer": "I couldn't find any relevant information in the course materials to answer your question. Please try rephrasing or asking about a different topic covered in the materials.",
                    "sources": [],
                    "retrieved_chunks": [],
                    "query": query,
                    "model": self.model,
                }

            logger.info(f"Retrieved {len(chunks)} relevant chunks")

        except Exception as e:
            logger.error(f"Retrieval error: {e}")
            raise ValueError(f"Failed to retrieve relevant documents: {e}")

        # Step 2: Build prompt messages
        messages = build_messages(query, chunks)

        # Step 3: Call OpenAI GPT
        try:
            logger.info(f"Calling OpenAI API with model={self.model}")

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=1000,  # Limit response length
            )

            answer = response.choices[0].message.content.strip()
            logger.info("Successfully generated answer")

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise Exception(f"Failed to generate answer: {e}")

        # Step 4: Extract unique sources
        sources = list(set(chunk.metadata.get("source", "Unknown") for chunk in chunks))
        sources.sort()  # Alphabetical order for consistency

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_chunks": chunks,
            "query": query,
            "model": self.model,
        }


# Convenience function for simple usage
def generate_answer(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
    temperature: float = 0.7,
) -> Dict:
    """
    Generate an answer to a query using RAG (convenience function).

    This is a simple wrapper around RAGChatbot.generate_answer() for
    single-shot usage without maintaining a chatbot instance.

    Args:
        query: User's question
        top_k: Number of chunks to retrieve
        min_score: Minimum similarity score
        temperature: LLM temperature (0.0-1.0)

    Returns:
        Dict with answer, sources, retrieved_chunks, query, and model

    Example:
        >>> result = generate_answer("What is RAG?")
        >>> print(result["answer"])
        >>> print("Sources:", result["sources"])
    """
    chatbot = RAGChatbot()
    return chatbot.generate_answer(query, top_k, min_score, temperature)


# ============================================================================
# CLI Helper Functions
# ============================================================================


def _format_verbose_output(result: Dict, elapsed_time: float) -> str:
    """Format query result for verbose console output.

    Args:
        result: Result dictionary from generate_answer()
        elapsed_time: Time taken for the query in seconds

    Returns:
        Formatted string for display
    """
    lines = []
    lines.append("=" * 80)
    lines.append("QUERY RESULT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Question: {result['query']}")
    lines.append("")
    lines.append("Answer:")
    lines.append("-" * 80)
    lines.append(result['answer'])
    lines.append("-" * 80)
    lines.append("")

    # Sources
    if result['sources']:
        lines.append(f"Sources ({len(result['sources'])} file(s)):")
        for source in result['sources']:
            lines.append(f"  - {source}")
    else:
        lines.append("Sources: None (no relevant context found)")
    lines.append("")

    # Retrieved chunks
    lines.append(f"Retrieved Chunks ({len(result['retrieved_chunks'])} chunk(s)):")
    lines.append("")
    for i, chunk in enumerate(result['retrieved_chunks'], 1):
        source = chunk.metadata.get('source', 'unknown')
        score = chunk.metadata.get('score', 0.0)
        distance = chunk.metadata.get('distance', 0.0)

        # Format chunk info based on source type
        source_type = chunk.metadata.get('source_type', '')
        if source_type == 'pdf':
            page = chunk.metadata.get('page', '?')
            chunk_info = f"{source} (page {page})"
        elif source_type == 'mp4':
            chunk_info = f"{source} (transcript)"
        else:
            chunk_info = source

        lines.append(f"  Chunk {i}: {chunk_info}")
        lines.append(f"  Relevance Score: {score:.4f} (distance: {distance:.4f})")
        lines.append(f"  Text Preview: {chunk.page_content[:200]}...")
        lines.append("")

    # Metadata
    lines.append("=" * 80)
    lines.append("METADATA")
    lines.append("=" * 80)
    lines.append(f"Model: {result['model']}")
    lines.append(f"Time Elapsed: {elapsed_time:.2f}s")
    lines.append("=" * 80)

    return "\n".join(lines)


def _log_query_to_file(result: Dict, elapsed_time: float, log_file: Path = Path("queries.log")) -> None:
    """Log query and result to a file.

    Args:
        result: Result dictionary from generate_answer()
        elapsed_time: Time taken for the query in seconds
        log_file: Path to log file (default: queries.log)
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with open(log_file, "a", encoding="utf-8") as f:
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"TIMESTAMP: {timestamp}\n")
            f.write("=" * 80 + "\n")
            f.write(f"QUESTION: {result['query']}\n")
            f.write("\n")
            f.write("ANSWER:\n")
            f.write("-" * 80 + "\n")
            f.write(result['answer'] + "\n")
            f.write("-" * 80 + "\n")
            f.write("\n")

            # Sources
            f.write(f"SOURCES ({len(result['sources'])} file(s)):\n")
            for source in result['sources']:
                f.write(f"  - {source}\n")
            f.write("\n")

            # Retrieved chunks
            f.write(f"RETRIEVED CHUNKS ({len(result['retrieved_chunks'])} chunk(s)):\n")
            f.write("\n")
            for i, chunk in enumerate(result['retrieved_chunks'], 1):
                source = chunk.metadata.get('source', 'unknown')
                score = chunk.metadata.get('score', 0.0)
                source_type = chunk.metadata.get('source_type', '')

                if source_type == 'pdf':
                    page = chunk.metadata.get('page', '?')
                    chunk_info = f"{source} (page {page})"
                elif source_type == 'mp4':
                    chunk_info = f"{source} (transcript)"
                else:
                    chunk_info = source

                f.write(f"  [{i}] {chunk_info} - Score: {score:.4f}\n")

            # Metadata
            f.write(f"MODEL: {result['model']}\n")
            f.write(f"TIME ELAPSED: {elapsed_time:.2f}s\n")
            f.write("=" * 80 + "\n")

        logger.info(f"Query logged to {log_file}")
    except Exception as e:
        logger.error(f"Failed to log query to file: {e}")


# ============================================================================
# CLI Commands
# ============================================================================


@app.command("build-index")
def build_index_command():
    """
    Build or rebuild the vector database index from source documents.

    This command will:
    1. Load all PDF and MP4 files from the data directory
    2. Transcribe audio from MP4 files using Whisper
    3. Chunk all text into smaller pieces
    4. Generate embeddings and store in ChromaDB

    Run this command before querying the chatbot.
    """
    from src.data_loader import load_and_chunk_documents
    from src.retriever import clear_index, index_documents, get_collection_stats

    print("\n" + "=" * 80)
    print("RAG CHATBOT - BUILD INDEX")
    print("=" * 80)
    print(f"Data Directory: {DATA_DIR.absolute()}")
    print(f"ChromaDB Directory: {CHROMA_DB_DIR.absolute()}")
    print("=" * 80)

    # Validate OpenAI API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        typer.echo("\n✗ Error: OpenAI API key not configured", err=True)
        typer.echo("Please set OPENAI_API_KEY in your .env file", err=True)
        raise typer.Exit(1)

    try:
        # Step 1: Clear existing index
        print("\nStep 1: Clearing existing index...")
        clear_index()

        # Step 2: Load and chunk documents
        print("\nStep 2: Loading and chunking documents...")
        documents = load_and_chunk_documents()

        if not documents:
            typer.echo("\n✗ Error: No documents loaded", err=True)
            raise typer.Exit(1)

        # Step 3: Index documents
        print("\nStep 3: Generating embeddings and indexing...")
        index_documents(documents)

        # Step 4: Show stats
        print("\nStep 4: Verifying index...")
        stats = get_collection_stats()

        print("\n" + "=" * 80)
        print("✓ INDEX BUILD COMPLETE")
        print("=" * 80)
        print(f"Collection: {stats['collection_name']}")
        print(f"Total Documents: {stats['count']}")
        print(f"Storage Path: {stats['storage_path']}")
        print("=" * 80)
        print("\nYou can now query the chatbot:")
        print('  python -m src.chatbot "What is RAG?"')
        print("=" * 80)

    except KeyboardInterrupt:
        typer.echo("\n\n✗ Build cancelled by user", err=True)
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"\n✗ Error building index: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


@app.command()
def query(
    question: str = typer.Argument(..., help="The question to ask the chatbot"),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Number of chunks to retrieve"),
    log_file: Path = typer.Option(Path("queries.log"), "--log-file", "-l", help="Path to log file"),
):
    """
    Ask a question to the RAG chatbot (default command).

    The chatbot will retrieve relevant context from indexed documents
    and generate an answer using GPT.

    Examples:
        python -m src.chatbot "What is RAG?"
        python -m src.chatbot "How does retrieval work?" --top-k 5
    """
    # Validate OpenAI API key
    if not OPENAI_API_KEY or OPENAI_API_KEY == "your-openai-api-key-here":
        typer.echo("\n✗ Error: OpenAI API key not configured", err=True)
        typer.echo("Please set OPENAI_API_KEY in your .env file", err=True)
        raise typer.Exit(1)

    # Check if index exists
    if not CHROMA_DB_DIR.exists():
        typer.echo("\n✗ Error: Index not found", err=True)
        typer.echo("Please build the index first:", err=True)
        typer.echo("  python -m src.chatbot build-index", err=True)
        raise typer.Exit(1)

    try:
        # Start timer
        start_time = time.time()

        # Generate answer
        result = generate_answer(query=question, top_k=top_k)

        # Calculate elapsed time
        elapsed_time = time.time() - start_time

        # Display verbose output to console
        output = _format_verbose_output(result, elapsed_time)
        print("\n" + output)

        # Log to file
        _log_query_to_file(result, elapsed_time, log_file)

        print(f"\n✓ Query logged to {log_file.absolute()}\n")

    except ValueError as e:
        # Handle empty index or validation errors
        if "Index is empty" in str(e):
            typer.echo("\n✗ Error: Index is empty", err=True)
            typer.echo("Please build the index first:", err=True)
            typer.echo("  python -m src.chatbot build-index", err=True)
        else:
            typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1)
    except KeyboardInterrupt:
        typer.echo("\n\n✗ Query cancelled by user", err=True)
        raise typer.Exit(130)
    except Exception as e:
        typer.echo(f"\n✗ Error processing query: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


# Test function for module validation
def _test_generate_answer():
    """Test the generate_answer function with a sample query."""
    print("=" * 60)
    print("Testing RAG Chatbot - generate_answer()")
    print("=" * 60)

    # Test query
    test_query = "What are embeddings?" # What is RAG? What are the problems with traditional RAG? What are embeddings?

    print(f"\nQuery: {test_query}\n")

    try:
        result = generate_answer(test_query, top_k=3)

        print("Answer:")
        print("-" * 60)
        print(result["answer"])
        print("-" * 60)

        print(f"\nSources used: {', '.join(result['sources'])}")
        print(f"Model: {result['model']}")
        print(f"Retrieved chunks: {len(result['retrieved_chunks'])}")

        print("\n✓ Test successful!")

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run CLI app
    app()
