"""Embeddings module for vector operations.

This module handles generation of vector embeddings using OpenAI's embedding models.
It provides functions for embedding both individual queries and batches of documents.
"""

from typing import List
import warnings

from langchain_openai import OpenAIEmbeddings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from openai import APIError, RateLimitError, APITimeoutError

from src.config import OPENAI_API_KEY, EMBEDDING_MODEL


# Global embeddings client (lazy initialization)
_embeddings_client = None


def _get_embeddings_client() -> OpenAIEmbeddings:
    """Get or create OpenAI embeddings client (lazy initialization).

    Returns:
        OpenAIEmbeddings: Configured LangChain embeddings client
    """
    global _embeddings_client
    if _embeddings_client is None:
        _embeddings_client = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            # Optimize for batch processing
            chunk_size=1000,  # Max texts per API call
            max_retries=3,
        )
    return _embeddings_client


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def embed_texts(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for a list of texts.

    This function uses OpenAI's embedding API to convert texts into vector
    representations. It includes automatic retry logic for rate limits and
    transient API errors.

    Args:
        texts: List of text strings to embed

    Returns:
        List of embedding vectors (each vector is a list of floats)

    Raises:
        ValueError: If texts list is empty
        Exception: If OpenAI API call fails after retries

    Example:
        >>> texts = ["Hello world", "RAG is useful"]
        >>> embeddings = embed_texts(texts)
        >>> len(embeddings)
        2
        >>> len(embeddings[0])  # Dimension depends on model
        1536
    """
    if not texts:
        raise ValueError("Cannot embed empty list of texts")

    # Filter out empty strings
    non_empty_texts = [t for t in texts if t.strip()]
    if len(non_empty_texts) < len(texts):
        warnings.warn(
            f"Filtered out {len(texts) - len(non_empty_texts)} empty text(s)"
        )

    if not non_empty_texts:
        raise ValueError("All texts are empty after filtering")

    try:
        client = _get_embeddings_client()
        # LangChain's embed_documents handles batching automatically
        embeddings = client.embed_documents(non_empty_texts)
        return embeddings

    except (RateLimitError, APITimeoutError) as e:
        # Let tenacity retry handle these
        raise
    except Exception as e:
        raise Exception(f"Error generating embeddings: {e}")


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def embed_query(query: str) -> List[float]:
    """Generate embedding for a single query string.

    Optimized for embedding user queries. Uses the same model as document
    embeddings to ensure compatibility.

    Args:
        query: Query string to embed

    Returns:
        Embedding vector as a list of floats

    Raises:
        ValueError: If query is empty
        Exception: If OpenAI API call fails after retries

    Example:
        >>> query = "What is RAG?"
        >>> embedding = embed_query(query)
        >>> len(embedding)
        1536
    """
    if not query or not query.strip():
        raise ValueError("Cannot embed empty query")

    try:
        client = _get_embeddings_client()
        # LangChain's embed_query is optimized for single queries
        embedding = client.embed_query(query)
        return embedding

    except (RateLimitError, APITimeoutError) as e:
        # Let tenacity retry handle these
        raise
    except Exception as e:
        raise Exception(f"Error generating query embedding: {e}")


def get_embedding_dimension() -> int:
    """Get the dimension of embeddings produced by the current model.

    Returns:
        int: Embedding vector dimension

    Note:
        - text-embedding-3-small: 1536 dimensions
        - text-embedding-3-large: 3072 dimensions
        - text-embedding-ada-002: 1536 dimensions
    """
    # Test with a dummy query to get dimension
    test_embedding = embed_query("test")
    return len(test_embedding)


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    """Test the embeddings module."""
    print("=" * 60)
    print("EMBEDDINGS MODULE TEST")
    print("=" * 60)

    # Test single query embedding
    print("\n1. Testing single query embedding...")
    query = "What is machine learning?"
    query_embedding = embed_query(query)
    print(f"   Query: '{query}'")
    print(f"   Embedding dimension: {len(query_embedding)}")
    print(f"   First 5 values: {query_embedding[:5]}")

    # Test batch text embedding
    print("\n2. Testing batch text embedding...")
    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "RAG combines retrieval with generation for better answers.",
    ]
    text_embeddings = embed_texts(texts)
    print(f"   Embedded {len(text_embeddings)} texts")
    print(f"   Embedding dimension: {len(text_embeddings[0])}")

    # Test get embedding dimension
    print("\n3. Testing get_embedding_dimension...")
    dim = get_embedding_dimension()
    print(f"   Model dimension: {dim}")

    print("\n" + "=" * 60)
    print("✓ All embeddings tests passed!")
    print("=" * 60)
