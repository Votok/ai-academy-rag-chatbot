"""
Chatbot module for handling conversations.

This module orchestrates the complete RAG pipeline:
1. Retrieve relevant chunks based on user query
2. Format context and build prompts
3. Call OpenAI GPT to generate answer
4. Return answer with source citations
"""

import logging
from typing import Dict, List, Optional

from openai import OpenAI

from src.config import OPENAI_API_KEY, GPT_MODEL, TOP_K
from src.retriever import retrieve_relevant_chunks
from src.prompts import build_messages

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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
    # Run test when module is executed directly
    _test_generate_answer()
