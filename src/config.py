"""Configuration management for the RAG chatbot.

This module centralizes all configuration parameters including API keys,
model names, paths, and processing parameters. It loads environment variables
from a .env file and provides sensible defaults.

All other modules should import configuration values from this module rather
than reading environment variables directly.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def _get_env_var(key: str, default: Optional[str] = None, required: bool = False) -> str:
    """Get an environment variable with optional default and validation.

    Args:
        key: Environment variable name
        default: Default value if not set
        required: If True, raises error when variable is not set

    Returns:
        The environment variable value or default

    Raises:
        ValueError: If required=True and variable is not set
    """
    value = os.getenv(key, default)
    if required and not value:
        raise ValueError(
            f"Missing required environment variable: {key}\n"
            f"Please set {key} in your .env file or environment.\n"
            f"See .env.example for configuration template."
        )
    return value


# ============================================================================
# OpenAI API Configuration
# ============================================================================

OPENAI_API_KEY: str = _get_env_var("OPENAI_API_KEY", required=True)
"""OpenAI API key for accessing GPT, Whisper, and Embeddings APIs.
Required. Get yours at https://platform.openai.com/api-keys
"""

GPT_MODEL: str = _get_env_var("GPT_MODEL", default="gpt-4")
"""GPT model to use for answer generation.
Examples: 'gpt-4', 'gpt-4-turbo', 'gpt-3.5-turbo'
"""

EMBEDDING_MODEL: str = _get_env_var("EMBEDDING_MODEL", default="text-embedding-3-small")
"""OpenAI embedding model for vector generation.
Examples: 'text-embedding-3-small', 'text-embedding-3-large', 'text-embedding-ada-002'
"""

WHISPER_MODEL: str = _get_env_var("WHISPER_MODEL", default="whisper-1")
"""Whisper model for audio transcription.
Default: 'whisper-1' (OpenAI API version)
"""

# ============================================================================
# Directory Paths
# ============================================================================

DATA_DIR: Path = Path(_get_env_var("DATA_DIR", default="./data"))
"""Directory containing source documents (PDFs and MP4 files)."""

CHROMA_DB_DIR: Path = Path(_get_env_var("CHROMA_DB_DIR", default="./embeddings"))
"""Directory for ChromaDB vector database storage."""

# ============================================================================
# Text Processing Parameters
# ============================================================================

CHUNK_SIZE: int = int(_get_env_var("CHUNK_SIZE", default="1000"))
"""Maximum size of text chunks for embedding (in characters).
Larger chunks provide more context but may be less precise for retrieval.
"""

CHUNK_OVERLAP: int = int(_get_env_var("CHUNK_OVERLAP", default="200"))
"""Number of characters to overlap between consecutive chunks.
Helps maintain context continuity across chunk boundaries.
"""

# ============================================================================
# Retrieval Parameters
# ============================================================================

TOP_K: int = int(_get_env_var("TOP_K", default="5"))
"""Number of most relevant chunks to retrieve for each query.
Higher values provide more context but increase token usage.
"""

# ============================================================================
# Configuration Validation
# ============================================================================

def validate_config() -> None:
    """Validate configuration and check for common issues.

    Raises:
        ValueError: If configuration is invalid
    """
    # Check that directories exist or can be created
    if not DATA_DIR.exists():
        raise ValueError(
            f"Data directory does not exist: {DATA_DIR}\n"
            f"Please create it and add your PDF and MP4 files."
        )

    # Ensure Chroma DB directory parent exists
    CHROMA_DB_DIR.parent.mkdir(parents=True, exist_ok=True)

    # Validate numeric parameters
    if CHUNK_SIZE <= 0:
        raise ValueError(f"CHUNK_SIZE must be positive, got: {CHUNK_SIZE}")

    if CHUNK_OVERLAP < 0:
        raise ValueError(f"CHUNK_OVERLAP must be non-negative, got: {CHUNK_OVERLAP}")

    if CHUNK_OVERLAP >= CHUNK_SIZE:
        raise ValueError(
            f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be less than CHUNK_SIZE ({CHUNK_SIZE})"
        )

    if TOP_K <= 0:
        raise ValueError(f"TOP_K must be positive, got: {TOP_K}")


def print_config() -> None:
    """Print current configuration (for debugging)."""
    print("=== RAG Chatbot Configuration ===")
    print(f"OpenAI API Key: {'*' * 20}{OPENAI_API_KEY[-4:] if len(OPENAI_API_KEY) > 4 else '****'}")
    print(f"GPT Model: {GPT_MODEL}")
    print(f"Embedding Model: {EMBEDDING_MODEL}")
    print(f"Whisper Model: {WHISPER_MODEL}")
    print(f"Data Directory: {DATA_DIR.absolute()}")
    print(f"ChromaDB Directory: {CHROMA_DB_DIR.absolute()}")
    print(f"Chunk Size: {CHUNK_SIZE}")
    print(f"Chunk Overlap: {CHUNK_OVERLAP}")
    print(f"Top K Results: {TOP_K}")
    print("=" * 40)


# Validate configuration on module import
try:
    validate_config()
except ValueError as e:
    # Don't fail import, but warn user
    import warnings
    warnings.warn(f"Configuration validation warning: {e}")
