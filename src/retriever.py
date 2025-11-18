"""Retriever module for finding relevant documents.

This module handles ChromaDB vector database operations including:
- Initializing and managing the persistent vector store
- Indexing document chunks with embeddings
- Performing similarity search for query retrieval
"""

import warnings
from typing import List, Optional, Dict, Any
from pathlib import Path

import chromadb
from chromadb.config import Settings
from langchain_core.documents import Document
from tqdm import tqdm

from src.config import CHROMA_DB_DIR, TOP_K
from src.embeddings import embed_texts, embed_query


# Global ChromaDB client and collection (lazy initialization)
_chroma_client = None
_collection = None

COLLECTION_NAME = "course_materials"


def _get_chroma_client() -> chromadb.Client:
    """Get or create ChromaDB persistent client (lazy initialization).

    Returns:
        chromadb.Client: Configured ChromaDB client with persistent storage
    """
    global _chroma_client
    if _chroma_client is None:
        # Ensure ChromaDB directory exists
        CHROMA_DB_DIR.mkdir(parents=True, exist_ok=True)

        # Create persistent client
        _chroma_client = chromadb.PersistentClient(
            path=str(CHROMA_DB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )
    return _chroma_client


def _get_collection() -> chromadb.Collection:
    """Get or create the ChromaDB collection.

    Returns:
        chromadb.Collection: The course materials collection
    """
    global _collection
    if _collection is None:
        client = _get_chroma_client()

        # Get or create collection
        # Note: ChromaDB handles embedding function separately
        # We'll provide embeddings directly when adding documents
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"description": "RAG chatbot course materials"}
        )
    return _collection


def get_collection_stats() -> Dict[str, Any]:
    """Get statistics about the indexed documents.

    Returns:
        Dictionary with collection statistics including:
        - count: Number of documents in the collection
        - collection_name: Name of the collection
        - storage_path: Path to ChromaDB storage

    Example:
        >>> stats = get_collection_stats()
        >>> print(f"Indexed {stats['count']} documents")
    """
    collection = _get_collection()
    count = collection.count()

    return {
        "count": count,
        "collection_name": COLLECTION_NAME,
        "storage_path": str(CHROMA_DB_DIR.absolute()),
    }


def clear_index() -> None:
    """Delete all documents from the collection (for rebuilding index).

    Warning:
        This operation is irreversible. All indexed documents will be removed.
    """
    global _collection, _chroma_client

    client = _get_chroma_client()

    try:
        # Delete the collection
        client.delete_collection(name=COLLECTION_NAME)
        print(f"✓ Deleted collection '{COLLECTION_NAME}'")

        # Reset the collection reference
        _collection = None

        # Recreate empty collection
        _get_collection()
        print(f"✓ Created new empty collection '{COLLECTION_NAME}'")

    except Exception as e:
        warnings.warn(f"Error clearing index: {e}")


def index_documents(
    documents: List[Document],
    batch_size: int = 100,
    show_progress: bool = True,
) -> None:
    """Index documents into ChromaDB with embeddings.

    This function:
    1. Extracts text and metadata from documents
    2. Generates embeddings for all document chunks
    3. Stores in ChromaDB with unique IDs
    4. Is idempotent (skips already indexed documents)

    Args:
        documents: List of Document objects to index
        batch_size: Number of documents to process per batch (default: 100)
        show_progress: Whether to show progress bar (default: True)

    Raises:
        ValueError: If documents list is empty
        Exception: If indexing fails

    Example:
        >>> from src.data_loader import load_and_chunk_documents
        >>> docs = load_and_chunk_documents()
        >>> index_documents(docs)
        Processing 150 chunks...
        Generating embeddings: 100%|████████| 150/150
        ✓ Indexed 150 documents
    """
    if not documents:
        raise ValueError("Cannot index empty document list")

    collection = _get_collection()

    # Get existing document IDs to avoid duplicates
    existing_ids = set()
    try:
        # Get all existing IDs (ChromaDB can handle this efficiently)
        existing_data = collection.get(limit=1000000, include=[])
        if existing_data and existing_data["ids"]:
            existing_ids = set(existing_data["ids"])
            print(f"Found {len(existing_ids)} existing documents in index")
    except Exception as e:
        warnings.warn(f"Could not fetch existing IDs: {e}")

    # Filter out documents that are already indexed
    new_documents = []
    for doc in documents:
        doc_id = doc.metadata.get("chunk_id", f"doc_{len(new_documents)}")
        if doc_id not in existing_ids:
            new_documents.append(doc)

    if not new_documents:
        print("✓ All documents already indexed, nothing to do")
        return

    print(f"\nIndexing {len(new_documents)} new documents (skipping {len(documents) - len(new_documents)} existing)...")

    # Process in batches
    total_batches = (len(new_documents) + batch_size - 1) // batch_size

    progress_bar = tqdm(
        total=len(new_documents),
        desc="Indexing documents",
        disable=not show_progress,
        unit="doc",
    )

    indexed_count = 0

    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, len(new_documents))
        batch = new_documents[start_idx:end_idx]

        try:
            # Extract data from documents
            ids = []
            texts = []
            metadatas = []

            for doc in batch:
                # Generate unique ID from metadata or use index
                doc_id = doc.metadata.get("chunk_id", f"doc_{start_idx + len(ids)}")
                ids.append(doc_id)
                texts.append(doc.page_content)

                # ChromaDB metadata must be flat dict with simple types
                # Convert all metadata values to strings to be safe
                metadata = {
                    key: str(value) for key, value in doc.metadata.items()
                }
                metadatas.append(metadata)

            # Generate embeddings for batch
            embeddings = embed_texts(texts)

            # Add to ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
            )

            indexed_count += len(batch)
            progress_bar.update(len(batch))

        except Exception as e:
            warnings.warn(f"Error indexing batch {batch_idx + 1}/{total_batches}: {e}")
            continue

    progress_bar.close()

    print(f"✓ Successfully indexed {indexed_count}/{len(new_documents)} new documents")
    print(f"✓ Total documents in index: {collection.count()}")


def retrieve_relevant_chunks(
    query: str,
    top_k: Optional[int] = None,
    min_score: Optional[float] = None,
) -> List[Document]:
    """Retrieve the most relevant document chunks for a query.

    Performs semantic similarity search in ChromaDB to find chunks that
    best match the user's query.

    Args:
        query: User's question or search query
        top_k: Number of results to return (default: from config)
        min_score: Minimum similarity score threshold (optional)

    Returns:
        List of Document objects with relevance scores in metadata

    Raises:
        ValueError: If query is empty or index is empty
        Exception: If retrieval fails

    Example:
        >>> results = retrieve_relevant_chunks("What is RAG?", top_k=3)
        >>> for doc in results:
        ...     print(f"Source: {doc.metadata['source']}")
        ...     print(f"Score: {doc.metadata['score']}")
        ...     print(f"Text: {doc.page_content[:100]}...")
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if top_k is None:
        top_k = TOP_K

    collection = _get_collection()

    # Check if collection is empty
    if collection.count() == 0:
        raise ValueError(
            "Index is empty. Please run index building first:\n"
            "  python -m src.build_index"
        )

    try:
        # Generate query embedding
        query_embedding = embed_query(query)

        # Perform similarity search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

        # Convert results to Document objects
        documents = []

        # ChromaDB returns results in nested lists
        if results["ids"] and results["ids"][0]:
            for idx in range(len(results["ids"][0])):
                doc_text = results["documents"][0][idx]
                doc_metadata = results["metadatas"][0][idx]
                distance = results["distances"][0][idx]

                # Convert distance to similarity score (lower distance = higher similarity)
                # For cosine distance, similarity = 1 - distance
                # ChromaDB uses squared L2 distance by default
                similarity_score = 1.0 / (1.0 + distance)

                # Apply minimum score filter if specified
                if min_score is not None and similarity_score < min_score:
                    continue

                # Add score to metadata
                doc_metadata["score"] = similarity_score
                doc_metadata["distance"] = distance

                doc = Document(
                    page_content=doc_text,
                    metadata=doc_metadata,
                )
                documents.append(doc)

        return documents

    except Exception as e:
        raise Exception(f"Error retrieving documents: {e}")


def format_retrieved_chunks(documents: List[Document], include_scores: bool = False) -> str:
    """Format retrieved chunks for display or LLM input.

    Args:
        documents: List of retrieved Document objects
        include_scores: Whether to include relevance scores (default: False)

    Returns:
        Formatted string with all chunks and their metadata

    Example:
        >>> docs = retrieve_relevant_chunks("What is RAG?")
        >>> formatted = format_retrieved_chunks(docs)
        >>> print(formatted)
    """
    if not documents:
        return "No relevant documents found."

    formatted_parts = []

    for idx, doc in enumerate(documents, 1):
        source = doc.metadata.get("source", "unknown")
        source_type = doc.metadata.get("source_type", "unknown")

        # Format source info based on type
        if source_type == "pdf":
            page = doc.metadata.get("page", "?")
            source_info = f"{source}, page {page}"
        elif source_type == "mp4":
            source_info = f"{source} (transcript)"
        else:
            source_info = source

        chunk_header = f"[Chunk {idx} from {source_info}]"

        if include_scores:
            score = doc.metadata.get("score", 0.0)
            chunk_header += f" (relevance: {score:.3f})"

        formatted_parts.append(f"{chunk_header}\n{doc.page_content}")

    return "\n\n---\n\n".join(formatted_parts)


# ============================================================================
# Testing / Demo
# ============================================================================

if __name__ == "__main__":
    """Test the retriever module with sample data."""
    print("=" * 60)
    print("RETRIEVER MODULE TEST")
    print("=" * 60)

    # Create some sample documents
    sample_docs = [
        Document(
            page_content="RAG stands for Retrieval-Augmented Generation. It combines retrieval with LLM generation.",
            metadata={"source": "test.pdf", "source_type": "pdf", "page": "1", "chunk_id": "test_0"},
        ),
        Document(
            page_content="Machine learning models learn patterns from data without being explicitly programmed.",
            metadata={"source": "test.pdf", "source_type": "pdf", "page": "2", "chunk_id": "test_1"},
        ),
        Document(
            page_content="Vector databases store embeddings and enable semantic similarity search.",
            metadata={"source": "test.pdf", "source_type": "pdf", "page": "3", "chunk_id": "test_2"},
        ),
    ]

    # Test indexing
    print("\n1. Testing document indexing...")
    try:
        index_documents(sample_docs, show_progress=False)
        stats = get_collection_stats()
        print(f"   ✓ Collection stats: {stats}")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test retrieval
    print("\n2. Testing document retrieval...")
    try:
        query = "What is RAG?"
        results = retrieve_relevant_chunks(query, top_k=2)
        print(f"   Query: '{query}'")
        print(f"   Retrieved {len(results)} documents")
        for doc in results:
            print(f"   - Source: {doc.metadata['source']}, Score: {doc.metadata['score']:.3f}")
            print(f"     Text: {doc.page_content[:80]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    # Test formatting
    print("\n3. Testing format_retrieved_chunks...")
    try:
        formatted = format_retrieved_chunks(results, include_scores=True)
        print("   Formatted output:")
        print("   " + formatted.replace("\n", "\n   ")[:200] + "...")
    except Exception as e:
        print(f"   ✗ Error: {e}")

    print("\n" + "=" * 60)
    print("✓ Retriever tests completed!")
    print("=" * 60)
    print("\nNote: Run 'clear_index()' to clean up test data")
