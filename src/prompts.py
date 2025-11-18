"""
Prompt templates for RAG chatbot.

This module contains all prompt templates used for generating responses.
Prompts are designed to ground LLM responses in the retrieved context
and encourage source citations.
"""

from typing import List
from langchain_core.documents import Document


# System prompt that defines the assistant's behavior
SYSTEM_PROMPT = """You are a helpful course assistant that answers questions based on course materials (PDFs and video transcripts).

Your guidelines:
1. **Ground your answers in the provided context** - Only use information from the context chunks provided below
2. **Cite your sources** - Always mention which source(s) you're drawing from (e.g., "According to lecture_notes.pdf..." or "As mentioned in the video transcript...")
3. **Be honest about limitations** - If the context doesn't contain enough information to answer the question, say so clearly
4. **Be concise but complete** - Provide thorough answers without unnecessary elaboration
5. **Maintain academic tone** - Be professional and educational in your responses

If you don't find relevant information in the context, respond with:
"I don't have enough information in the course materials to answer that question. The available context doesn't cover this topic."
"""


def format_context_chunks(documents: List[Document]) -> str:
    """
    Format retrieved document chunks into a readable context string.

    Args:
        documents: List of Document objects with page_content and metadata

    Returns:
        Formatted string with numbered context chunks and source attribution
    """
    if not documents:
        return "No relevant context found."

    context_parts = []
    for i, doc in enumerate(documents, 1):
        metadata = doc.metadata
        source = metadata.get("source", "Unknown")
        source_type = metadata.get("source_type", "unknown")

        # Build source attribution
        if source_type == "pdf":
            page = metadata.get("page", "?")
            attribution = f"[Source: {source}, page {page}]"
        elif source_type == "mp4":
            attribution = f"[Source: {source} (video transcript)]"
        else:
            attribution = f"[Source: {source}]"

        # Add chunk with attribution
        context_parts.append(
            f"--- Context Chunk {i} {attribution} ---\n{doc.page_content}\n"
        )

    return "\n".join(context_parts)


def build_user_prompt(query: str, context_documents: List[Document]) -> str:
    """
    Build the complete user prompt with context and question.

    Args:
        query: User's question
        context_documents: Retrieved relevant document chunks

    Returns:
        Formatted prompt string combining context and question
    """
    context_str = format_context_chunks(context_documents)

    prompt = f"""Based on the following context from course materials, please answer the question.

{context_str}

---

**Question:** {query}

**Answer:**"""

    return prompt


def build_messages(query: str, context_documents: List[Document]) -> List[dict]:
    """
    Build the complete message list for OpenAI Chat API.

    Args:
        query: User's question
        context_documents: Retrieved relevant document chunks

    Returns:
        List of message dicts in OpenAI format [{"role": "...", "content": "..."}]
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(query, context_documents)},
    ]
