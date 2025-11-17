"""Data loader module for ingesting documents.

This module handles loading and processing of PDF and MP4 files:
- PDF: Text extraction with page metadata
- MP4: Audio extraction → Whisper transcription → text
- All text is chunked into semantically meaningful pieces for embedding
"""

import os
import tempfile
from pathlib import Path
from typing import List, Optional
import warnings

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from tqdm import tqdm
import ffmpeg
from openai import OpenAI

from src.config import (
    DATA_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
    WHISPER_MODEL,
)


# Initialize OpenAI client for Whisper API
_openai_client = None


def _get_openai_client() -> OpenAI:
    """Get or create OpenAI client (lazy initialization)."""
    global _openai_client
    if _openai_client is None:
        _openai_client = OpenAI(
            api_key=OPENAI_API_KEY,
            timeout=300.0,  # 5 minute timeout for large audio files
        )
    return _openai_client


def _get_transcript_cache_path(mp4_path: Path, data_dir: Path) -> Path:
    """Get the cache file path for a transcript.

    Args:
        mp4_path: Path to the MP4 file
        data_dir: Data directory containing the MP4

    Returns:
        Path to the transcript cache file
    """
    transcripts_dir = data_dir / "transcripts"
    # Use stem to get filename without extension, add .txt
    return transcripts_dir / f"{mp4_path.stem}.txt"


def _load_cached_transcript(cache_path: Path, mp4_path: Path) -> Optional[str]:
    """Load transcript from cache if it exists and is newer than MP4.

    Args:
        cache_path: Path to cached transcript file
        mp4_path: Path to original MP4 file

    Returns:
        Cached transcript text if valid, None otherwise
    """
    # Check if cache exists
    if not cache_path.exists():
        return None

    # Check if cache is newer than MP4
    cache_mtime = cache_path.stat().st_mtime
    mp4_mtime = mp4_path.stat().st_mtime

    if cache_mtime < mp4_mtime:
        # Cache is outdated
        return None

    # Load and return cached transcript
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        warnings.warn(f"Error reading cached transcript {cache_path}: {e}")
        return None


def _save_transcript_cache(cache_path: Path, transcript: str) -> None:
    """Save transcript to cache file.

    Args:
        cache_path: Path to save transcript to
        transcript: Transcript text to save
    """
    try:
        # Create transcripts directory if it doesn't exist
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Write transcript to file
        with open(cache_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    except Exception as e:
        warnings.warn(f"Error saving transcript cache {cache_path}: {e}")


def _get_partial_transcript_dir(data_dir: Path) -> Path:
    """Get the directory for partial transcript chunks.

    Args:
        data_dir: Data directory

    Returns:
        Path to partial transcripts directory
    """
    return data_dir / "transcripts" / ".partial"


def _save_partial_transcript(mp4_path: Path, data_dir: Path, chunk_index: int, transcript: str) -> None:
    """Save a partial transcript chunk.

    Args:
        mp4_path: Path to the MP4 file
        data_dir: Data directory
        chunk_index: Index of the chunk (0-based)
        transcript: Transcript text for this chunk
    """
    try:
        partial_dir = _get_partial_transcript_dir(data_dir)
        partial_dir.mkdir(parents=True, exist_ok=True)

        partial_path = partial_dir / f"{mp4_path.stem}_chunk_{chunk_index}.txt"
        with open(partial_path, "w", encoding="utf-8") as f:
            f.write(transcript)
    except Exception as e:
        warnings.warn(f"Error saving partial transcript: {e}")


def _load_partial_transcripts(mp4_path: Path, data_dir: Path, num_chunks: int) -> Optional[List[str]]:
    """Load partial transcripts if all chunks exist.

    Args:
        mp4_path: Path to the MP4 file
        data_dir: Data directory
        num_chunks: Expected number of chunks

    Returns:
        List of transcript chunks if all exist, None otherwise
    """
    try:
        partial_dir = _get_partial_transcript_dir(data_dir)
        if not partial_dir.exists():
            return None

        transcripts = []
        for i in range(num_chunks):
            partial_path = partial_dir / f"{mp4_path.stem}_chunk_{i}.txt"
            if not partial_path.exists():
                return None

            with open(partial_path, "r", encoding="utf-8") as f:
                transcripts.append(f.read())

        return transcripts
    except Exception:
        return None


def _cleanup_partial_transcripts(mp4_path: Path, data_dir: Path) -> None:
    """Clean up partial transcript files for a given MP4.

    Args:
        mp4_path: Path to the MP4 file
        data_dir: Data directory
    """
    try:
        partial_dir = _get_partial_transcript_dir(data_dir)
        if not partial_dir.exists():
            return

        # Remove all partial files for this MP4
        for partial_file in partial_dir.glob(f"{mp4_path.stem}_chunk_*.txt"):
            partial_file.unlink()
    except Exception as e:
        warnings.warn(f"Error cleaning up partial transcripts: {e}")


# ============================================================================
# Part A: PDF Processing
# ============================================================================


def _load_pdfs(data_dir: Path) -> List[Document]:
    """Load all PDF files from the data directory.

    Args:
        data_dir: Directory containing PDF files

    Returns:
        List of Document objects with text and metadata (source, page)
    """
    pdf_files = list(data_dir.glob("*.pdf"))
    if not pdf_files:
        warnings.warn(f"No PDF files found in {data_dir}")
        return []

    all_documents = []

    print(f"\nLoading {len(pdf_files)} PDF file(s)...")
    for pdf_path in tqdm(pdf_files, desc="Loading PDFs"):
        try:
            # Use LangChain's PyPDFLoader
            loader = PyPDFLoader(str(pdf_path))
            documents = loader.load()

            # Add source filename to metadata
            for doc in documents:
                doc.metadata["source"] = pdf_path.name
                doc.metadata["source_type"] = "pdf"

            all_documents.extend(documents)
            print(f"  ✓ {pdf_path.name}: {len(documents)} pages")

        except Exception as e:
            warnings.warn(f"Error loading PDF {pdf_path.name}: {e}")
            continue

    return all_documents


# ============================================================================
# Part B: MP4 Audio Transcription
# ============================================================================


def _extract_audio_from_mp4(mp4_path: Path, output_path: Path) -> None:
    """Extract audio from MP4 file using ffmpeg.

    Args:
        mp4_path: Path to input MP4 file
        output_path: Path to output audio file (WAV format)

    Raises:
        Exception: If ffmpeg extraction fails
    """
    try:
        # Extract audio using ffmpeg-python
        # Convert to mono, 16kHz (Whisper's preferred format)
        (
            ffmpeg
            .input(str(mp4_path))
            .output(
                str(output_path),
                acodec="pcm_s16le",  # PCM 16-bit
                ac=1,  # Mono
                ar="16000",  # 16kHz sample rate
            )
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        raise Exception(f"ffmpeg error: {e.stderr.decode()}")


def _split_audio_file(audio_path: Path, chunk_duration_seconds: int = 300) -> List[Path]:
    """Split large audio file into smaller chunks.

    Args:
        audio_path: Path to audio file to split
        chunk_duration_seconds: Duration of each chunk in seconds (default: 5 minutes)

    Returns:
        List of paths to audio chunk files
    """
    # Check file size first
    file_size_mb = audio_path.stat().st_size / (1024 * 1024)

    # If file is under 24 MB (leaving 1 MB safety margin), no need to split
    if file_size_mb < 24:
        return [audio_path]

    print(f"    Audio file is {file_size_mb:.1f} MB, splitting into chunks...")

    # Create temporary directory for chunks
    temp_dir = Path(tempfile.mkdtemp(prefix="audio_chunks_"))
    chunk_paths = []

    try:
        # Get audio duration
        probe = ffmpeg.probe(str(audio_path))
        duration = float(probe['format']['duration'])

        # Calculate number of chunks needed
        num_chunks = int(duration / chunk_duration_seconds) + 1

        print(f"    Splitting {duration:.0f}s audio into {num_chunks} chunks of ~{chunk_duration_seconds}s each...")

        # Split audio into chunks
        for i in range(num_chunks):
            start_time = i * chunk_duration_seconds
            chunk_path = temp_dir / f"chunk_{i:03d}.wav"

            (
                ffmpeg
                .input(str(audio_path), ss=start_time, t=chunk_duration_seconds)
                .output(
                    str(chunk_path),
                    acodec="pcm_s16le",
                    ac=1,
                    ar="16000",
                )
                .overwrite_output()
                .run(quiet=True, capture_stdout=True, capture_stderr=True)
            )

            if chunk_path.exists() and chunk_path.stat().st_size > 0:
                chunk_paths.append(chunk_path)

        print(f"    Created {len(chunk_paths)} audio chunks")
        return chunk_paths

    except Exception as e:
        # Clean up on error
        for chunk in chunk_paths:
            if chunk.exists():
                os.unlink(chunk)
        if temp_dir.exists():
            os.rmdir(temp_dir)
        raise Exception(f"Error splitting audio: {e}")


def _transcribe_audio_with_whisper(
    audio_path: Path,
    source_filename: str = None,
    mp4_path: Path = None,
    data_dir: Path = None
) -> str:
    """Transcribe audio file using OpenAI Whisper API.

    Handles large files by automatically splitting them into chunks.
    Saves progress for each chunk to allow resuming after failures.

    Args:
        audio_path: Path to audio file (WAV, MP3, etc.)
        source_filename: Original filename for error messages (optional)
        mp4_path: Path to original MP4 file for progress saving (optional)
        data_dir: Data directory for progress saving (optional)

    Returns:
        Transcribed text

    Raises:
        Exception: If Whisper API call fails
    """
    client = _get_openai_client()

    # Split audio if needed
    chunk_paths = _split_audio_file(audio_path)
    should_cleanup_chunks = len(chunk_paths) > 1

    # Check for existing partial transcripts if we have MP4 info
    if mp4_path and data_dir and len(chunk_paths) > 1:
        partial_transcripts = _load_partial_transcripts(mp4_path, data_dir, len(chunk_paths))
        if partial_transcripts:
            print(f"    Found {len(partial_transcripts)} cached chunk(s), using saved progress")
            full_transcript = " ".join(partial_transcripts)
            _cleanup_partial_transcripts(mp4_path, data_dir)
            return full_transcript

    try:
        transcripts = []

        for i, chunk_path in enumerate(chunk_paths, 1):
            if len(chunk_paths) > 1:
                print(f"    Transcribing chunk {i}/{len(chunk_paths)}...")

            try:
                with open(chunk_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model=WHISPER_MODEL,
                        file=audio_file,
                        response_format="text"
                    )
                    transcripts.append(transcript)

                # Save partial transcript if we have MP4 info
                if mp4_path and data_dir and len(chunk_paths) > 1:
                    _save_partial_transcript(mp4_path, data_dir, i - 1, transcript)

            except Exception as chunk_error:
                # Provide detailed error message about which chunk failed
                source_info = f" for {source_filename}" if source_filename else ""
                raise Exception(
                    f"Whisper API error on chunk {i}/{len(chunk_paths)}{source_info}: {chunk_error}\n"
                    f"Suggestion: This may be a transient API issue (502 errors are common with OpenAI). "
                    f"Try running the script again - chunks 1-{i-1} are cached and will be reused."
                )

        # Combine transcripts with space separation
        full_transcript = " ".join(transcripts)

        # Clean up partial transcripts on success
        if mp4_path and data_dir:
            _cleanup_partial_transcripts(mp4_path, data_dir)

        return full_transcript

    except Exception as e:
        # Re-raise if already formatted, otherwise wrap
        if "Whisper API error on chunk" in str(e):
            raise
        source_info = f" for {source_filename}" if source_filename else ""
        raise Exception(f"Whisper API error{source_info}: {e}")

    finally:
        # Clean up chunk files if we created them
        if should_cleanup_chunks:
            for chunk_path in chunk_paths:
                if chunk_path.exists():
                    try:
                        os.unlink(chunk_path)
                    except Exception:
                        pass

            # Try to remove the temp directory
            try:
                if chunk_paths and chunk_paths[0].parent.exists():
                    os.rmdir(chunk_paths[0].parent)
            except Exception:
                pass


def _load_mp4s(data_dir: Path) -> List[Document]:
    """Load all MP4 files from the data directory and transcribe audio.

    Args:
        data_dir: Directory containing MP4 files

    Returns:
        List of Document objects with transcribed text and metadata
    """
    mp4_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.MP4"))
    if not mp4_files:
        warnings.warn(f"No MP4 files found in {data_dir}")
        return []

    all_documents = []

    print(f"\nProcessing {len(mp4_files)} MP4 file(s)...")
    for mp4_path in tqdm(mp4_files, desc="Transcribing MP4s"):
        temp_audio_path = None
        transcript = None

        try:
            # Check for cached transcript first
            cache_path = _get_transcript_cache_path(mp4_path, data_dir)
            cached_transcript = _load_cached_transcript(cache_path, mp4_path)

            if cached_transcript:
                # Use cached transcript
                print(f"\n  ✓ Using cached transcript for {mp4_path.name}")
                transcript = cached_transcript
            else:
                # Need to transcribe
                # Create temporary WAV file
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as temp_file:
                    temp_audio_path = Path(temp_file.name)

                # Step 1: Extract audio from MP4
                print(f"\n  Extracting audio from {mp4_path.name}...")
                _extract_audio_from_mp4(mp4_path, temp_audio_path)

                # Step 2: Transcribe with Whisper
                print(f"  Transcribing {mp4_path.name} with Whisper...")
                transcript = _transcribe_audio_with_whisper(
                    temp_audio_path,
                    source_filename=mp4_path.name,
                    mp4_path=mp4_path,
                    data_dir=data_dir
                )

                # Step 3: Save transcript to cache
                _save_transcript_cache(cache_path, transcript)
                print(f"  ✓ {mp4_path.name}: transcribed {len(transcript)} characters")

            # Create Document with metadata
            doc = Document(
                page_content=transcript,
                metadata={
                    "source": mp4_path.name,
                    "source_type": "mp4",
                    "transcription_length": len(transcript),
                },
            )
            all_documents.append(doc)

        except Exception as e:
            warnings.warn(f"Error processing MP4 {mp4_path.name}: {e}")
            continue

        finally:
            # Clean up temporary audio file
            if temp_audio_path and temp_audio_path.exists():
                try:
                    os.unlink(temp_audio_path)
                except Exception:
                    pass

    return all_documents


# ============================================================================
# Part C: Text Chunking
# ============================================================================


def _chunk_documents(documents: List[Document]) -> List[Document]:
    """Split documents into smaller chunks for embedding.

    Args:
        documents: List of Document objects to chunk

    Returns:
        List of chunked Document objects with preserved metadata
    """
    if not documents:
        return []

    # Use LangChain's RecursiveCharacterTextSplitter
    # Splits at natural boundaries: paragraphs, sentences, words
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    print(f"\nChunking {len(documents)} documents...")
    chunked_documents = []

    # Track global chunk counter per source file
    source_counters = {}

    for doc in tqdm(documents, desc="Chunking documents"):
        try:
            # Split document into chunks
            chunks = text_splitter.split_documents([doc])

            # Get source filename from the document
            source = doc.metadata.get('source', 'unknown')

            # Initialize counter for this source if not exists
            if source not in source_counters:
                source_counters[source] = 0

            # Add chunk index to metadata with global counter
            for i, chunk in enumerate(chunks):
                # Use global counter for chunk_id (unique across all pages)
                chunk.metadata["chunk_id"] = f"{source}_{source_counters[source]}"
                chunk.metadata["chunk_index"] = source_counters[source]
                chunk.metadata["chunk_index_in_page"] = i  # Keep local index for debugging

                # Increment global counter for this source
                source_counters[source] += 1

            chunked_documents.extend(chunks)

        except Exception as e:
            warnings.warn(f"Error chunking document from {doc.metadata.get('source')}: {e}")
            continue

    print(f"  ✓ Created {len(chunked_documents)} chunks from {len(documents)} documents")
    return chunked_documents


# ============================================================================
# Part D: Pipeline Orchestration
# ============================================================================


def load_and_chunk_documents(data_dir: Optional[Path] = None) -> List[Document]:
    """Main pipeline: Load PDFs and MP4s, transcribe audio, and chunk all text.

    This is the primary entry point for data ingestion. It:
    1. Loads all PDF files and extracts text
    2. Loads all MP4 files, extracts audio, and transcribes with Whisper
    3. Chunks all text into semantically meaningful pieces
    4. Returns list of Document objects ready for embedding

    Args:
        data_dir: Directory containing source files (default: config.DATA_DIR)

    Returns:
        List of chunked Document objects with metadata

    Raises:
        ValueError: If data directory doesn't exist or is empty
    """
    # Use configured data directory if not provided
    if data_dir is None:
        data_dir = DATA_DIR

    # Validate data directory
    if not data_dir.exists():
        raise ValueError(
            f"Data directory does not exist: {data_dir}\n"
            f"Please create it and add PDF and MP4 files."
        )

    # Check for supported files
    pdf_files = list(data_dir.glob("*.pdf"))
    mp4_files = list(data_dir.glob("*.mp4")) + list(data_dir.glob("*.MP4"))

    if not pdf_files and not mp4_files:
        raise ValueError(
            f"No PDF or MP4 files found in {data_dir}\n"
            f"Please add source documents to process."
        )

    print("=" * 60)
    print("RAG CHATBOT - DATA INGESTION PIPELINE")
    print("=" * 60)
    print(f"Data directory: {data_dir.absolute()}")
    print(f"Found: {len(pdf_files)} PDF(s), {len(mp4_files)} MP4(s)")
    print(f"Chunk size: {CHUNK_SIZE}, Overlap: {CHUNK_OVERLAP}")
    print("=" * 60)

    all_documents = []

    # Load PDFs
    try:
        pdf_docs = _load_pdfs(data_dir)
        all_documents.extend(pdf_docs)
    except Exception as e:
        warnings.warn(f"Error loading PDFs: {e}")

    # Load and transcribe MP4s
    try:
        mp4_docs = _load_mp4s(data_dir)
        all_documents.extend(mp4_docs)
    except Exception as e:
        warnings.warn(f"Error processing MP4s: {e}")

    # Check if we loaded any documents
    if not all_documents:
        raise ValueError(
            "No documents were successfully loaded. "
            "Check warnings above for errors."
        )

    # Chunk all documents
    chunked_docs = _chunk_documents(all_documents)

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"Total documents loaded: {len(all_documents)}")
    print(f"Total chunks created: {len(chunked_docs)}")
    print(f"Average chunk size: {sum(len(d.page_content) for d in chunked_docs) / len(chunked_docs):.0f} chars")
    print("=" * 60)

    return chunked_docs


# ============================================================================
# Testing / Demo
# ============================================================================


if __name__ == "__main__":
    """Test the data loading pipeline."""
    try:
        # Load and chunk all documents
        documents = load_and_chunk_documents()

        # Display sample chunks
        print("\n" + "=" * 60)
        print("SAMPLE CHUNKS")
        print("=" * 60)

        for i, doc in enumerate(documents[:3], 1):
            print(f"\nChunk {i}:")
            print(f"  Source: {doc.metadata.get('source')}")
            print(f"  Type: {doc.metadata.get('source_type')}")
            print(f"  Chunk ID: {doc.metadata.get('chunk_id')}")
            print(f"  Length: {len(doc.page_content)} chars")
            print(f"  Preview: {doc.page_content[:200]}...")

        print("\n✓ Data loading pipeline test completed successfully!")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
