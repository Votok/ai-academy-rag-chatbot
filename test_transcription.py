"""Test script for MP4 transcription debugging.

This script helps diagnose issues with Whisper API transcription by:
1. Extracting audio from MP4
2. Showing file sizes
3. Testing transcription with detailed error reporting
"""

import os
from pathlib import Path
import tempfile
import ffmpeg
from openai import OpenAI
from src.config import OPENAI_API_KEY, WHISPER_MODEL, DATA_DIR


def extract_audio(mp4_path: Path, output_path: Path) -> None:
    """Extract audio from MP4 file."""
    print(f"Extracting audio from: {mp4_path.name}")
    print(f"  Size: {mp4_path.stat().st_size / (1024*1024):.2f} MB")

    (
        ffmpeg
        .input(str(mp4_path))
        .output(
            str(output_path),
            acodec="pcm_s16le",
            ac=1,
            ar="16000",
        )
        .overwrite_output()
        .run(quiet=True, capture_stdout=True, capture_stderr=True)
    )

    audio_size_mb = output_path.stat().st_size / (1024*1024)
    print(f"  Extracted audio size: {audio_size_mb:.2f} MB")
    print(f"  Whisper API limit: 25 MB")

    if audio_size_mb > 25:
        print(f"  ⚠️  WARNING: Audio file exceeds Whisper API 25 MB limit!")
        print(f"  ⚠️  This will cause a 400 error")

    return audio_size_mb


def test_transcription(audio_path: Path) -> str:
    """Test Whisper API transcription."""
    print(f"\nAttempting transcription...")

    client = OpenAI(api_key=OPENAI_API_KEY)

    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model=WHISPER_MODEL,
                file=audio_file,
                response_format="text"
            )

        print(f"✓ Success! Transcribed {len(transcript)} characters")
        print(f"\nFirst 200 chars of transcript:")
        print(f"{transcript[:200]}...")

        return transcript

    except Exception as e:
        print(f"✗ Error: {e}")

        # Provide helpful suggestions
        if "400" in str(e):
            print("\nThis is likely a file size issue.")
            print("Solution: We need to implement audio chunking.")
        elif "401" in str(e):
            print("\nThis is an API key issue.")
            print("Check your OPENAI_API_KEY in .env")

        raise


def main():
    """Main test function."""
    print("=" * 70)
    print("MP4 TRANSCRIPTION TEST")
    print("=" * 70)

    # Find MP4 files
    mp4_files = list(DATA_DIR.glob("*.mp4")) + list(DATA_DIR.glob("*.MP4"))

    if not mp4_files:
        print("No MP4 files found in data directory")
        return

    print(f"Found {len(mp4_files)} MP4 file(s):\n")
    for i, mp4 in enumerate(mp4_files, 1):
        print(f"{i}. {mp4.name} ({mp4.stat().st_size / (1024*1024):.2f} MB)")

    # Test first MP4
    mp4_path = mp4_files[0]
    print(f"\nTesting: {mp4_path.name}")
    print("=" * 70)

    # Create temporary audio file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        temp_audio_path = Path(temp_file.name)

    try:
        # Extract audio
        audio_size_mb = extract_audio(mp4_path, temp_audio_path)

        # Test transcription
        if audio_size_mb > 25:
            print("\n⚠️  Skipping transcription test - file too large")
            print("We need to implement audio chunking to handle this file.")
        else:
            transcript = test_transcription(temp_audio_path)

    finally:
        # Cleanup
        if temp_audio_path.exists():
            os.unlink(temp_audio_path)

    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
