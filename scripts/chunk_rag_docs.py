#!/usr/bin/env python3
"""
rag_docs_chunker.py

This script reads a JSONL file of startup documents (data_processed/rag_docs.jsonl),
splits any document "text" longer than ~512 words into overlapping chunks of
up to 512 words (whitespace-based splitting), and writes the results to
data_processed/rag_docs_chunked.jsonl.

Each chunk retains the original metadata but contains only a subset of the text.
Chunks overlap by 50 words to preserve context across splits.

Usage:
    python3 rag_docs_chunker.py

Ensure:
    - The input file 'data_processed/rag_docs.jsonl' exists.
    - The output directory 'data_processed' is writable.

Logging:
    Progress and errors are logged to stdout.

Example:
    mkdir -p data_processed
    # Place rag_docs.jsonl in data_processed/
    python3 rag_docs_chunker.py
"""

import json
import logging
import pathlib
from typing import Any, Dict, Iterator, List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger: logging.Logger = logging.getLogger(__name__)

# Paths
INPUT_FILE: pathlib.Path = pathlib.Path("data_processed/rag_docs.jsonl")
OUTPUT_FILE: pathlib.Path = pathlib.Path("data_processed/rag_docs_chunked.jsonl")

# Chunk parameters
MAX_WORDS: int = 512  # Maximum words per chunk
OVERLAP_WORDS: int = 50  # Overlap between consecutive chunks


def chunk_text(text: str, max_words: int, overlap: int) -> Iterator[str]:
    """Split the input text into overlapping chunks based on whitespace.

    This function breaks a long text into chunks of up to `max_words` words,
    with `overlap` words overlapping between consecutive chunks. If the total
    word count of `text` is less than or equal to `max_words`, the original
    text is returned as a single chunk.

    Args:
        text: The original text to be chunked.
        max_words: Maximum number of words in each chunk.
        overlap: Number of words to overlap between consecutive chunks.

    Yields:
        Each chunk of the text as a string of words.

    Example:
        >>> list(chunk_text("one two three", max_words=2, overlap=1))
        ["one two", "two three"]
    """
    words: List[str] = text.split()
    total_words: int = len(words)

    # If text is short enough, yield it as a single chunk
    if total_words <= max_words:
        yield text
        return

    # Step size accounts for overlap
    step: int = max_words - overlap
    for start in range(0, total_words, step):
        end: int = start + max_words
        chunk_words: List[str] = words[start:end]
        yield " ".join(chunk_words)


def process_file(input_path: pathlib.Path, output_path: pathlib.Path) -> int:
    """Read the input JSONL, chunk long texts, and write to output JSONL.

    Each line in the input JSONL must be a valid JSON object with keys:
        - "text": str
        - "meta": Dict[str, Any]

    For each record, if the "text" field contains more than `MAX_WORDS` words,
    it is split into multiple overlapping chunks. Each chunk is written as a new
    line in the output JSONL, preserving the original "meta".

    Args:
        input_path: Path to the input JSONL file.
        output_path: Path to the output JSONL file.

    Returns:
        The total number of lines (chunks) written to the output file.

    Raises:
        FileNotFoundError: If the input file does not exist.
        IOError: If reading or writing fails.
    """
    if not input_path.exists():
        msg: str = f"Input file '{input_path}' not found."
        logger.error(msg)
        raise FileNotFoundError(msg)

    lines_written: int = 0
    try:
        with input_path.open("r", encoding="utf-8") as infile, \
                output_path.open("w", encoding="utf-8") as outfile:
            for line_number, line in enumerate(infile, start=1):
                try:
                    obj: Dict[str, Any] = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON on line {line_number}: {e}")
                    continue

                text: str = obj.get("text", "")
                meta: Dict[str, Any] = obj.get("meta", {})

                for chunk in chunk_text(text, MAX_WORDS, OVERLAP_WORDS):
                    new_obj: Dict[str, Any] = {"text": chunk, "meta": meta}
                    try:
                        outfile.write(json.dumps(new_obj, ensure_ascii=False) + "\n")
                        lines_written += 1
                    except (OSError, IOError) as e:
                        logger.error(f"Failed to write chunk for line {line_number}: {e}")
                        raise
        logger.info(f"✅  Saved {output_path} with {lines_written} total chunks.")
        return lines_written
    except (OSError, IOError) as e:
        logger.error(f"Error processing file '{input_path}': {e}")
        raise


def main() -> None:
    """Main entry point for chunking rag_docs.jsonl."""
    try:
        total: int = process_file(INPUT_FILE, OUTPUT_FILE)
        logger.info(f"Processing complete: {total} lines written.")
    except Exception as e:
        logger.error(f"Unhandled exception in main: {e}")


if __name__ == "__main__":
    main()
