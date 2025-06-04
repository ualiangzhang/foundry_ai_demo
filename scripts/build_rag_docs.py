#!/usr/bin/env python3
"""
scripts/rag_docs_builder.py

This script consolidates startup descriptions from three sources—
Y Combinator (YC) funded companies, generic startup descriptions,
and Shark Tank pitches—into a single JSON Lines file named
'rag_docs.jsonl'. Each line in the output file contains a JSON object
with the following structure:

    {
        "text": "<cleaned_text>",
        "meta": {
            "source": "<source_tag>",
            "title": "<title>"
        }
    }

Only entries where the cleaned text length is at least 200 characters
are included.

Usage:
    python scripts/rag_docs_builder.py

Ensure that the following directory structure exists under 'data_raw':
    data_raw/yc/             # Contains CSV files of YC funded companies
    data_raw/startup_desc/   # Contains CSV files of generic startup descriptions
    data_raw/shark/          # Contains CSV files of Shark Tank pitch summaries

The script writes the output file to:
    data_processed/rag_docs.jsonl

Setup:
    1. Prepare raw CSV files in the directories listed above.
    2. Install required Python packages, e.g.:
       pip install tqdm

Logging:
    The script logs progress, warnings, and errors to stdout. Check the logs
    for any skipped files or missing directories.

Example:
    mkdir -p data_raw/yc data_raw/startup_desc data_raw/shark
    # Place appropriate CSV files in each folder
    mkdir -p data_processed
    python scripts/rag_docs_builder.py
"""

import csv
import json
import logging
import pathlib
import re
from typing import List, Optional, Tuple

from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Define raw and output directories
RAW: pathlib.Path = pathlib.Path("data_raw")
OUT: pathlib.Path = pathlib.Path("data_processed")
OUT.mkdir(exist_ok=True, parents=True)

try:
    writer = open(OUT / "rag_docs.jsonl", "w", encoding="utf-8")  # type: ignore[name-defined]
except (OSError, IOError) as e:
    logger.error(f"Failed to open output file for writing: {e}")
    raise


def norm(txt: str) -> str:
    """Normalize whitespace in text by collapsing consecutive whitespace into single spaces.

    Args:
        txt (str): Original text string.

    Returns:
        str: A cleaned string with trimmed leading/trailing whitespace and no duplicate spaces.
    """
    return re.sub(r"\s+", " ", txt).strip()


def dump(text: str, source: str, title: str) -> None:
    """Write a record to the JSONL file if the normalized text length is at least 200 characters.

    Args:
        text (str): Raw text describing a startup or pitch.
        source (str): Tag indicating the source of the text (e.g., "yc_company", "generic_desc", "shark_pitch").
        title (str): Title or name associated with the text entry.

    Raises:
        ValueError: If the writer is not initialized or if writing to file fails.
    """
    if writer is None:  # type: ignore[name-defined]
        raise ValueError("Output writer is not initialized.")

    normalized_text: str = norm(text)
    if len(normalized_text) < 200:
        return

    record: dict = {
        "text": normalized_text,
        "meta": {
            "source": source,
            "title": title
        }
    }
    try:
        writer.write(json.dumps(record, ensure_ascii=False) + "\n")
    except (OSError, IOError) as e:
        logger.error(f"Failed to write record to JSONL: {e}")
        raise


def detect_keys(fieldnames: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """Detect which columns in a CSV header correspond to title and description fields.

    Inspects the provided list of field names from a CSV file and attempts to find:
      - A title_key that contains "company", starts with "name", or contains "title".
      - A desc_key that contains "description", equals "desc", or contains "idea" or "summary".

    Args:
        fieldnames (List[str]): List of column names (headers) from a CSV.

    Returns:
        Tuple[Optional[str], Optional[str]]: A tuple (title_key, desc_key). Each element is
            either the matched column name or None if not found.
    """
    title_key: Optional[str] = None
    desc_key: Optional[str] = None

    for k in fieldnames:
        lower_k: str = k.lower()
        # Detect title column
        if title_key is None and ("company" in lower_k or lower_k.startswith("name") or "title" in lower_k):
            title_key = k
        # Detect description column
        if desc_key is None and (
                "description" in lower_k or lower_k == "desc" or "idea" in lower_k or "summary" in lower_k
        ):
            desc_key = k

    return title_key, desc_key


# ─── 1. Process YC funded companies ────────────────────────────────────────────────────
yc_folder: pathlib.Path = RAW / "yc"
processed_any: bool = False

if yc_folder.exists():
    csv_files = list(yc_folder.glob("*.csv"))
    for yc_csv in csv_files:
        try:
            with open(yc_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                title_key, desc_key = detect_keys(fieldnames)

                if title_key and desc_key:
                    logger.info(f"[YC] Using '{yc_csv.name}' with title='{title_key}', desc='{desc_key}'")
                    for row in reader:
                        text = row.get(desc_key, "")
                        title = row.get(title_key, "untitled")
                        dump(text, "yc_company", title)
                    processed_any = True
                else:
                    logger.warning(
                        f"[YC] Skipping '{yc_csv.name}' (no title/desc columns). "
                        f"Found columns: {fieldnames}"
                    )
        except (OSError, IOError) as e:
            logger.error(f"[YC] Error processing file '{yc_csv.name}': {e}")

    if not processed_any:
        logger.warning("⚠️  Warning: No valid CSV in data_raw/yc contained both title & description columns.")
else:
    logger.warning("⚠️  Warning: Folder data_raw/yc not found; skipping YC step.")

# ─── 2. Process generic startup descriptions ───────────────────────────────────────
gen_folder: pathlib.Path = RAW / "startup_desc"
processed_any = False

if gen_folder.exists():
    csv_files = list(gen_folder.glob("*.csv"))
    for gen_csv in csv_files:
        try:
            with open(gen_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []
                title_key, desc_key = detect_keys(fieldnames)

                if title_key and desc_key:
                    logger.info(f"[Generic] Using '{gen_csv.name}' with title='{title_key}', desc='{desc_key}'")
                    for row in reader:
                        text = row.get(desc_key, "")
                        title = row.get(title_key, "untitled")
                        dump(text, "generic_desc", title)
                    processed_any = True
                else:
                    logger.warning(
                        f"[Generic] Skipping '{gen_csv.name}' (no title/desc columns). "
                        f"Found columns: {fieldnames}"
                    )
        except (OSError, IOError) as e:
            logger.error(f"[Generic] Error processing file '{gen_csv.name}': {e}")

    if not processed_any:
        logger.warning(
            "⚠️  Warning: No valid CSV in data_raw/startup_desc contained both title & description columns."
        )
else:
    logger.warning("⚠️  Warning: Folder data_raw/startup_desc not found; skipping generic startup step.")

# ─── 3. Process Shark-Tank pitches ─────────────────────────────────────────────────
shark_folder: pathlib.Path = RAW / "shark"
processed_any = False

if shark_folder.exists():
    csv_files = list(shark_folder.glob("*.csv"))
    for st_csv in csv_files:
        try:
            with open(st_csv, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames or []

                # Prefer "title" and "description" columns if present
                title_key: Optional[str] = next((k for k in fieldnames if "title" in k.lower()), None)
                desc_key: Optional[str] = next((k for k in fieldnames if "description" in k.lower()), None)
                # Fallback keys: idea, ask/amount, equity
                idea_key: Optional[str] = next((k for k in fieldnames if "idea" in k.lower()), None)
                amt_key: Optional[str] = next((k for k in fieldnames if "ask" in k.lower() or "amount" in k.lower()),
                                              None)
                eq_key: Optional[str] = next(
                    (k for k in fieldnames if "exchange" in k.lower() or "equity" in k.lower()), None
                )

                if title_key and desc_key:
                    logger.info(f"[Shark] Using '{st_csv.name}' with description='{desc_key}', title='{title_key}'")
                    for row in reader:
                        text: str = row.get(desc_key, "")
                        ask: str = row.get(amt_key, "").strip() if amt_key else ""
                        equity: str = row.get(eq_key, "").strip() if eq_key else ""
                        # Append ASK and equity information if present
                        if ask or equity:
                            if not text.endswith((".", "?", "!")):
                                text += " "
                            text += "ASK: "
                            if ask:
                                text += f"{ask}"
                            if ask and equity:
                                text += " for "
                            if equity:
                                text += f"{equity}%"
                        title: str = row.get(title_key, "untitled")
                        dump(text, "shark_pitch", title)
                    processed_any = True

                # Fallback: use idea + title if no description present
                elif idea_key and title_key:
                    logger.info(f"[Shark] Using fallback '{st_csv.name}' with idea='{idea_key}', title='{title_key}'")
                    for row in reader:
                        idea: str = row.get(idea_key, "")
                        ask = row.get(amt_key, "").strip() if amt_key else ""
                        equity = row.get(eq_key, "").strip() if eq_key else ""
                        text = idea
                        if ask or equity:
                            if not text.endswith((".", "?", "!")):
                                text += " "
                            text += "ASK: "
                            if ask:
                                text += f"{ask}"
                            if ask and equity:
                                text += " for "
                            if equity:
                                text += f"{equity}%"
                        title = row.get(title_key, "untitled")
                        dump(text, "shark_pitch", title)
                    processed_any = True

                else:
                    logger.warning(
                        f"[Shark] Skipping '{st_csv.name}' (no usable columns). Found columns: {fieldnames}"
                    )
        except (OSError, IOError) as e:
            logger.error(f"[Shark] Error processing file '{st_csv.name}': {e}")

    if not processed_any:
        logger.warning("⚠️  Warning: No valid CSV in data_raw/shark contained required columns.")
else:
    logger.warning("⚠️  Warning: Folder data_raw/shark not found; skipping Shark-Tank step.")

# Close the output file handle
try:
    writer.close()  # type: ignore[name-defined]
    logger.info(f"✅  Saved {OUT / 'rag_docs.jsonl'}")
except (OSError, IOError) as e:
    logger.error(f"Failed to close the writer properly: {e}")
