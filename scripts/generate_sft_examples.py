#!/usr/bin/env python3
"""
scripts/generate_sft_examples.py
────────────────────────────────
Generate a ShareGPT-style SFT dataset with the following logical flow:
  1. Generate a 10–50-word entrepreneurial idea ("summary") for each theme.
  2. Use that summary to search for a numeric market snippet via DuckDuckGo.
  3. From the snippet and summary, generate a ~100-word market background ("context").
  4. Based on both context and summary, generate four VC recommendations.

When an example is rejected by the critic, log the specific reason for rejection.

Process:
1. Idea Generation – For each theme, call GPT to produce a 10–50-word startup idea.
2. DuckDuckGo retrieval – For each summary (idea), fetch exactly one numeric snippet
   (≤50 words) by querying “<summary> <market_keyword>”. If no snippet, skip theme.
3. Context Generation – Provide snippet and summary to GPT to produce ~100 words of market background.
4. Recommendation Generation – Provide context and summary to GPT to produce four numbered VC recommendations.
5. Critic QA – Validate format and lengths; when rejecting, record the explicit reason.

Outputs:
    data_processed/
     ├─ sft_train.jsonl      ← ShareGPT conversations (ready for LLaMA-Factory)
     ├─ dataset_info.json    ← Dataset metadata descriptor
     └─ sft_generation.log   ← Detailed generation/rejection logs

Requirements:
    pip install openai>=1.23 duckduckgo-search>=8.0 tqdm python-dotenv
    export OPENAI_API_KEY="sk-..."

Author: you@example.com
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import openai  # Used for catching OpenAI exception classes
from dotenv import load_dotenv
from duckduckgo_search import DDGS
from duckduckgo_search.exceptions import DuckDuckGoSearchException
from openai import OpenAI
from openai.types.chat import ChatCompletion
from tqdm import tqdm

# ─────────────────────────── Environment & Logging ───────────────────────────
load_dotenv()
API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
if not API_KEY:
    sys.stderr.write("❌  OPENAI_API_KEY not set – aborting.\n")
    sys.exit(1)

# Initialize OpenAI client with API key and default timeout
client: OpenAI = OpenAI(api_key=API_KEY, timeout=60)

# Define output directory and files for processed data and logs
OUT_DIR: Path = Path("data_processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SFT_JSONL: Path = OUT_DIR / "sft_train.jsonl"
DATASET_INFO: Path = OUT_DIR / "dataset_info.json"
LOG_FILE: Path = OUT_DIR / "sft_generation.log"

# Configure logging to both console and log file
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
    ],
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SFT-Generator")

# ─────────────────────────── Constants & Prompts ─────────────────────────────
GEN_MODEL: str = "gpt-4o"
CRITIC_MODEL: str = "gpt-4o"

# List of themes representing different market domains
THEMES: List[str] = [
    "bioprinting", "agri-drones", "web3 identity", "quantum SaaS", "food-tech",
    "elder-care robotics", "climate fintech", "gene therapy", "VR fitness",
    "circular fashion", "digital twins", "satellite ESG imaging",
]

# Keywords to append when searching for market data
MARKET_KEYWORDS: List[str] = [
    "market size 2025", "Total Addressable Market", "Compound Annual Growth Rate",
    "Annual Growth Rate", "Market Revenue", "user growth statistics", "market forcast"
]

# System prompt for generating a startup idea ("summary") given a theme
SUMMARY_GEN_SYS: str = (
    "You are a startup ideation expert.\n"
    "Given the theme provided by the user, generate a single-line JSON with key:\n"
    "  • \"summary\": a concise entrepreneurial idea in 10–50 words.\n"
    "Rules:\n"
    "  – Do NOT include markdown fences.\n"
    "  – The summary must be between 10 and 50 words.\n"
)

# System prompt for generating market context from snippet and summary
CONTEXT_GEN_SYS: str = (
    "You are a veteran VC partner and market analyst.\n"
    "You receive:\n"
    "  • \"summary\": a startup idea.\n"
    "  • \"snippet\": a small text containing a numeric market fact.\n"
    "Produce a single-line JSON with key:\n"
    "  • \"context\": approximately 100 words of concise market background,\n"
    "    weaving in the snippet’s numeric fact and showing how it relates to the startup idea.\n"
    "Rules:\n"
    "  – Do NOT include markdown fences.\n"
    "  – The context should be about 95–105 words.\n"
)

# System prompt for generating four VC recommendations from context and summary
RECS_GEN_SYS: str = (
    "You are a seasoned VC partner.\n"
    "You receive:\n"
    "  • \"summary\": startup idea (10–50 words).\n"
    "  • \"context\": market background text (~100 words).\n"
    "Produce a single-line JSON with key:\n"
    "  • \"recommendations\": four bullets, each ≤50 words,\n"
    "    covering Market, Product, Business Model, and Team. Each bullet should reference\n"
    "    either the context or summary.\n"
    "Rules:\n"
    "  – Do NOT include markdown fences.\n"
    "  – Exactly four numbered bullets.\n"
    "  – Each bullet must be ≤50 words.\n"
    "  – Do not use question marks.\n"
)

# System prompt used by the critic for QA validation
CRITIC_SYS: str = (
    "You are a strict QA reviewer.\n"
    "Given one JSON example with keys summary, context, recommendations, verify:\n"
    "recommendations: covering Market, Product, Business Model, and Team, no questions.\n"
    "Return JSON: {\"pass\": bool, \"reason\": str, \"fix\": {…}|null}."
)

# Wrapper system prompt for final ShareGPT-style message
SYSTEM_PROMPT: str = (
    "You are a veteran VC partner. Using the provided summary and context,\n"
    "produce exactly FOUR recommendations covering Market, Product,\n"
    "Business Model, and Team. Each bullet ≤50 words, no questions.\n"
    "If context or summary is missing, reply exactly INSUFFICIENT_CONTEXT."
)

# Prefixes for user messages in ShareGPT conversation format
USER_PREFIX: str = "### Startup summary\n"
CTX_PREFIX: str = "\n\n### Market context\n"

# DuckDuckGo backends and limits
DDG_BACKENDS: List[str] = ["api", "html", "lite"]
MAX_SNIPPET_WORDS: int = 50  # Maximum words for a market snippet
MAX_DDG_RETRIES: int = 6  # Number of DuckDuckGo retry attempts
MAX_GEN_ATTEMPTS: int = 3  # Number of GPT generation attempts per stage


# ─────────────────────────── DuckDuckGo Retrieval ────────────────────────────
def duck_top1_snippet(query: str) -> Optional[str]:
    """
    Retrieve exactly one numeric snippet (≤ MAX_SNIPPET_WORDS words) for the given query.
    Tries multiple backends with exponential back-off in case of rate limits.

    Args:
        query (str): Search query string, e.g., "my startup idea TAM market size 2025".

    Returns:
        Optional[str]: A snippet string (≤50 words, containing at least one digit),
        or None if no valid snippet is found after all retries.
    """
    for attempt in range(MAX_DDG_RETRIES):
        backend: str = DDG_BACKENDS[attempt % len(DDG_BACKENDS)]
        try:
            with DDGS() as ddgs:
                for res in ddgs.text(query, backend=backend, max_results=1):
                    body: str = res.get("body", "")
                    words: List[str] = re.findall(r"\S+", body)[:MAX_SNIPPET_WORDS]
                    snippet: str = " ".join(words)
                    if any(ch.isdigit() for ch in snippet):
                        return snippet
            logger.debug("No numeric snippet for '%s' via %s; retrying.", query, backend)
        except DuckDuckGoSearchException as exc:
            wait: float = 1.5 * (2 ** attempt)
            logger.warning(
                "DuckDuckGo %s backend rate-limit: %s; sleeping %.1fs", backend, exc, wait
            )
            time.sleep(wait + random.random() * 0.5)
    return None


# ─────────────────────────── OpenAI Chat Helper ─────────────────────────────
def chat(
        sys_prompt: str,
        user_prompt: str,
        *,
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 400,
) -> str:
    """
    Call OpenAI ChatCompletion with exponential back-off on rate limits
    and transient network errors.

    Args:
        sys_prompt (str): Content for the system role prompt.
        user_prompt (str): Content for the user role prompt.
        model (str): Model name, e.g., "gpt-4o-mini".
        temperature (float): Sampling temperature for generation.
        max_tokens (int): Maximum tokens to generate.

    Returns:
        str: The assistant-generated text, or an empty string if all retries fail.
    """
    for attempt in range(6):
        try:
            resp: ChatCompletion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait: int = 2 ** attempt
            logger.warning("OpenAI rate-limit; sleeping %ds", wait)
            time.sleep(wait)
        except (openai.APIConnectionError, openai.APIStatusError) as exc:
            logger.warning("Transient OpenAI error: %s; retry in 2s", exc)
            time.sleep(2)
        except Exception as exc:  # pragma: no cover
            logger.error("Fatal OpenAI error: %s", exc, exc_info=True)
            break
    return ""


# ─────────────────────────── Validation Helpers ─────────────────────────────
def _count_words(text: str) -> int:
    """
    Count the number of words in a given text by splitting on whitespace.

    Args:
        text (str): Input text string.

    Returns:
        int: Word count.
    """
    return len(re.findall(r"\S+", text))


def _normalize_recommendations_field(field: Any) -> str:
    """
    Normalize the 'recommendations' field which might be a string or list of strings.
    If it's a list, join items with newline characters.

    Args:
        field (Any): Recommendations field from parsed JSON.

    Returns:
        str: Single string with recommendations separated by newlines if applicable.
    """
    if isinstance(field, list):
        return "\n".join(str(item).strip() for item in field)
    return str(field)


def _is_valid_example(ex: Dict[str, Any]) -> bool:
    """
    Client-side sanity check to ensure that 'summary', 'context', and 'recommendations'
    exist and satisfy basic length constraints before passing to the critic.

    Constraints:
      - 'summary' must be a non-empty string with 5–90 words.
      - 'context' must be a non-empty string with 80–145 words.
      - 'recommendations' must be a non-empty field (further validated by critic).

    Args:
        ex (Dict[str, Any]): Example dictionary with keys 'summary', 'context', 'recommendations'.

    Returns:
        bool: True if example passes basic checks; False otherwise.
    """
    summary: str = ex.get("summary", "")
    context: str = ex.get("context", "")
    recs_field: Any = ex.get("recommendations", "")

    if not isinstance(summary, str) or not summary.strip():
        return False
    if not isinstance(context, str) or not context.strip():
        return False

    recs_text: str = _normalize_recommendations_field(recs_field)
    if not recs_text.strip():
        return False

    summary_wc: int = _count_words(summary)
    if summary_wc < 5 or summary_wc > 90:
        return False

    context_wc: int = _count_words(context)
    if context_wc < 80 or context_wc > 145:
        return False

    return True


# ─────────────────────────── Sequence: Summary → Context → Recommendations ────
def generate_summary(theme: str, temperature: float) -> Optional[str]:
    """
    Generate a 10–50-word startup idea ("summary") given a market theme.

    Steps:
      1. Construct user prompt with the provided theme.
      2. Call `chat` to get a JSON response from GPT.
      3. Parse the JSON and validate the summary word count (5–80 words for leniency).
      4. Return the summary string if valid, else None.

    Args:
        theme (str): Market theme (e.g., "food-tech").
        temperature (float): Sampling temperature for GPT generation.

    Returns:
        Optional[str]: The generated startup idea summary, or None if invalid.
    """
    user_prompt: str = f"Theme: {theme}\n\nGenerate JSON as specified above."
    raw_output: str = chat(
        SUMMARY_GEN_SYS,
        user_prompt,
        model=GEN_MODEL,
        temperature=temperature,
        max_tokens=100,
    )

    try:
        parsed: Dict[str, Any] = json.loads(raw_output)
        summary: str = parsed.get("summary", "").strip()
        if 5 <= _count_words(summary) <= 80:
            return summary
    except (json.JSONDecodeError, KeyError):
        logger.debug("Summary JSON parse failed:\n%s", raw_output)
    return None


def generate_context(summary: str, temperature: float) -> Optional[str]:
    """
    Using the given startup idea summary, search for a numeric snippet via DuckDuckGo
    and then generate a ~100-word market background ("context") from GPT.

    Steps:
      1. Randomly shuffle MARKET_KEYWORDS and iterate until a numeric snippet is found.
      2. If no snippet, log and return None.
      3. Construct GPT prompt using summary and snippet.
      4. Attempt up to MAX_GEN_ATTEMPTS to generate valid context JSON (80–140 words).
      5. Return the context string if valid, else None.

    Args:
        summary (str): Generated startup idea summary.
        temperature (float): Sampling temperature for GPT generation.

    Returns:
        Optional[str]: Market background context (~100 words), or None on failure.
    """
    snippet: Optional[str] = None
    keywords: List[str] = random.sample(MARKET_KEYWORDS, k=len(MARKET_KEYWORDS))
    for kw in keywords:
        query: str = f"{summary} {kw}"
        snippet = duck_top1_snippet(query)
        if snippet:
            break

    if snippet is None:
        logger.info("No market snippet found for summary: '%s'; skipping.", summary)
        return None

    user_prompt: str = (
        f"Summary: {summary}\n\n"
        f"Snippet: {snippet}\n\n"
        "Generate JSON as specified above."
    )

    for _ in range(MAX_GEN_ATTEMPTS):
        raw_output: str = chat(
            CONTEXT_GEN_SYS,
            user_prompt,
            model=GEN_MODEL,
            temperature=temperature,
            max_tokens=300,
        )
        try:
            parsed: Dict[str, Any] = json.loads(raw_output)
            context: str = parsed.get("context", "").strip()
            if 80 <= _count_words(context) <= 140:
                return context
        except (json.JSONDecodeError, KeyError):
            logger.debug("Context JSON parse failed:\n%s", raw_output)
            time.sleep(0.5)

    logger.debug("Failed to generate valid context for summary: '%s'.", summary)
    return None


def generate_recommendations(
        context: str, summary: str, temperature: float
) -> Optional[str]:
    """
    Generate four numbered VC recommendations (each ≤50 words) based on context and summary.

    Steps:
      1. Construct GPT prompt using summary and context.
      2. Attempt up to MAX_GEN_ATTEMPTS to get valid recommendations JSON.
      3. Normalize the field and ensure non-empty (critic will enforce exact constraints).
      4. Return the recommendations string if valid, else None.

    Args:
        context (str): Market background context (~100 words).
        summary (str): Generated startup idea summary.
        temperature (float): Sampling temperature for GPT generation.

    Returns:
        Optional[str]: Recommendations as a single string, or None if invalid.
    """
    user_prompt: str = (
        f"Summary: {summary}\n\n"
        f"Context: {context}\n\n"
        "Generate JSON as specified above."
    )

    for _ in range(MAX_GEN_ATTEMPTS):
        raw_output: str = chat(
            RECS_GEN_SYS,
            user_prompt,
            model=GEN_MODEL,
            temperature=temperature,
            max_tokens=300,
        )
        try:
            parsed: Dict[str, Any] = json.loads(raw_output)
            recs_field: Any = parsed.get("recommendations", "")
            recs_text: str = _normalize_recommendations_field(recs_field).strip()
            if recs_text:
                return recs_text
        except (json.JSONDecodeError, KeyError):
            logger.debug("Recommendations JSON parse failed:\n%s", raw_output)
            time.sleep(0.5)

    logger.debug("Failed to generate valid recommendations for summary: '%s'.", summary)
    return None


def generate_one(theme: str, temperature: float) -> Optional[Dict[str, str]]:
    """
    Generate a single SFT example for a given theme following the flow:
      1. Generate summary (startup idea).
      2. Generate context (market background) via DuckDuckGo snippet.
      3. Generate four VC recommendations.
      4. Validate client-side and return example dict if valid.

    Args:
        theme (str): Market theme for this example.
        temperature (float): Sampling temperature for GPT calls.

    Returns:
        Optional[Dict[str, str]]: Dictionary with keys 'summary', 'context',
        'recommendations' if all steps succeed and pass client-side checks; otherwise None.
    """
    summary: Optional[str] = generate_summary(theme, temperature)
    if not summary:
        logger.info("Failed to generate summary for theme '%s'; skipping.", theme)
        return None

    context: Optional[str] = generate_context(summary, temperature)
    if not context:
        return None

    recommendations: Optional[str] = generate_recommendations(context, summary, temperature)
    if not recommendations:
        logger.info("Failed to generate recommendations for summary: '%s'.", summary)
        return None

    example: Dict[str, str] = {
        "summary": summary,
        "context": context,
        "recommendations": recommendations,
    }

    if _is_valid_example(example):
        return example

    return None


def review_example(ex: Dict[str, str]) -> Tuple[bool, Optional[Dict[str, str]], str]:
    """
    Critic QA step for a single example. Uses CRITIC_SYS to validate fields.

    Steps:
      1. Serialize the example as JSON and send to GPT critic.
      2. Parse critic response JSON for pass/fail, reason, and optional fix.
      3. If critic provides a valid fix, return (True, fix_dict, reason).
      4. Otherwise, return (passed_flag, None, reason).

    Args:
        ex (Dict[str, str]): Example with keys 'summary', 'context', 'recommendations'.

    Returns:
        Tuple[bool, Optional[Dict[str, str]], str]:
          - passed_flag: True if critic approves or provides a valid fix.
          - fix_dict: Corrected example if critic returned a valid fix; otherwise None.
          - reason: Critic's explicit rejection reason or "No reason provided".
    """
    raw_verdict: str = chat(
        CRITIC_SYS,
        json.dumps(ex, ensure_ascii=False),
        model=CRITIC_MODEL,
        temperature=0.0,
        max_tokens=400,
    )

    try:
        verdict: Dict[str, Any] = json.loads(raw_verdict)
        passed: bool = verdict.get("pass", False)
        reason: str = verdict.get("reason", "No reason provided")
        fix_raw: Any = verdict.get("fix")

        if fix_raw and isinstance(fix_raw, dict):
            fix_summary: str = fix_raw.get("summary", "").strip()
            fix_context: str = fix_raw.get("context", "").strip()
            fix_recs_field: Any = fix_raw.get("recommendations", "")
            fix_recs_text: str = _normalize_recommendations_field(fix_recs_field).strip()

            fix: Dict[str, str] = {
                "summary": fix_summary,
                "context": fix_context,
                "recommendations": fix_recs_text,
            }
            if _is_valid_example(fix):
                return True, fix, reason

        return passed, None, reason
    except json.JSONDecodeError:
        logger.debug("Critic JSON parse failed:\n%s", raw_verdict)
        return False, None, "Critic JSON parse error"


def to_sharegpt(ex: Dict[str, str]) -> Dict[str, List[Dict[str, str]]]:
    """
    Convert a validated example into ShareGPT conversation format.

    Structure:
      • system: SYSTEM_PROMPT
      • user:   USER_PREFIX + summary + CTX_PREFIX + context
      • assistant: recommendations

    Args:
        ex (Dict[str, str]): Example with keys 'summary', 'context', 'recommendations'.

    Returns:
        Dict[str, List[Dict[str, str]]]: ShareGPT-formatted conversation dict.
    """
    user_msg: str = USER_PREFIX + ex["summary"] + CTX_PREFIX + ex["context"]
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": ex["recommendations"]},
        ]
    }


# ─────────────────────────── Main Synthesis Loop ─────────────────────────────
def synthesize(n: int, temperature: float) -> List[Dict[str, str]]:
    """
    Generate and validate 'n' SFT examples following the new flow:
      1. Randomly pick a theme.
      2. Generate summary → context → recommendations.
      3. Submit to critic; if accepted or critic provides a valid fix, append to approved list.
      4. Log rejections with reasons.

    Args:
        n (int): Number of validated examples to produce.
        temperature (float): Sampling temperature for GPT calls.

    Returns:
        List[Dict[str, str]]: List of approved examples with keys 'summary', 'context',
                              and 'recommendations'.
    """
    approved: List[Dict[str, str]] = []
    pbar = tqdm(total=n, desc="Approved")

    while len(approved) < n:
        theme: str = random.choice(THEMES)
        candidate: Optional[Dict[str, str]] = generate_one(theme, temperature)
        if not candidate:
            continue

        passed, fix, reason = review_example(candidate)
        if passed:
            final_example: Dict[str, str] = fix if fix else candidate
            approved.append(final_example)
            pbar.update(1)
            logger.info("PASS – %s", final_example["summary"][:60])
        else:
            logger.info("REJECT – %s | Reason: %s", candidate["summary"][:60], reason)
    pbar.close()
    return approved


# ─────────────────────────── File Writers ────────────────────────────────────
def write_jsonl(path: Path, rows: List[Dict[str, str]]) -> None:
    """
    Write each approved example as a ShareGPT conversation into a JSONL file.

    Each line contains one JSON-serialized ShareGPT conversation.

    Args:
        path (Path): File path to write the JSONL.
        rows (List[Dict[str, str]]): List of approved examples (with keys 'summary',
                                     'context', 'recommendations').
    """
    with path.open("w", encoding="utf-8") as fp:
        for ex in rows:
            fp.write(json.dumps(to_sharegpt(ex), ensure_ascii=False) + "\n")
    logger.info("Saved %d examples → %s", len(rows), path)


def write_dataset_info() -> None:
    """
    Write 'dataset_info.json' for LLaMA-Factory ingestion.

    Defines format and column tags for the SFT dataset.
    """
    info: Dict[str, Any] = {
        "sft_train": {
            "file_name": SFT_JSONL.name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }
    }
    DATASET_INFO.write_text(json.dumps(info, indent=2), encoding="utf-8")
    logger.info("Created %s", DATASET_INFO)


# ─────────────────────────── CLI & Entry-Point ───────────────────────────────
def parse_cli() -> argparse.Namespace:
    """
    Parse command-line arguments for the number of examples and temperature.

    Returns:
        argparse.Namespace: Parsed arguments with attributes 'num' (int) and
                            'temperature' (float).
    """
    parser = argparse.ArgumentParser(
        description="Generate SFT dataset following: summary → context → recommendations."
    )
    parser.add_argument(
        "-n", "--num", type=int, default=150,
        help="Number of validated examples to produce (100–200 recommended)."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Sampling temperature for the generator model."
    )
    return parser.parse_args()


def main() -> None:
    """
    Main entry point for the script:
      1. Parse CLI arguments.
      2. Generate 'n' validated examples.
      3. Shuffle and write results to SFT_JSONL.
      4. Write dataset_info.json descriptor.
    """
    args = parse_cli()
    logger.info("Target=%d | temperature=%.2f", args.num, args.temperature)

    rows: List[Dict[str, str]] = synthesize(args.num, args.temperature)
    random.shuffle(rows)

    write_jsonl(SFT_JSONL, rows)
    write_dataset_info()
    logger.info("🎉 Completed – %d examples saved to %s", len(rows), SFT_JSONL)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as exc:  # pragma: no cover
        logger.exception("Fatal error: %s", exc)
        sys.exit(1)
