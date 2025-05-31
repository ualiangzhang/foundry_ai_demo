#!/usr/bin/env python3
"""
Read data_processed/rag_docs.jsonl, split any very long "text" into
~512-word chunks (whitespace-based), and write a new file
data_processed/rag_docs_chunked.jsonl.

This version uses text.split() instead of nltk.word_tokenize to avoid
any NLTK resource downloads or SSL issues.
"""

import json
import pathlib

# Paths
infile  = pathlib.Path("data_processed/rag_docs.jsonl")
outfile = pathlib.Path("data_processed/rag_docs_chunked.jsonl")

# Chunk parameters
MAX_WORDS    = 512   # max words per chunk
OVERLAP_WORDS = 50   # overlap between consecutive chunks

with open(infile, "r", encoding="utf-8") as fi, \
     open(outfile, "w", encoding="utf-8") as fo:
    for line in fi:
        obj = json.loads(line)
        text = obj["text"]
        words = text.split()  # whitespace-based tokenization
        n_words = len(words)

        if n_words <= MAX_WORDS:
            # Short enough → write unchanged
            fo.write(json.dumps(obj, ensure_ascii=False) + "\n")
        else:
            # Break into overlapping chunks
            step = MAX_WORDS - OVERLAP_WORDS
            for i in range(0, n_words, step):
                chunk_words = words[i : i + MAX_WORDS]
                chunk_text = " ".join(chunk_words)
                new_obj = {
                    "text": chunk_text,
                    "meta": obj["meta"]
                }
                fo.write(json.dumps(new_obj, ensure_ascii=False) + "\n")

print("✅  Saved data_processed/rag_docs_chunked.jsonl")
