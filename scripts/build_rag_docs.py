#!/usr/bin/env python3
"""
Unify YC companies, generic startup blurbs, and Shark-Tank pitch summaries
into rag_docs.jsonl (min‐length 200 chars). Each line:
    {"text": <cleaned_text>, "meta": {"source": <source_tag>, "title": <title>}}

"""

import csv
import json
import pathlib
import re
from tqdm import tqdm

RAW = pathlib.Path("data_raw")
OUT = pathlib.Path("data_processed")
OUT.mkdir(exist_ok=True, parents=True)

writer = open(OUT / "rag_docs.jsonl", "w", encoding="utf-8")


def norm(txt: str) -> str:
    """
    Pitch summaries into rag_docs.jsonl (min‐length 200 chars)
    """
    return re.sub(r"\s+", " ", txt).strip()


def dump(text: str, source: str, title: str):
    """
    如果规范化后的文本长度 >= 200 字符，就写入一行 JSONL。
    """
    text = norm(text)
    if len(text) < 200:
        return
    record = {
        "text": text,
        "meta": {
            "source": source,
            "title": title
        }
    }
    writer.write(json.dumps(record, ensure_ascii=False) + "\n")


def detect_keys(fieldnames: list[str]) -> tuple[str | None, str | None]:
    """
    给定 CSV 的 header 列名列表，返回 (title_key, desc_key)。
    title_key：包含 "company" 或以 "name" 开头的列，desc_key：包含 "description"、"desc"、"idea" 或 "summary" 的列。
    若找不到，则返回 (None, None)。
    """
    title_key = None
    desc_key = None

    for k in fieldnames:
        kl = k.lower()
        # 检测 title 列
        if title_key is None and ("company" in kl or kl.startswith("name") or "title" in kl):
            title_key = k
        # 检测 description 列
        if desc_key is None and (
            "description" in kl or kl == "desc" or "idea" in kl or "summary" in kl
        ):
            desc_key = k

    return title_key, desc_key


# ─── 1. YC funded companies ─────────────────────────────────────────────────────
yc_folder = RAW / "yc"
processed_any = False

if yc_folder.exists():
    csv_files = list(yc_folder.glob("*.csv"))
    for yc_csv in csv_files:
        with open(yc_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            title_key, desc_key = detect_keys(fieldnames)

            if title_key and desc_key:
                print(f"[YC] Using '{yc_csv.name}' with title='{title_key}', desc='{desc_key}'")
                for row in reader:
                    text = row.get(desc_key, "")
                    title = row.get(title_key, "untitled")
                    dump(text, "yc_company", title)
                processed_any = True
            else:
                print(f"[YC] Skipping '{yc_csv.name}' (no title/desc columns). Found columns: {fieldnames}")

    if not processed_any:
        print("⚠️  Warning: No valid CSV in data_raw/yc contained both title & description columns.")
else:
    print("⚠️  Warning: Folder data_raw/yc not found; skipping YC step.")


# ─── 2. Generic startup descriptions ─────────────────────────────────────────────
gen_folder = RAW / "startup_desc"
processed_any = False

if gen_folder.exists():
    csv_files = list(gen_folder.glob("*.csv"))
    for gen_csv in csv_files:
        with open(gen_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []
            title_key, desc_key = detect_keys(fieldnames)

            if title_key and desc_key:
                print(f"[Generic] Using '{gen_csv.name}' with title='{title_key}', desc='{desc_key}'")
                for row in reader:
                    text = row.get(desc_key, "")
                    title = row.get(title_key, "untitled")
                    dump(text, "generic_desc", title)
                processed_any = True
            else:
                print(f"[Generic] Skipping '{gen_csv.name}' (no title/desc columns). Found columns: {fieldnames}")

    if not processed_any:
        print("⚠️  Warning: No valid CSV in data_raw/startup_desc contained both title & description columns.")
else:
    print("⚠️  Warning: Folder data_raw/startup_desc not found; skipping generic startup step.")


# ─── 3. Shark-Tank pitches ───────────────────────────────────────────────────────
shark_folder = RAW / "shark"
processed_any = False

if shark_folder.exists():
    csv_files = list(shark_folder.glob("*.csv"))
    for st_csv in csv_files:
        with open(st_csv, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or []

            # 优先检测 description 和 title 列
            title_key = next((k for k in fieldnames if "title" in k.lower()), None)
            desc_key  = next((k for k in fieldnames if "description" in k.lower()), None)
            # 兼作 idea/amount/equity 的备选方案
            idea_key = next((k for k in fieldnames if "idea" in k.lower()), None)
            amt_key  = next((k for k in fieldnames if "ask" in k.lower() or "amount" in k.lower()), None)
            eq_key   = next((k for k in fieldnames if "exchange" in k.lower() or "equity" in k.lower()), None)

            if title_key and desc_key:
                print(f"[Shark] Using '{st_csv.name}' with description='{desc_key}', title='{title_key}'")
                for row in reader:
                    text = row.get(desc_key, "")
                    # 拼接 ask/for 信息（如果存在）
                    ask = row.get(amt_key, "").strip() if amt_key else ""
                    equity = row.get(eq_key, "").strip() if eq_key else ""
                    if ask or equity:
                        # 如果 description 末尾不是句号，则加一个空格
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

            # 如果没有 description/title，再试 idea+title 的组合
            elif idea_key and title_key:
                print(f"[Shark] Using fallback '{st_csv.name}' with idea='{idea_key}', title='{title_key}'")
                for row in reader:
                    idea = row.get(idea_key, "")
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
                print(f"[Shark] Skipping '{st_csv.name}' (no usable columns). Found columns: {fieldnames}")

    if not processed_any:
        print("⚠️  Warning: No valid CSV in data_raw/shark contained required columns.")
else:
    print("⚠️  Warning: Folder data_raw/shark not found; skipping Shark-Tank step.")


writer.close()
print("✅  Saved", OUT / "rag_docs.jsonl")
