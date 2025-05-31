# tests/test_rag.py

import sys
from pathlib import Path

# ─── Add project root (one level up from tests/) to Python path ──────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ─── Now we can safely import from src/ ────────────────────────────────────────
from src.rag.chains import build_chain

if __name__ == "__main__":
    chain = build_chain(kind="eval", store="chroma")
    query = "Our startup produces mushroom-based leather. Could you critique our go-to-market plan?"
    print(chain(query)["result"])
