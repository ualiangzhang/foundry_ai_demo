# tests/test_rag.py
from src.rag.chains import build_chain

if __name__ == "__main__":
    chain = build_chain(kind="eval", store="chroma")
    query = ("Our startup produces mushroom-based leather. "
             "Could you critique our go-to-market plan?")
    print(chain(query)["result"])
