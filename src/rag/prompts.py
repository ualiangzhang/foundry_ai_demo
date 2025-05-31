# src/rag/prompts.py
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_STARTUP = (
    "You are a seasoned startup mentor. Use the CONTEXT below to answer "
    "the user's question. If the context is insufficient, say so instead of guessing."
)

RAG_WRAPPER = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_STARTUP),
    ("user",   "CONTEXT:\n{context}\n\nQUESTION:\n{question}")
])

PROJECT_EVAL = ChatPromptTemplate.from_messages([
    ("system",
     "You are an investor evaluating startup summaries. Provide exactly FOUR "
     "numbered, actionable pieces of feedback covering market, product, "
     "business model, and team."),
    ("user",
     "### Summary:\n{question}\n\n### Reference materials:\n{context}")
])

PITCH_DECK = ChatPromptTemplate.from_messages([
    ("system",
     "You write concise bullet points for startup pitch-deck slides."),
    ("user",
     "### Venture:\n{question}\n\n### Research snippets:\n{context}\n\n"
     "Generate slides for Problem, Solution, Market, Business Model, Team.")
])
