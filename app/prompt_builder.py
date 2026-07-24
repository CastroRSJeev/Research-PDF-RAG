from __future__ import annotations
import re
from typing import Dict, List, Tuple

from openai import OpenAI

import config
from app.sentence_splitter import split_chunk_into_sentences


def _index_to_letters(index: int) -> str:
    letters: List[str] = []
    index += 1
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        letters.append(chr(ord("A") + remainder))
    return "".join(reversed(letters))


def prepare_sentences_for_prompt(chunks: List[Dict]) -> Tuple[str, Dict[str, Dict]]:
    lines: List[str] = []
    sentence_map: Dict[str, Dict] = {}

    for chunk_idx, chunk in enumerate(chunks):
        letter = _index_to_letters(chunk_idx)
        text = chunk.get("text", "") if isinstance(chunk, dict) else str(chunk)
        page = chunk.get("page") if isinstance(chunk, dict) else None
        source = chunk.get("source", "unknown") if isinstance(chunk, dict) else "unknown"

        for sent_idx, sentence in enumerate(split_chunk_into_sentences(text), start=1):
            sid = f"{letter}{sent_idx}"
            lines.append(f"[{sid}] {sentence}")
            sentence_map[sid] = {"text": sentence, "page": page, "source": source}

    return "\n".join(lines), sentence_map


# ── Regex heuristic fallback ──────────────────────────────────────────

_BROAD_RE = re.compile(
    r"\b(what is|what are|explain|tell me about|describe|overview|summary|"
    r"summarize|how does|how do|walk me through|give me|list|what types|what kind)\b",
    re.IGNORECASE,
)


def _classify_heuristic(question: str) -> str:
    return "broad" if _BROAD_RE.search(question) else "narrow"


# ── LLM-based classification using HF_SECOND_MODEL ───────────────────
# DISABLED to save credits - use regex heuristic only
def classify_query(question: str) -> str:
    """Classify query as 'broad' or 'narrow' using regex heuristic only."""
    return _classify_heuristic(question)


# ── Prompt templates (from D:\RAG\prompt_templates.py) ───────────────

def build_prompt(
    *,
    user_query: str,
    numbered_sentences_block: str,
    query_type: str = "narrow",
) -> str:
    if query_type == "broad":
        return (
            "Answer the question using ONLY the numbered source sentences below. "
            "Each sentence has a unique ID like [A1].\n"
            "Rules: use bold section headers; cite every claim with its sentence ID(s) e.g. [A1][B2]; "
            "paraphrase, do not copy verbatim; skip any sub-topic not covered by the sources; "
            "never use outside knowledge or invent IDs; each ID at most once.\n\n"
            "SOURCE SENTENCES:\n"
            f"{numbered_sentences_block}\n\n"
            f"QUESTION: {user_query}\n\n"
            "ANSWER:"
        )

    return (
        "Answer the question using ONLY the numbered source sentences below. "
        "Each sentence has a unique ID like [A1].\n"
        "Rules: one or two sentences max; end with the supporting ID(s) e.g. [A1][B2]; "
        "no elaboration; if not covered respond: "
        "'The provided documents do not contain sufficient information to answer this question.'; "
        "never use outside knowledge or invent IDs.\n\n"
        "SOURCE SENTENCES:\n"
        f"{numbered_sentences_block}\n\n"
        f"QUESTION: {user_query}\n\n"
        "ANSWER:"
    )
