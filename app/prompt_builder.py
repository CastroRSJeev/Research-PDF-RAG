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
            "RULE 0 (MOST IMPORTANT): Use ONLY the numbered source sentences below.\n"
            "Do not add facts, names, dates, or sections not present in them, even\n"
            "if you recognise them from general knowledge.\n\n"

            "You are answering a broad question using ONLY the numbered source sentences below.\n"
            "Each sentence has a unique ID in brackets, e.g. [A1].\n\n"

            "RULES:\n"
            "1. Structure your answer with bold section headers (e.g. **Overview**, **Key Features**, **Significance**).\n"
            "   Choose headers relevant to the topic — do not add a section for a sub-topic\n"
            "   that isn't covered by the source sentences.\n"
            "2. Every factual claim must end with the sentence ID(s) it came from: [A1].\n"
            "3. If a claim draws on more than one sentence, cite all of them together: [A1][C2].\n"
            "4. Paraphrase and synthesise — do not copy source text verbatim.\n"
            "5. Only include content that directly addresses the question. Ignore unrelated source sentences.\n"
            "6. If the source sentences do not contain enough information for a section, skip that section\n"
            "   and say so rather than filling it with outside knowledge.\n"
            "7. Do not use outside knowledge. Do not invent sentence IDs.\n"
            "8. Use ONLY square-bracket IDs copied exactly from the SOURCE SENTENCES block.\n"
            "9. Each sentence ID must appear AT MOST ONCE across the entire answer. If a sentence\n"
            "   already supports an earlier claim, do not cite it again in a later claim.\n\n"

            "WORKED EXAMPLE:\n"
            "SOURCE SENTENCES:\n"
            "[A1] Battery storage systems can respond within milliseconds.\n"
            "[A2] Battery storage systems reduce curtailment by storing surplus solar generation.\n"
            "[A3] Large-scale battery projects have been deployed in Australia and the US.\n"
            "QUESTION: Explain battery storage systems, including their environmental impact.\n"
            "ANSWER:\n"
            "**How They Work**\n"
            "Battery storage systems react almost instantly to grid fluctuations. [A1]\n\n"
            "**Benefits**\n"
            "They prevent wasted energy by storing excess solar output that would otherwise be discarded. [A2]\n\n"
            "**Deployment**\n"
            "Large-scale projects have been rolled out in Australia and the United States. [A3]\n\n"
            "**Environmental Impact**\n"
            "The source sentences do not cover environmental impact, so this cannot be answered from the provided material.\n\n"

            "SOURCE SENTENCES:\n"
            f"{numbered_sentences_block}\n\n"

            f"QUESTION: {user_query}\n\n"

            "Reminder: use ONLY the source sentences above, cite every claim, and\n"
            "explicitly skip anything not covered rather than filling gaps.\n\n"
            "ANSWER (use bold section headers, cite every factual claim with sentence IDs only):"
        )

    return (
        "RULE 0 (MOST IMPORTANT): Use ONLY the numbered source sentences below.\n"
        "Do not add facts not present in them, even if you know them to be true.\n\n"

        "You are answering a specific factual question using ONLY the numbered\n"
        "source sentences below. Each sentence has a unique ID in brackets, e.g. [A1].\n\n"

        "RULES:\n"
        "1. Give a direct, concise answer — one or two sentences maximum.\n"
        "2. End your answer with the sentence ID(s) that support it: [A1].\n"
        "3. If the answer draws on more than one sentence, cite all of them\n"
        "   together with no separators: [A1][B2].\n"
        "4. Do not add background, context, or elaboration beyond what is needed\n"
        "   to answer the question directly.\n"
        "5. If the source sentences do not contain the answer, respond only with:\n"
        "   'The provided documents do not contain sufficient information to\n"
        "    answer this question.'\n"
        "6. Do not use outside knowledge. Do not invent sentence IDs.\n"
        "7. Use ONLY square-bracket IDs copied exactly from the SOURCE SENTENCES\n"
        "   block. Never use parentheses like (A1) or bare labels like A1.\n"
        "8. Each sentence ID must appear AT MOST ONCE in your answer. Do not repeat\n"
        "   the same ID for multiple claims.\n\n"

        "WORKED EXAMPLE:\n"
        "SOURCE SENTENCES:\n"
        "[A1] The Eiffel Tower was completed in 1889.\n"
        "QUESTION: When was the Eiffel Tower completed?\n"
        "ANSWER: The Eiffel Tower was completed in 1889. [A1]\n\n"

        "SOURCE SENTENCES:\n"
        f"{numbered_sentences_block}\n\n"

        f"QUESTION: {user_query}\n\n"

        "Reminder: use ONLY the source sentences above.\n\n"
        "ANSWER (cite every factual claim with sentence IDs only):"
    )
