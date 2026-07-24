from __future__ import annotations
import re
from functools import lru_cache
from typing import List

_MIN_TOKENS = 3
_CAPTIONISH_RE = re.compile(
    r"^(?:sidebar:.*|fig(?:ure)?\.?\s*\d+.*|table\.?\s*\d+.*|equation\.?\s*\d+.*|ref\.?\s*\d+.*)$",
    re.IGNORECASE,
)
_HEADING_RE = re.compile(r"^(?:[#*•\-]+\s*|\d+\.\s+|[A-Z][A-Z\s]{4,}$|_{2,}|={2,})")
_BULLET_RE = re.compile(r"^[•\-\*o]\s+")
_PROTECTED_ABBREVIATIONS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "Prof.", "Sec.", "Fig.", "Eq.", "Ref.",
    "No.", "vs.", "e.g.", "i.e.", "et al.",
)


def _normalise_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\x08", " ").replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _clean_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence.strip())
    return re.sub(r"\s+([,.;:!?])", r"\1", sentence).strip(" -\t")


def _token_count(text: str) -> int:
    return len(re.findall(r"\b[\w\-]+\b", text))


def _is_noise_fragment(sentence: str) -> bool:
    if not sentence or not re.search(r"[A-Za-z0-9]", sentence):
        return True
    if _CAPTIONISH_RE.match(sentence):
        return True
    if _HEADING_RE.match(sentence) and len(sentence.split()) < 8:
        return True
    if _BULLET_RE.match(sentence):
        return True
    return _token_count(sentence) < _MIN_TOKENS


@lru_cache(maxsize=1)
def _get_pysbd_segmenter():
    try:
        import pysbd
        return pysbd.Segmenter(language="en", clean=False)
    except Exception:
        return None


def _regex_fallback(text: str) -> List[str]:
    text = re.sub(r"(\b\d+)\.(\d+\b)", r"\1<DECIMAL>\2", text)
    for abbr in _PROTECTED_ABBREVIATIONS:
        text = text.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\[(])", text)
    return [
        p.replace("<DOT>", ".").replace("<DECIMAL>", ".")
        for p in parts if p.strip()
    ]


def split_chunk_into_sentences(chunk_text: str) -> List[str]:
    text = _normalise_text(chunk_text)
    if not text:
        return []
    segmenter = _get_pysbd_segmenter()
    raw = segmenter.segment(text) if segmenter else _regex_fallback(text)
    return [s for s in (_clean_sentence(r) for r in raw) if not _is_noise_fragment(s)]
