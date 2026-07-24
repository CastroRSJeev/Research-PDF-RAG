from __future__ import annotations
import re
from typing import Tuple
import config

_INPUT_PATTERNS = [re.compile(p, re.IGNORECASE) for p in config.BLOCKED_PROMPT_PATTERNS]
_PII_PATTERNS = {label: re.compile(pat) for label, pat in config.PII_PATTERNS.items()}


def check_input(text: str) -> Tuple[bool, str]:
    """Return (is_safe, reason). Blocks on any matched injection pattern."""
    for pat in _INPUT_PATTERNS:
        if pat.search(text):
            return False, "Request blocked by security policy."
    return True, ""


def redact_pii(text: str) -> str:
    """Redact PII from output text."""
    for label, pat in _PII_PATTERNS.items():
        text = pat.sub(f"[REDACTED-{label.upper()}]", text)
    return text
