from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Optional, Set, Tuple
import config

# ── JSON store helpers ────────────────────────────────────────────────

def load_store(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_store(path: str, data: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# ── Source registry ───────────────────────────────────────────────────
# In-memory set of (source, page) pairs from chunk_text_store.json.
# Built at startup; refreshed after each ingestion.

_registry: Set[Tuple[str, Optional[int]]] = set()
# Per-incognito-namespace entries for clean removal on close
_incognito_entries: dict[str, Set[Tuple[str, Optional[int]]]] = {}


def _build_registry(store: dict) -> Set[Tuple[str, Optional[int]]]:
    result: Set[Tuple[str, Optional[int]]] = set()
    for entry in store.values():
        if isinstance(entry, dict):
            src = entry.get("source")
            if src:
                src = Path(src).name
                raw = entry.get("page")
                try:
                    page = int(raw) if raw is not None else None
                except (ValueError, TypeError):
                    page = None
                result.add((src, page))
    return result


def load_registry() -> None:
    global _registry
    store = load_store(config.TEXT_STORE_PATH)
    _registry = _build_registry(store)


def refresh_registry() -> None:
    load_registry()


def is_valid_citation(source: str, page) -> bool:
    try:
        page = int(page) if page is not None else None
    except (ValueError, TypeError):
        page = None
    return (Path(source).name, page) in _registry


def register_incognito(namespace: str, store_path: str) -> None:
    global _registry
    entries = _build_registry(load_store(store_path))
    _incognito_entries[namespace] = entries
    _registry |= entries


def unregister_incognito(namespace: str) -> None:
    global _registry
    entries = _incognito_entries.pop(namespace, set())
    # Only remove entries not present in any other incognito session or the global store
    still_needed = set()
    for other_entries in _incognito_entries.values():
        still_needed |= other_entries
    global_store = _build_registry(load_store(config.TEXT_STORE_PATH))
    still_needed |= global_store
    _registry -= (entries - still_needed)
