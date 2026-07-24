from __future__ import annotations
import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import fitz
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config
from app.store import load_store, save_store


# ── Text helpers (ported from reference project) ──────────────────────

def _strip_references(text: str) -> str:
    m = re.search(
        r"\n(?:#{1,3}\s*[\*_]{0,2}\s*)?(references|bibliography|works cited|reference list)[\*_]{0,2}\s*(\n|$)",
        text, flags=re.IGNORECASE,
    )
    return text[: m.start()] if m else text


def _strip_toc(text: str) -> str:
    heading = re.search(r"(?:^|\n)\s*(table\s+of\s+contents|contents)\s*\n", text, flags=re.IGNORECASE)
    if not heading:
        return text
    entry_re = re.compile(r"^.+[.\s]{2,}\d+\s*$")
    lines = text[heading.start():].splitlines(keepends=True)
    last_pos = 0
    running = heading.start()
    for line in lines:
        if entry_re.match(line.rstrip("\n")):
            last_pos = running + len(line)
        running += len(line)
    return text[last_pos:] if last_pos else text


def _strip_academic_citations(text: str) -> str:
    return re.sub(r" +", " ", re.sub(r"\[\d+(?:[\s,-]*\d+)*\]", "", text))


def _clean_chunk(text: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*\|\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*\|\s*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _is_toc_like(text: str) -> bool:
    lines = [l for l in text.splitlines() if l.strip()]
    if not lines:
        return False
    toc_re = re.compile(r"^.+[.\s]{2,}\d+\s*$")
    return sum(1 for l in lines if toc_re.match(l.strip())) / len(lines) > 0.50


def _is_citation_like(text: str) -> bool:
    markers = len(re.findall(r"\[\d{1,3}\]", text))
    vol_pp = len(re.findall(r"\b(vol\.|pp\.|no\.|Art\. no\.)\b", text, flags=re.IGNORECASE))
    words = max(len(text.split()), 1)
    if (markers + vol_pp) / words > 0.20:
        return True
    ref_lines = len(re.findall(r"^\s*(?:\[\d{1,3}\]|\d{1,3}\.\s)\s*\S", text, flags=re.MULTILINE))
    return ref_lines / max(text.count("\n") + 1, 1) > 0.40


# ── Block/column reordering (ported from reference project) ──────────

def _reorder_blocks(page) -> str:
    blocks = [
        {"bbox": b[:4], "text": b[4].strip()}
        for b in page.get_text("blocks")
        if b[6] == 0 and b[4].strip()
    ]
    if not blocks:
        return ""
    x0_counts = Counter(round(b["bbox"][0] / 5) * 5 for b in blocks)
    dom_x0 = x0_counts.most_common(1)[0][0]
    main = [b for b in blocks if b["bbox"][0] <= dom_x0 + 20.0]
    side = [b for b in blocks if b["bbox"][0] > dom_x0 + 20.0]
    parts = [b["text"] for b in sorted(main, key=lambda b: b["bbox"][1])]
    if side:
        parts.append("SIDEBAR: " + "\n\n".join(b["text"] for b in sorted(side, key=lambda b: b["bbox"][1])))
    return "\n\n".join(parts)


def _strip_boilerplate(pages: List[str], threshold: float = 0.35) -> List[str]:
    if len(pages) < 4:
        return pages
    line_counts: Dict[str, int] = {}
    for pt in pages:
        for line in set(pt.splitlines()):
            line = line.strip()
            if 4 <= len(line) <= 120:
                line_counts[line] = line_counts.get(line, 0) + 1
    boilerplate = {l for l, c in line_counts.items() if c / len(pages) >= threshold}
    return [
        "\n".join(l for l in pt.splitlines() if l.strip() not in boilerplate)
        for pt in pages
    ]


def _strip_author_bio(text: str) -> str:
    """Remove short author biography paragraphs that often appear after references.
    Heuristically drops trailing paragraphs under 200 characters containing a name pattern.
    """
    lines = text.splitlines()
    # Find the last index of a references heading if present
    ref_idx = -1
    for i, line in enumerate(lines):
        if re.search(r"\\n(?:#{1,3}\\s*[\\*_]{0,2}\\s*)?(references|bibliography|works cited|reference list)[\\*_]{0,2}\\s*(\\n|$)", line, flags=re.IGNORECASE):
            ref_idx = i
    if ref_idx == -1:
        return text
    # Consider lines after the references section as candidate bio
    candidate = lines[ref_idx + 1:]
    # If candidate paragraph is short and looks like a name, drop it
    if candidate:
        paragraph = " ".join(candidate).strip()
        if len(paragraph) < 200 and re.search(r"^[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*,?\s+(Ph\.D|M\.D|Prof\.|Dr\.)", paragraph):
            return "\n".join(lines[:ref_idx + 1])
    return text

# ── Chunk ID ──────────────────────────────────────────────────────────

def make_chunk_id(source: str, page, text: str) -> str:
    return hashlib.sha256(f"{source}::{page}::{text}".encode()).hexdigest()


# ── Pinecone embedding (Inference API only) ───────────────────────────

class PineconeEmbedder:
    def __init__(self, pc):
        self.pc = pc
        self.model = config.EMBEDDING_MODEL

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        results = []
        batch_size = 90
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = self.pc.inference.embed(
                model=self.model,
                inputs=batch,
                parameters={"input_type": "passage", "truncate": "END"},
            )
            results.extend(x.values for x in resp)
        return results

    def embed_query(self, text: str) -> List[float]:
        resp = self.pc.inference.embed(
            model=self.model,
            inputs=[text],
            parameters={"input_type": "query", "truncate": "END"},
        )
        return resp[0].values

# ── Core chunking ─────────────────────────────────────────────────────

def _chunk_text(text: str, splitter: RecursiveCharacterTextSplitter) -> List[str]:
    text = _strip_toc(text)
    text = _strip_references(text)
    text = _strip_academic_citations(text)
    text = _strip_author_bio(text)
    chunks = [_clean_chunk(c) for c in splitter.split_text(text)]
    chunks = [c for c in chunks if len(c.strip()) > config.MIN_CHUNK_LENGTH]
    chunks = [c for c in chunks if not _is_toc_like(c)]
    chunks = [c for c in chunks if not _is_citation_like(c)]
    return chunks


# ── Public ingestion function ─────────────────────────────────────────

def ingest_pdf(
    file_path: 'pdfs/DeepLearning.pdf',
    embedder: PineconeEmbedder,
    pc_index,
    namespace: str,
    text_store_path: 'data/chunk_text_store.json',
) -> Tuple[int, int]:
    """Extract, chunk, embed, upsert one PDF. Returns (total_chunks, new_chunks)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )

    # Extract pages
    page_chunks = pymupdf4llm.to_markdown(str(file_path), page_chunks=True)
    raw_pages = [(p.get("text", ""), p.get("metadata", {}).get("page_number", 1)) for p in page_chunks]

    doc = fitz.open(str(file_path))
    corrected: List[str] = []
    for (orig, _), fitz_page in zip(raw_pages, doc):
        reordered = _reorder_blocks(fitz_page)
        corrected.append(reordered if reordered.strip() else orig)
    doc.close()

    page_texts = _strip_boilerplate(corrected)
    raw_pages = [(corrected[i], pn) for i, (_, pn) in enumerate(raw_pages)]

    # Build chunk dicts
    all_chunks: List[Dict] = []
    for (_, page_num), text in zip(raw_pages, page_texts):
        for c in _chunk_text(text, splitter):
            all_chunks.append({"text": c, "source": file_path.name, "page": page_num})

    text_store = load_store(text_store_path)
    new_chunks = [c for c in all_chunks if make_chunk_id(c["source"], c["page"], c["text"]) not in text_store]

    if not new_chunks:
        return len(all_chunks), 0

    # Embed
    vectors = embedder.embed_documents([c["text"] for c in new_chunks])

    # Upsert to Pinecone + write to text store
    upsert_data = []
    for chunk, vec in zip(new_chunks, vectors):
        cid = make_chunk_id(chunk["source"], chunk["page"], chunk["text"])
        text_store[cid] = {"text": chunk["text"], "source": chunk["source"], "page": chunk["page"]}
        upsert_data.append({"id": cid, "values": vec, "metadata": {"source": chunk["source"], "page": chunk["page"]}})

    batch_size = 100
    for i in range(0, len(upsert_data), batch_size):
        pc_index.upsert(vectors=upsert_data[i: i + batch_size], namespace=namespace)

    save_store(text_store_path, text_store)
    return len(all_chunks), len(new_chunks)
