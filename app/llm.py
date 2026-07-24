from __future__ import annotations
import json
import re
from typing import Dict, List, Optional, Tuple

from openai import OpenAI

import config
from app.store import is_valid_citation
from app import audit as audit_log
from app.prompt_builder import prepare_sentences_for_prompt, build_prompt, classify_query
from app.nli_verifier import nli_label


# ── Shared OpenAI-SDK client factory ─────────────────────────────────

def _make_client(base_url: str, api_key: str) -> OpenAI:
    return OpenAI(base_url=base_url, api_key=api_key)


# ── SID parsing ───────────────────────────────────────────────────────

_SID_RE = re.compile(r"\[([A-Z]{1,3}\d+)\]")
_SID_CLUSTER_RE = re.compile(r"((?:\[[A-Z]{1,3}\d+\])+)")
_PAREN_SID_RE = re.compile(r"\(([A-Z]{1,3}\d+)\)")


def _normalize_sids(text: str) -> str:
    """Convert (A1) style to [A1] style."""
    return _PAREN_SID_RE.sub(r"[\1]", text)


def _parse_cited_response(llm_output: str, sentence_map: Dict[str, Dict]) -> List[Dict]:
    """Parse claim text + trailing SID clusters from free-text LLM output.

    Returns list of dicts: {claim, cited_sids, premise_text, page, source, status}
    """
    llm_output = _normalize_sids(llm_output)
    parsed: List[Dict] = []
    cursor = 0

    for match in _SID_CLUSTER_RE.finditer(llm_output):
        claim_text = llm_output[cursor:match.start()].strip()
        cursor = match.end()
        if not claim_text:
            continue

        cited_sids = _SID_RE.findall(match.group(1))
        valid_sids, sources, pages = [], [], []
        for sid in cited_sids:
            meta = sentence_map.get(sid)
            if not meta:
                continue
            valid_sids.append(sid)
            sources.append(meta.get("source", "unknown"))
            pages.append(meta.get("page"))

        parsed.append({
            "claim": claim_text,
            "cited_sids": valid_sids,
            "premise_text": " ".join(sentence_map[s]["text"] for s in valid_sids),
            "sources": sources,
            "pages": pages,
            "status": "cited" if valid_sids else "unverified",
        })

    # Trailing uncited text
    trailing = llm_output[cursor:].strip()
    if trailing:
        parsed.append({
            "claim": trailing,
            "cited_sids": [],
            "premise_text": "",
            "sources": [],
            "pages": [],
            "status": "unverified",
        })

    return parsed


# ── Source registry validation ────────────────────────────────────────

def _norm_page(page) -> Optional[int]:
    if page is None:
        return None
    try:
        return int(page)
    except (ValueError, TypeError):
        return None


def _build_source_tag(source: str, page, store_path: str | None = None) -> Optional[str]:
    """Return a [Source: ...] tag if the (source, page) pair is in the registry."""
    page = _norm_page(page)
    if not is_valid_citation(source, page, store_path):
        return None
    return f"[Source: {source}, Page: {page}]" if page is not None else f"[Source: {source}]"


# ── LLM call ──────────────────────────────────────────────────────────

def _call_llm(prompt: str, model: str, max_tokens: int) -> str:
    client = _make_client(config.HF_BASE_URL, config.HF_TOKEN)
    try:
        # Check prompt length and warn if too long
        prompt_tokens = len(prompt.split())
        if prompt_tokens > 30000:
            print(f"[DEBUG] WARNING: Prompt is very long ({prompt_tokens} tokens)")
        
        result = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=0.0,
            messages=[{"role": "user", "content": prompt}],
            stop=["<|im_end|>", "<|im_start|>"],
        ).choices[0].message.content or ""
        result = result.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        print(f"[DEBUG] LLM response len={len(result)} | preview={repr(result[:200])}")
        return result
    except Exception as e:
        print(f"[DEBUG] LLM call failed: {e}")
        import traceback
        traceback.print_exc()
        return ""


# ── Sentence-level rendering ──────────────────────────────────────────

def _render_sentence_level(
    raw_output: str,
    sentence_map: Dict[str, Dict],
    query: str,
    store_path: str | None = None,
) -> Tuple[str, List[Dict]]:
    """Parse SID-cited output, run NLI, return (rendered_answer, claims_for_ui).

    claims_for_ui items: {text, source, page}  — same shape as before for the frontend.
    """
    parsed = _parse_cited_response(raw_output, sentence_map)
    rendered_lines: List[str] = []
    claims_for_ui: List[Dict] = []

    for item in parsed:
        claim = item["claim"].strip()
        if not claim or not re.search(r"[A-Za-z0-9]", claim):
            continue

        if item["status"] != "cited" or not item["cited_sids"]:
            # Pass through uncited lines (headers, transitions) as-is
            rendered_lines.append(claim)
            continue

        if nli_label(item["premise_text"], claim) == "Contradiction":
            continue

        # Build source tags, filtering registry violations
        tags: List[str] = []
        seen = set()
        for src, pg in zip(item["sources"], item["pages"]):
            key = (src, _norm_page(pg))
            if key in seen:
                continue
            seen.add(key)
            tag = _build_source_tag(src, pg, store_path)
            if tag:
                tags.append(tag)

        tag_str = " ".join(tags)
        rendered_lines.append(f"{claim} {tag_str}".strip())

        # One UI claim entry per unique source/page pair
        for src, pg in zip(item["sources"], item["pages"]):
            pg_norm = _norm_page(pg)
            if is_valid_citation(src, pg_norm, store_path):
                claims_for_ui.append({"text": claim, "source": src, "page": pg_norm})

    return "\n".join(rendered_lines).strip(), claims_for_ui


# ── Verification fallback (broad queries, second model) ───────────────

_VERIFY_SYSTEM = (
    "For each numbered (claim, chunk) pair below, respond with exactly one JSON object "
    "mapping each number to 'Y' (claim is supported by the chunk) or 'N' (not supported). "
    "No explanation. Only the JSON object."
)


def _verify_claims_llm(claims: List[Dict], chunks_by_key: Dict[str, str]) -> List[Dict]:
    """Batch-verify claims with HF_SECOND_MODEL. Returns only supported claims."""
    if not claims:
        return []
    pairs = [
        f'{i}. Claim: "{c["text"]}"\n   Chunk: "{chunks_by_key.get(c.get("source","") + "|" + str(c.get("page")), "")[:300]}"'
        for i, c in enumerate(claims, 1)
    ]
    client = _make_client(config.HF_BASE_URL, config.HF_TOKEN)
    try:
        raw = client.chat.completions.create(
            model=config.HF_SECOND_MODEL,
            max_tokens=config.HF_VERIFY_MAX_TOKENS,
            temperature=0.0,
            messages=[
                {"role": "system", "content": _VERIFY_SYSTEM},
                {"role": "user", "content": "\n\n".join(pairs)},
            ],
        ).choices[0].message.content or ""
        start, end = raw.find("{"), raw.rfind("}")
        if start != -1 and end != -1:
            data = json.loads(raw[start:end + 1])
            if isinstance(data, dict):
                return [c for i, c in enumerate(claims, 1) if data.get(str(i)) == "Y"]
    except Exception:
        pass
    return claims


# ── Public API ────────────────────────────────────────────────────────

def generate_answer(query: str, chunks: List[Dict], store_path: str | None = None) -> Tuple[str, List[Dict]]:
    """Sentence-level cited answer pipeline.

    1. Split chunks into sentences, assign SIDs.
    2. Build grounded prompt and call HF_FIRST_MODEL.
    3. Parse SID citations, run NLI verification, validate registry.
    4. Optionally run LLM verification pass for broad queries.
    5. Return (answer_text, claims_list) — same interface as before.
    """
    numbered_sentences_block, sentence_map = prepare_sentences_for_prompt(chunks)
    print(f"[DEBUG] chunks={len(chunks)} | sentences={len(sentence_map)} | block_len={len(numbered_sentences_block)}")

    if not numbered_sentences_block:
        return "The provided documents do not contain sufficient information to answer this question.", []

    query_type = classify_query(query)
    prompt = build_prompt(
        user_query=query,
        numbered_sentences_block=numbered_sentences_block,
        query_type=query_type,
    )

    print(f"[DEBUG] calling LLM model={config.HF_FIRST_MODEL} query_type={query_type}")
    print(f"[DEBUG] prompt length={len(prompt)} chars")
    raw = _call_llm(prompt, config.HF_FIRST_MODEL, config.HF_MAX_TOKENS)
    if not raw:
        print(f"[DEBUG] LLM returned empty response - checking if model is accessible")
        return "The provided documents do not contain sufficient information to answer this question.", []

    answer_text, claims = _render_sentence_level(raw, sentence_map, query, store_path)
    print(f"[DEBUG] raw_len={len(raw)} | answer_text_len={len(answer_text)} | claims={len(claims)}")

    if not answer_text:
        return "The provided documents do not contain sufficient information to answer this question.", []

    # Clean up any leaked SID markers (e.g. [A1]) but preserve [Source: ...] tags
    answer_text = re.sub(r"\s*\[[A-Z]{1,3}\d+\]\s*", " ", answer_text)
    answer_text = re.sub(r"  +", " ", answer_text).strip()

    # Optional LLM verification pass for broad queries
    if query_type == "broad" and len(claims) > 1:
        chunks_by_key = {
            f"{c['source']}|{_norm_page(c.get('page'))}": c["text"]
            for c in chunks
        }
        claims = _verify_claims_llm(claims, chunks_by_key)

    return answer_text, claims


def validate_claims(claims: List[Dict], query: str, store_path: str | None = None) -> List[Dict]:
    """Hard-check every (source, page) against the registry. Log violations."""
    valid = []
    for claim in claims:
        src = claim.get("source", "")
        page = _norm_page(claim.get("page"))
        claim["page"] = page
        if not is_valid_citation(src, page, store_path):
            audit_log.log_guardrail(
                query=query,
                event_type="citation_registry_violation",
                detail=f"source={src!r} page={page}",
            )
            continue
        valid.append(claim)
    return valid
