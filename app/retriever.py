from __future__ import annotations
from typing import Dict, List, Tuple

from rank_bm25 import BM25Okapi

import config
from app.store import load_store



def retrieve(
    query: str,
    embedder,
    pc_index,
    namespace: str,
    text_store_path: str,
    top_k: int = config.TOP_K,
    query_type: str = "narrow",
) -> List[Dict]:
    """Return up to top_k chunk dicts via Pinecone + BM25 + RRF.
    
    For narrow queries: return only top 2 chunks (1 from Pinecone + 1 from BM25).
    For broad queries: use full RRF with top_k chunks.
    """
    text_store = load_store(text_store_path)
    print(f"[DEBUG] namespace={namespace} | store_path={text_store_path} | store_size={len(text_store)} | query_type={query_type}")

    # 1. Vector search
    vector_ids: List[str] = []
    matches = []
    try:
        qvec = embedder.embed_query(query)
        res = pc_index.query(vector=qvec, top_k=top_k, namespace=namespace, include_metadata=True)
        matches = res.get("matches", [])
        for m in matches:
            print(f"ID: {m['id']}  Score: {m.get('score', 'N/A')}")

        vector_ids = [m["id"] for m in matches]
    except Exception as e:
        print(f"Vector search failed: {e}")

    # Print Vector Search chunks
    print("\n--- Vector Search Results ---")
    for m in matches:
        cid = m["id"]
        entry = text_store.get(cid, {}) if text_store else {}
        text = entry.get("text", m.get("metadata", {}).get("text", "N/A")) if isinstance(entry, dict) else entry
        print(f"ID: {cid}\nText: {text}\n")

    # 2. BM25
    bm25_ids: List[str] = []
    if text_store:
        try:
            keys = list(text_store.keys())
            corpus = [e["text"] if isinstance(e, dict) else e for e in text_store.values()]
            bm25 = BM25Okapi([doc.split() for doc in corpus])
            scores = bm25.get_scores(query.split())
            top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            bm25_ids = [keys[i] for i in top_idx]
        except Exception as e:
            print(f"BM25 search failed: {e}")

    # Print BM25 Search chunks
    print("--- BM25 Search Results ---")
    for cid in bm25_ids:
        entry = text_store.get(cid)
        text = entry["text"] if isinstance(entry, dict) else entry
        print(f"ID: {cid}\nText: {text}\n")

    # 3. RRF (k=60)
    rrf: Dict[str, float] = {}
    for rank, cid in enumerate(vector_ids):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (60 + rank + 1)
    for rank, cid in enumerate(bm25_ids):
        rrf[cid] = rrf.get(cid, 0) + 1.0 / (60 + rank + 1)

    ranked = sorted(rrf, key=lambda x: rrf[x], reverse=True)[:top_k]
    print(f"[DEBUG] vector_ids={len(vector_ids)} | bm25_ids={len(bm25_ids)} | ranked={len(ranked)}")

    if query_type == "narrow":
        ranked = ranked[:1]
        print(f"[DEBUG] Narrow query - limiting to top 1 chunk: {ranked}")

    results: List[Dict] = []
    for cid in ranked:
        entry = text_store.get(cid) if text_store else None
        match = next((m for m in matches if m.get("id") == cid), {})
        print(f"[DEBUG] cid={cid} | entry_found={entry is not None}")
        if not entry:
            continue
        if isinstance(entry, dict):
            results.append({
                "id": cid,
                "text": entry.get("text"),
                "source": entry.get("source"),
                "page": entry.get("page"),
                "score": match.get("score")
            })
        else:
            results.append({
                "id": cid,
                "text": entry,
                "source": "unknown",
                "page": None,
                "score": match.get("score")
            })
    return results
    # for cid in ranked:
    #     entry = text_store.get(cid)
    #     if not entry:
    #         continue
    #     if isinstance(entry, dict):
    #         results.append({"id": cid, "text": entry["text"], "source": entry["source"], "page": entry.get("page")})
    #     else:
    #         results.append({"id": cid, "text": entry, "source": "unknown", "page": None})
    
    # print("Final RRF combined results:", results)
    # return results