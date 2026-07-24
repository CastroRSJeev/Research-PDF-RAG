from __future__ import annotations
import os
import time
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from pinecone import Pinecone, ServerlessSpec

import config
from app.ingest import PineconeEmbedder, ingest_pdf
from app.retriever import retrieve
from app.prompt_builder import classify_query
from app.llm import generate_answer, validate_claims
from app.store import (
    load_registry, refresh_registry,
    register_incognito, unregister_incognito,
    load_store,
)
from app.guardrails import check_input, redact_pii
from app import audit as audit_log

# ── Globals ───────────────────────────────────────────────────────────
pc: Pinecone = None
pc_index = None
embedder: PineconeEmbedder = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global pc, pc_index, embedder
    pc = Pinecone(api_key=config.PINECONE_API_KEY)

    # Ensure index exists
    if config.PINECONE_INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=config.PINECONE_INDEX_NAME,
            dimension=config.VECTOR_DIM,
            metric="cosine",
            spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
        )
    pc_index = pc.Index(config.PINECONE_INDEX_NAME)
    embedder = PineconeEmbedder(pc)
    load_registry()
    yield


app = FastAPI(title="RAG Render", lifespan=lifespan)


# ── Request / Response models ─────────────────────────────────────────

class ChatRequest(BaseModel):
    question: str
    namespace: str | None = None


class ChatResponse(BaseModel):
    answer: str
    claims: list


class CloseRequest(BaseModel):
    namespace: str


# ── Helpers ───────────────────────────────────────────────────────────

def _text_store_path(namespace: str | None) -> str:
    if namespace and namespace.startswith("incognito_"):
        return str(config.DATA_DIR / "temp" / f"{namespace}.json")
    return config.TEXT_STORE_PATH


# ── Endpoints ─────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest_endpoint(file: UploadFile = File(...)):
    """Ingest a PDF into the default namespace."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    upload_dir = config.DATA_DIR / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with open(dest, "wb") as f:
        f.write(await file.read())

    try:
        total, new = ingest_pdf(dest, embedder, pc_index, config.NS_DEFAULT, config.TEXT_STORE_PATH)
        refresh_registry()
        audit_log.log_ingestion(file.filename, total, new)
        return {"file": file.filename, "total_chunks": total, "new_chunks": new}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_incognito")
async def upload_incognito(file: UploadFile = File(...)):
    """Ingest a PDF into a temporary incognito namespace."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    session_id = str(uuid.uuid4())
    namespace = f"incognito_{session_id}"
    temp_dir = config.DATA_DIR / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)

    temp_pdf = temp_dir / file.filename
    store_path = str(temp_dir / f"{namespace}.json")

    with open(temp_pdf, "wb") as f:
        f.write(await file.read())

    try:
        total, new = ingest_pdf(temp_pdf, embedder, pc_index, namespace, store_path)
        os.remove(temp_pdf)
        register_incognito(namespace, store_path)
        audit_log.log_ingestion(file.filename, total, new)
        return {"namespace": namespace, "total_chunks": total, "new_chunks": new}
    except Exception as e:
        if temp_pdf.exists():
            os.remove(temp_pdf)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/close_incognito")
async def close_incognito(req: CloseRequest):
    """Delete incognito namespace from Pinecone and local store."""
    ns = req.namespace
    store_path = str(config.DATA_DIR / "temp" / f"{ns}.json")
    try:
        pc_index.delete(delete_all=True, namespace=ns)
    except Exception:
        pass
    if os.path.exists(store_path):
        os.remove(store_path)
    unregister_incognito(ns)
    return {"message": "Incognito session closed."}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")

    # Input guardrail
    safe, reason = check_input(question)
    if not safe:
        audit_log.log_guardrail(question, "input_blocked", reason)
        return ChatResponse(answer=f"⚠️ {reason}", claims=[])

    namespace = req.namespace or config.NS_DEFAULT
    store_path = _text_store_path(req.namespace)
    query_type = classify_query(question)

    t0 = time.monotonic()

    # Retrieve
    chunks = retrieve(question, embedder, pc_index, namespace, store_path, top_k=config.TOP_K, query_type=query_type)
    if not chunks:
        return ChatResponse(answer="No relevant documents found.", claims=[])

    # Generate
    answer, claims = generate_answer(question, chunks, store_path)

    # Source registry hard-check
    claims = validate_claims(claims, question, store_path)

    # PII redaction on output
    answer = redact_pii(answer)

    latency_ms = (time.monotonic() - t0) * 1000
    audit_log.log_query(question, len(chunks), latency_ms)

    return ChatResponse(answer=answer, claims=claims)


@app.get("/documents")
async def list_documents():
    store = load_store(str(config.DATA_DIR / "file_hash_store.json"))
    names = [Path(p).name for p in store.keys()]
    return {"files": names}


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()
