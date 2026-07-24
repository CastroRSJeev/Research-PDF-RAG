import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

# Mounted disk path — override via env var for Render
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
LOG_DIR = Path(os.getenv("LOG_DIR", str(BASE_DIR / "logs")))

DATA_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ── Pinecone ──────────────────────────────────────────────────────────
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-render")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
VECTOR_DIM = 1024
EMBEDDING_MODEL = "multilingual-e5-large"
NS_DEFAULT = "pdf-documents"

# ── LLM ───────────────────────────────────────────────────────────────
HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_BASE_URL = "https://router.huggingface.co/v1"
HF_FIRST_MODEL=os.getenv("HF_FIRST_MODEL", "microsoft/Phi-3-mini-4k-instruct")
HF_MAX_TOKENS = int(os.getenv("GEMINI_MAX_TOKENS", "1000"))
HF_VERIFY_MAX_TOKENS = 128

# ── Retrieval ─────────────────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "5"))

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "2000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
MIN_CHUNK_LENGTH = int(os.getenv("MIN_CHUNK_LENGTH", "50"))

# ── Storage ───────────────────────────────────────────────────────────
TEXT_STORE_PATH = str(DATA_DIR / "chunk_text_store.json")
AUDIT_LOG_PATH = str(LOG_DIR / "audit.jsonl")

# ── Guardrails ────────────────────────────────────────────────────────
BLOCKED_PROMPT_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"ignore\s+(all\s+)?above",
    r"disregard\s+(all\s+)?previous",
    r"you\s+are\s+now\s+(?:a|an)\s+(?!helpful)",
    r"pretend\s+(?:you\s+are|to\s+be)",
    r"act\s+as\s+(?:if|though)",
    r"reveal\s+(?:your|the)\s+(?:system|initial)\s+prompt",
    r"show\s+(?:me\s+)?(?:your|the)\s+(?:system|initial)\s+prompt",
    r"(?:sudo|admin)\s+mode",
    r"developer\s+mode",
    r"(?:DAN|jailbreak)\s+mode",
    r"generate\s+(?:the\s+)?bot\s+(?:response|message|reply)",
    r"(?:new|updated|override)\s+instructions?",
    r"system\s+prompt\s*:",
    r"<\s*(?:system|instruction|prompt)\s*>",
]

PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b",
    "phone_us": r"\b(?:\+1[\s.-]?)?(?:\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}\b",
    "phone_intl": r"\b\+\d{1,3}[\s.-]?\d{4,14}\b",
    "ssn": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit_card": r"\b(?:\d{4}[\s-]?){3}\d{4}\b",
    "ipv4": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "passport": r"\b[A-Z]{1,2}\d{6,9}\b",
}