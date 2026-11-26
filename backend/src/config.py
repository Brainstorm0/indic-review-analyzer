"""Central configuration for the Indic Review & Sentiment Analyzer (SMAI A3 - T8)."""
import os
from pathlib import Path
from dotenv import load_dotenv
BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

DATA_RAW = BACKEND_DIR / "data" / "raw"
DATA_PROC = BACKEND_DIR / "data" / "processed"
ARTIFACTS = BACKEND_DIR / "artifacts"
MODELS_DIR = ARTIFACTS / "models"
RESULTS_DIR = ARTIFACTS / "results"
FIG_DIR = ARTIFACTS / "figures"
CACHE_DIR = ARTIFACTS / "llm_cache"
for _d in (DATA_RAW, DATA_PROC, MODELS_DIR, RESULTS_DIR, FIG_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)
SEED = int(os.getenv("SEED", "42"))
LANGS = [x.strip() for x in os.getenv("LANGS", "hi,ta,te,bn,mr").split(",") if x.strip()]
LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "bn": "Bengali", "mr": "Marathi",
    "gu": "Gujarati", "kn": "Kannada", "ml": "Malayalam", "pa": "Punjabi", "or": "Odia",
    "ur": "Urdu", "as": "Assamese", "hinglish": "Hinglish (code-mixed)", "en": "English",
}
LABELS = ["negative", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
HF_DATASET = "ai4bharat/IndicSentiment"
INDICBERT = "ai4bharat/indic-bert"
MURIL = "google/muril-base-cased"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
LLM_MIN_INTERVAL = float(os.getenv("LLM_MIN_INTERVAL", "4.0"))
PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "key_env": "GEMINI_API_KEY",
        "model": os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "key_env": "GROQ_API_KEY",
        "model": os.getenv("GROQ_MODEL", "openai/gpt-oss-20b"),
    },
    "ollama": {
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
        "key_env": None,
        "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
    },
}
