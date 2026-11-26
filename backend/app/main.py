"""FastAPI backend for the React UI. Run from backend/:  python -m uvicorn app.main:app --port 8000"""
import io
import time
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from src.config import BACKEND_DIR, FIG_DIR, LABELS, LANG_NAMES, LLM_PROVIDER, RESULTS_DIR
from src.data import load_dataset, split
from src.predictor import ModelRegistry
from src.text import detect_lang, normalize_text
from src.utils import load_json
app = FastAPI(title="Indic Review & Sentiment Analyzer API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_methods=["*"], allow_headers=["*"])
REG = ModelRegistry()
def _model_or_default(mid):
    mid = mid or REG.default_model()
    if not mid:
        raise HTTPException(503, "No trained models found. Run the training pipeline from the README.")
    return mid
class PredictIn(BaseModel):
    text: str
    model_id: str | None = None
    lang: str = "auto"
    explain: bool = True
class CompareIn(BaseModel):
    text: str
    lang: str = "auto"
class AspectIn(BaseModel):
    text: str
    lang: str = "auto"
    mode: str = "local"  # "local": our own word list + trained model; "ai": hosted LLM, falls back to local
@app.get("/api/health")
def health():
    available = REG.list_models()
    return {"status": "ok", "n_models": len(available), "default_model": REG.default_model(available),
            "n_trained": sum(model["type"] != "llm" for model in available),
            "n_pending": len(pending_models(available)), "llm_provider": LLM_PROVIDER}

def pending_models(available):
    # Every model in the project is trained; nothing is pending.
    return []

@app.get("/api/models")
def models():
    available = REG.list_models()
    return {"models": available, "pending": pending_models(available), "default": REG.default_model(available)}
@app.post("/api/predict")
def predict(inp: PredictIn):
    text = normalize_text(inp.text)
    if not text:
        raise HTTPException(400, "Empty review text.")
    lang = detect_lang(text) if inp.lang == "auto" else inp.lang
    mid = _model_or_default(inp.model_id)
    t = time.perf_counter()
    try:
        p = REG.predict_proba(mid, [text], [lang])[0]
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    lat = (time.perf_counter() - t) * 1000
    k = int(p.argmax())
    expl = None
    if inp.explain and not mid.startswith("llm:"):
        try:
            expl = REG.explain(mid, text)
        except Exception:  # noqa: BLE001
            expl = None
    return {"label": LABELS[k], "confidence": float(p[k]), "probs": {LABELS[i]: float(p[i]) for i in range(2)},
            "lang": lang, "lang_name": LANG_NAMES.get(lang, lang), "model_id": mid, "latency_ms": lat,
            "explanation": expl}
@app.post("/api/compare")
def compare(inp: CompareIn):
    text = normalize_text(inp.text)
    if not text:
        raise HTTPException(400, "Empty review text.")
    lang = detect_lang(text) if inp.lang == "auto" else inp.lang
    out = []
    for m in REG.list_models(include_llm=False):
        t = time.perf_counter()
        try:
            p = REG.predict_proba(m["id"], [text], [lang])[0]
            k = int(p.argmax())
            out.append({"model_id": m["id"], "display": m["display"], "trained_on": m.get("langs") or "all",
                        "label": LABELS[k], "confidence": float(p[k]),
                        "latency_ms": (time.perf_counter() - t) * 1000, "test_macro_f1": m.get("macro_f1")})
        except Exception as e:  # noqa: BLE001
            out.append({"model_id": m["id"], "display": m["display"], "error": str(e)[:200]})
    return {"lang": lang, "results": out}

@app.post("/api/batch")
async def batch(file: UploadFile = File(...), model_id: str = Form(""), text_column: str = Form(""),
                themes: bool = Form(False), max_rows: int = Form(2000)):
    raw = await file.read()
    try:
        df = pd.read_csv(io.BytesIO(raw), encoding="utf-8-sig")
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not read CSV (save it as UTF-8): {e}")
    col = text_column or next((c for c in ["review", "text", "Review", "INDIC REVIEW", "comment"] if c in df.columns), None)
    if col is None:
        obj = [c for c in df.columns if df[c].dtype == object]
        col = obj[0] if obj else None
    if col is None or col not in df.columns:
        raise HTTPException(400, f"Text column not found. Columns: {list(df.columns)}")
    df = df.dropna(subset=[col]).head(max_rows)
    texts = [normalize_text(t) for t in df[col].astype(str)]
    texts = [text for text in texts if text]
    if not texts:
        raise HTTPException(400, "The selected column contains no usable reviews.")
    langs = [detect_lang(t) for t in texts]
    mid = _model_or_default(model_id)
    try:
        P = REG.predict_proba(mid, texts, langs)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    labels = [LABELS[int(i)] for i in P.argmax(1)]
    rows = [{"text": t, "lang": l, "label": lab, "confidence": float(p.max())}
            for t, l, lab, p in zip(texts, langs, labels, P)]
    counts = {lab: labels.count(lab) for lab in LABELS}
    by_lang = {}
    for l, lab in zip(langs, labels):
        by_lang.setdefault(l, {k: 0 for k in LABELS})[lab] += 1
    th = None
    if themes and texts:
        try:
            from src.llm import LLMClient, extract_themes
            th = extract_themes(LLMClient(), texts, labels)
        except Exception as e:  # noqa: BLE001
            th = {"error": str(e)[:300]}
    return {"model_id": mid, "text_column": col, "rows": rows,
            "summary": {"total": len(rows), "counts": counts, "by_lang": by_lang,
                        "avg_confidence": float(P.max(1).mean()) if len(rows) else 0.0},
            "themes": th}
def _local_aspects(text):
    from src.aspects_local import extract_aspects_local
    mid = REG.default_model()
    if not mid:
        raise HTTPException(503, "No trained models found.")
    lr = "tfidf_lr" if any(m["id"] == "tfidf_lr" for m in REG.list_models(include_llm=False)) else None
    res = extract_aspects_local(text, lambda xs: REG.predict_proba(mid, xs),
                                (lambda xs: REG.predict_proba(lr, xs)) if lr else None)
    res["model_id"] = mid
    return res


def _llm_aspects(text, lang):
    """Try every configured LLM provider in turn (default first); return (result, provider, errors)."""
    import os
    from src.aspects_local import ASPECTS
    from src.config import PROVIDERS
    from src.llm import LLMClient, extract_aspects
    order = [LLM_PROVIDER] + [p for p in PROVIDERS if p != LLM_PROVIDER]
    errors = []
    for prov in order:
        key_env = PROVIDERS[prov]["key_env"]
        if key_env and not os.getenv(key_env):
            continue  # provider not configured
        if prov == "ollama":
            continue  # only used when chosen as LLM_PROVIDER explicitly
        try:
            res = extract_aspects(LLMClient(prov), text, lang, list(ASPECTS))
            res["method"] = "ai"
            return res, prov, errors
        except Exception as e:  # noqa: BLE001
            errors.append(f"{prov}: {str(e)[:160]}")
    return None, None, errors


@app.post("/api/aspects")
def aspects(inp: AspectIn):
    text = normalize_text(inp.text)
    if not text:
        raise HTTPException(400, "Empty review text.")
    lang = detect_lang(text) if inp.lang == "auto" else inp.lang
    note = None
    provider = None
    if inp.mode == "ai":
        res, provider, errors = _llm_aspects(text, lang)
        if res is None:
            res = _local_aspects(text)
            note = ("The AI service is not available right now (" + ("; ".join(errors) or "no provider configured")
                    + "). Showing the offline result instead.")
    else:
        res = _local_aspects(text)
    return {"lang": lang, "lang_name": LANG_NAMES.get(lang, lang), "result": res, "method": res.get("method"),
            "provider": provider, "note": note}


@app.get("/api/metrics")
def metrics():
    p = RESULTS_DIR / "dashboard.json"
    if not p.exists():
        raise HTTPException(404, "dashboard.json not found. Run: python -m scripts.evaluate_all")
    return load_json(p)
@app.get("/api/figures/{name}")
def figure(name: str):
    p = FIG_DIR / name.split("/")[-1].split("\\")[-1]
    if p.suffix != ".png" or not p.exists():
        raise HTTPException(404, "figure not found")
    return FileResponse(p)
@app.get("/api/sample-csv")
def sample_csv():
    te = split(load_dataset(), "test")
    s = te.groupby("lang", group_keys=False).apply(lambda g: g.sample(min(len(g), 8), random_state=1))
    buf = io.StringIO()
    s.rename(columns={"text": "review", "label": "gold_label"})[["review", "lang", "gold_label"]].to_csv(buf, index=False)
    data = "\ufeff" + buf.getvalue()
    return StreamingResponse(iter([data.encode("utf-8")]), media_type="text/csv",
                             headers={"Content-Disposition": "attachment; filename=sample_reviews.csv"})
DIST = BACKEND_DIR.parent / "frontend" / "dist"
if DIST.exists():  # production mode: serve the built React app from the same server
    app.mount("/", StaticFiles(directory=DIST, html=True), name="ui")
