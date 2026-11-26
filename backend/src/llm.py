"""One OpenAI-compatible client for Gemini, Groq and Ollama, with disk cache, throttling and retries."""
import hashlib
import json
import os
import re
import time
import numpy as np
from .config import CACHE_DIR, LABELS, LANG_NAMES, LLM_MIN_INTERVAL, LLM_PROVIDER, PROVIDERS
class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        from openai import OpenAI
        self.provider = (provider or LLM_PROVIDER).lower()
        if self.provider not in PROVIDERS:
            raise ValueError(f"Unknown provider '{self.provider}'. Use one of {list(PROVIDERS)}")
        cfg = PROVIDERS[self.provider]
        key = os.getenv(cfg["key_env"], "") if cfg["key_env"] else "ollama"
        if not key:
            raise RuntimeError(f"{cfg['key_env']} is empty in backend/.env")
        self.model = model or cfg["model"]
        self.client = OpenAI(base_url=cfg["base_url"], api_key=key, timeout=180, max_retries=0)
        self.min_interval = 0.0 if self.provider == "ollama" else LLM_MIN_INTERVAL
        self._last = 0.0
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", self.model)
        self.cache_path = CACHE_DIR / f"{self.provider}__{safe}.jsonl"
        self.cache = {}
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                try:
                    d = json.loads(line)

                    self.cache[d["k"]] = d["v"]
                except Exception:
                    pass
    @property
    def name(self) -> str:
        return f"{self.provider}:{self.model}"
    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 4096) -> str:
        k = hashlib.sha256(f"{self.model}\n{system}\n{user}\n{temperature}".encode("utf-8")).hexdigest()
        if k in self.cache:
            return self.cache[k]
        last_err = None
        for attempt in range(6):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                self._last = time.time()
                r = self.client.chat.completions.create(
                    model=self.model, temperature=temperature, max_tokens=max_tokens,
                    messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
                text = (r.choices[0].message.content or "").strip()
                if not text:
                    raise RuntimeError("empty response")
                self.cache[k] = text
                with open(self.cache_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"k": k, "v": text}, ensure_ascii=False) + "\n")
                return text
            except Exception as e:  # noqa: BLE001
                last_err = e
                msg = str(e)
                if any(x in msg for x in ("401", "403", "404", "invalid_api_key", "model_not_found", "decommissioned")):
                    break
                backoff = min(90, 5 * 2 ** attempt)
                print(f"  [LLM {self.name}] {type(e).__name__}: {msg[:160]} -> retry in {backoff}s")
                time.sleep(backoff)
        raise RuntimeError(f"LLM call failed ({self.name}): {last_err}")
def extract_json(text: str, prefer: str = "array"):
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    t = re.sub(r"```(?:json)?", "", t).strip()
    pairs = [("[", "]"), ("{", "}")] if prefer == "array" else [("{", "}"), ("[", "]")]
    for o, c in pairs:
        s, e = t.find(o), t.rfind(c)
        if s != -1 and e > s:
            try:
                return json.loads(t[s:e + 1])
            except json.JSONDecodeError:
                continue
    raise ValueError(f"No JSON found in LLM output: {text[:200]}")
SENT_SYSTEM = ("You are an expert multilingual sentiment annotator for Indian-language product and service "
               "reviews (Hindi, Tamil, Telugu, Bengali, Marathi, Hinglish and others). Classify the OVERALL "

               "sentiment of each review as 'positive' or 'negative'. Reply with JSON only.")
def _sent_prompt(items, few_shot):
    lines = []
    if few_shot:
        lines.append("Labelled examples:")
        for ex in few_shot:
            lines.append(f'- ({LANG_NAMES.get(ex["lang"], ex["lang"])}) "{ex["text"]}" -> {ex["label"]}')
        lines.append("")
    lines.append("Classify these reviews:")
    for i, (t, l) in enumerate(items, 1):
        lines.append(f"{i}. ({LANG_NAMES.get(l, l)}) {t}")
    lines.append('\nReturn a JSON array with one object per review, in order: '
                 '[{"id": 1, "label": "positive" or "negative", "confidence": number between 0.5 and 1.0}]')
    return "\n".join(lines)
def classify_texts(client: LLMClient, texts, langs, batch_size: int = 10, few_shot=None):
    out = []
    for s in range(0, len(texts), batch_size):
        items = list(zip(texts[s:s + batch_size], langs[s:s + batch_size]))
        res = [{"label": None, "confidence": 0.5} for _ in items]
        try:
            data = extract_json(client.chat(SENT_SYSTEM, _sent_prompt(items, few_shot)), "array")
            if isinstance(data, dict):
                data = data.get("results") or data.get("reviews") or [data]
            for j, obj in enumerate(data):
                idx = int(obj.get("id", j + 1)) - 1
                lab = str(obj.get("label", "")).strip().lower()
                if 0 <= idx < len(items) and lab in LABELS:
                    try:
                        conf = float(obj.get("confidence", 0.8))
                    except (TypeError, ValueError):
                        conf = 0.8
                    res[idx] = {"label": lab, "confidence": min(1.0, max(0.5, conf))}
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] LLM batch starting at {s}: {e}")
        out.extend(res)
    return out
def to_probs(results) -> np.ndarray:
    P = np.zeros((len(results), 2))
    for i, r in enumerate(results):
        if r["label"] is None:
            P[i] = [0.5, 0.5]
        else:
            c = r["confidence"]
            P[i] = [1 - c, c] if r["label"] == "positive" else [c, 1 - c]
    return P
HING_SYSTEM = ("You rewrite English product reviews as natural Hinglish (Hindi-English code-mixed) written ONLY "
               "in Roman/Latin script, the way Indian e-commerce customers write. Keep the exact meaning and "
               "sentiment. Reply with JSON only.")

def to_hinglish(client: LLMClient, texts, batch_size: int = 10):
    out = []
    for s in range(0, len(texts), batch_size):
        chunk = texts[s:s + batch_size]
        user = ("Rewrite each review in Hinglish (Roman script, mix Hindi and English words naturally).\n"
                + "\n".join(f"{i}. {t}" for i, t in enumerate(chunk, 1))
                + '\n\nReturn a JSON array: [{"id": 1, "hinglish": "..."}]')
        res = [""] * len(chunk)
        try:
            data = extract_json(client.chat(HING_SYSTEM, user, temperature=0.7), "array")
            for j, o in enumerate(data):
                idx = int(o.get("id", j + 1)) - 1
                if 0 <= idx < len(chunk):
                    res[idx] = str(o.get("hinglish", "")).strip()
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] Hinglish batch {s}: {e}")
        out.extend(res)
    return out
ASPECT_SYSTEM = "You are an aspect-based sentiment analysis (ABSA) engine for Indian-language reviews. Reply with JSON only."
def extract_aspects(client: LLMClient, text: str, lang: str = "auto", aspect_vocab=None) -> dict:
    hint = f"Prefer these aspect names when they fit: {', '.join(aspect_vocab[:40])}.\n" if aspect_vocab else ""
    user = (f"Review ({LANG_NAMES.get(lang, lang)}): {text}\n\n{hint}"
            "Identify every product/service aspect mentioned (e.g. price, quality, battery, delivery, taste, "
            "service, design, performance). For each give its sentiment (positive/negative/neutral), a short "
            "evidence phrase copied from the review, and an English translation of the evidence.\n"
            'Return JSON: {"overall": "positive|negative|neutral", "translation_en": "...", '
            '"aspects": [{"aspect": "...", "sentiment": "...", "evidence": "...", "evidence_en": "..."}]}')
    data = extract_json(client.chat(ASPECT_SYSTEM, user), "object")
    if not isinstance(data, dict):
        data = {"aspects": data}
    data.setdefault("aspects", [])
    return data
def extract_themes(client: LLMClient, texts, labels, max_reviews: int = 60) -> dict:
    rng = np.random.default_rng(0)
    idx = rng.choice(len(texts), size=min(max_reviews, len(texts)), replace=False)
    listing = "\n".join(f"- [{labels[i]}] {texts[i][:300]}" for i in idx)
    user = ("Below are customer reviews (with predicted sentiment) for a small Indian business. They may be in "
            "different Indian languages. Identify the most frequent recurring themes.\n\n" + listing +
            '\n\nReturn JSON in English: {"positive_themes": [{"theme": "...", "count_estimate": 0}], '
            '"negative_themes": [{"theme": "...", "count_estimate": 0}], '
            '"summary": "2-3 sentences for the business owner", "action_items": ["..."]}')
    return extract_json(client.chat("You are a customer-insights analyst. Reply with JSON only.", user), "object")
