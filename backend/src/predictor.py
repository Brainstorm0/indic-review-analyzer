
"""Model registry used by the API: loads any trained model lazily and returns class probabilities."""
import os
import urllib.request
import numpy as np
from .config import LABELS, MODELS_DIR, PROVIDERS, RESULTS_DIR
from .text import detect_lang, normalize_text
from .utils import get_device, load_json
def _ollama_up() -> bool:
    try:
        with urllib.request.urlopen(PROVIDERS["ollama"]["base_url"].rstrip("/") + "/models", timeout=1.5):
            return True
    except Exception:  # noqa: BLE001
        return False
class ModelRegistry:
    def __init__(self):
        self._loaded = {}
    def list_models(self, include_llm=True):
        items = []
        if MODELS_DIR.exists():
            for d in sorted(MODELS_DIR.iterdir()):
                if not (d.is_dir() and (d / "meta.json").exists()):
                    continue
                meta = load_json(d / "meta.json")
                mp = RESULTS_DIR / f"metrics_{d.name}.json"
                met = load_json(mp) if mp.exists() else {}
                items.append({"id": d.name, "type": meta["type"], "display": meta.get("display", d.name),
                              "langs": meta.get("langs"), "macro_f1": met.get("macro_f1"),
                              "accuracy": met.get("accuracy"), "latency_ms": met.get("latency_ms")})
        if not include_llm:
            return items
        for prov, cfg in PROVIDERS.items():
            available = _ollama_up() if prov == "ollama" else bool(os.getenv(cfg["key_env"] or "", ""))
            if available:
                mp = RESULTS_DIR / f"metrics_llm_{prov}_zs.json"
                met = load_json(mp) if mp.exists() else {}
                items.append({"id": f"llm:{prov}", "type": "llm",
                              "display": f"Zero-shot LLM ({prov}: {cfg['model']})", "langs": None,
                              "macro_f1": met.get("macro_f1"), "accuracy": met.get("accuracy"),
                              "latency_ms": None, "metric_scope": "llm-subset"})
        return items
    def default_model(self, models=None):
        models = self.list_models(include_llm=False) if models is None else models
        local = [m for m in models if m["type"] != "llm" and m.get("macro_f1") is not None]
        multi = [m for m in local if not m.get("langs")]
        pool = multi or local
        return max(pool, key=lambda m: m["macro_f1"])["id"] if pool else None
    def _load(self, model_id: str):
        if model_id in self._loaded:
            return self._loaded[model_id]
        if model_id.startswith("llm:"):
            from .llm import LLMClient
            obj = {"type": "llm", "client": LLMClient(model_id.split(":", 1)[1])}

        else:
            d = MODELS_DIR / model_id
            if not (d / "meta.json").exists():
                raise FileNotFoundError(f"Model '{model_id}' not found in {MODELS_DIR}")
            meta = load_json(d / "meta.json")
            if meta["type"] == "tfidf":
                import joblib
                obj = {"type": meta["type"], "pipe": joblib.load(d / "model.joblib")}
            elif meta["type"] == "rnn":
                import torch
                from .rnn_models import RNNClassifier
                from .text import TOKENIZERS, Vocab
                vocab = Vocab(load_json(d / "vocab.json"))
                model = RNNClassifier(len(vocab), emb_dim=meta["emb"], hidden=meta["hidden"],
                                      num_layers=meta["layers"], cell=meta["cell"],
                                      bidirectional=meta["bidirectional"], dropout=meta["dropout"],
                                      pooling=meta["pooling"])
                model.load_state_dict(torch.load(d / "model.pt", map_location="cpu", weights_only=True))
                model.eval()
                if meta["tokenizer"] == "subword-local":
                    from sentencepiece import SentencePieceProcessor

                    tok = SentencePieceProcessor(model_file=str(d / "tokenizer.model")).EncodeAsPieces
                else:
                    tok = TOKENIZERS[meta["tokenizer"]]
                obj = {"type": "rnn", "model": model, "vocab": vocab,
                       "tok": tok, "max_len": meta["max_len"]}
            elif meta["type"] == "transformer":
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
                device = get_device()
                tok = AutoTokenizer.from_pretrained(d, use_fast=not meta.get("slow_tokenizer", False))
                model = AutoModelForSequenceClassification.from_pretrained(d).to(device).eval()
                obj = {"type": "transformer", "tok": tok, "model": model, "device": device}
            else:
                raise ValueError(f"unknown model type {meta['type']}")
            obj["meta"] = meta
        self._loaded[model_id] = obj
        return obj
    def predict_proba(self, model_id: str, texts, langs=None) -> np.ndarray:
        obj = self._load(model_id)
        texts = [normalize_text(t) for t in texts]
        t = obj["type"]
        if t == "tfidf":
            return obj["pipe"].predict_proba(texts)
        if t == "rnn":
            import torch
            from .rnn_models import pad_batch
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 64):
                    seqs = [obj["vocab"].encode(obj["tok"](x), obj["max_len"]) for x in texts[i:i + 64]]
                    ids, lens = pad_batch(seqs)
                    out.append(torch.softmax(obj["model"](ids, lens), -1).numpy())
            return np.concatenate(out)
        if t == "transformer":
            import torch
            out = []
            with torch.no_grad():
                for i in range(0, len(texts), 32):
                    enc = obj["tok"](texts[i:i + 32], padding=True, truncation=True,
                                     max_length=obj["meta"].get("max_len", 128), return_tensors="pt").to(obj["device"])
                    out.append(torch.softmax(obj["model"](**enc).logits.float(), -1).cpu().numpy())
            return np.concatenate(out)

        if t == "llm":
            from .llm import classify_texts, to_probs
            langs = langs or [detect_lang(x) for x in texts]
            return to_probs(classify_texts(obj["client"], texts, langs, batch_size=10))
        raise ValueError(t)
    def explain(self, model_id: str, text: str, max_words: int = 60):
        """Occlusion importance: drop in P(predicted class) when each word is removed."""
        if model_id.startswith("llm:"):
            return None
        words = normalize_text(text).split()[:max_words]
        if len(words) < 2:
            return []
        base = self.predict_proba(model_id, [" ".join(words)])[0]
        pred = int(base.argmax())
        variants = [" ".join(words[:i] + words[i + 1:]) for i in range(len(words))]
        P = self.predict_proba(model_id, variants)
        return [{"token": w, "score": float(base[pred] - P[i][pred])} for i, w in enumerate(words)]
__all__ = ["ModelRegistry", "LABELS"]
