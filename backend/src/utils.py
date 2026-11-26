"""Seeding, device selection, JSON I/O and statistical metrics."""
import json
import math
import random
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from .config import LABELS

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
def get_device(allow_mps: bool = True) -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if allow_mps and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and (math.isnan(o) or math.isinf(o)):
        return None
    if isinstance(o, Path):
        return str(o)
    return o
def save_json(obj, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(obj), f, ensure_ascii=False, indent=2, default=str)
def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)
def expected_calibration_error(y_true, probs, n_bins: int = 10) -> float:
    probs = np.asarray(probs, dtype=float)
    y = np.asarray(y_true)
    conf, pred = probs.max(1), probs.argmax(1)
    acc = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(ece)

def compute_metrics(y_true, y_pred, probs=None) -> dict:
    labels = list(range(len(LABELS)))
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "per_class": {LABELS[i]: {"precision": float(p[i]), "recall": float(r[i]),
                                  "f1": float(f[i]), "support": int(s[i])} for i in labels},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
    if probs is not None:
        out["ece"] = expected_calibration_error(y_true, probs)
    return out
def bootstrap_ci(y_true, y_pred, metric: str = "macro_f1", n: int = 1000, seed: int = 42, groups=None):
    """95% percentile bootstrap CI. With `groups`, resamples whole groups (cluster bootstrap), so that
    correlated rows such as translations of the same source review are kept together."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if groups is not None:
        codes = np.unique(np.asarray(groups).astype(str), return_inverse=True)[1]
        members = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
    vals = []
    for _ in range(n):
        if groups is not None:
            pick = rng.integers(0, len(members), len(members))
            idx = np.concatenate([members[g] for g in pick])
        else:
            idx = rng.integers(0, len(y_true), len(y_true))
        if metric == "accuracy":
            vals.append(accuracy_score(y_true[idx], y_pred[idx]))
        else:
            vals.append(f1_score(y_true[idx], y_pred[idx], average="macro", labels=[0, 1], zero_division=0))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]
