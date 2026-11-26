"""Uniform evaluation: every model writes preds_<id>.csv and metrics_<id>.json."""
import time
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from .config import RESULTS_DIR
from .utils import bootstrap_ci, compute_metrics, save_json
def write_eval(model_id: str, test_df: pd.DataFrame, probs, extra: dict | None = None) -> dict:
    probs = np.asarray(probs, dtype=float)
    test_df = test_df.reset_index(drop=True)
    y_true = test_df["y"].astype(int).values
    y_pred = probs.argmax(1)
    pd.DataFrame({"id": test_df["id"], "lang": test_df["lang"], "y_true": y_true, "y_pred": y_pred,
                  "p_neg": probs[:, 0], "p_pos": probs[:, 1]}).to_csv(
        RESULTS_DIR / f"preds_{model_id}.csv", index=False, encoding="utf-8")
    m = compute_metrics(y_true, y_pred, probs)
    groups = test_df["group"].values if "group" in test_df.columns else None
    m["ci95_macro_f1"] = bootstrap_ci(y_true, y_pred, "macro_f1", groups=groups)
    m["ci95_accuracy"] = bootstrap_ci(y_true, y_pred, "accuracy", groups=groups)
    m["ci_method"] = "group bootstrap (source review)" if groups is not None else "row bootstrap"

    per = {}
    tmp = test_df.assign(_p=y_pred)
    for lang, g in tmp.groupby("lang"):
        per[lang] = {"n": int(len(g)), "accuracy": float(accuracy_score(g["y"], g["_p"])),
                     "macro_f1": float(f1_score(g["y"], g["_p"], average="macro", labels=[0, 1], zero_division=0))}
    m.update({"per_lang": per, "model_id": model_id, "n_test": int(len(test_df))})
    if extra:
        m.update(extra)
    save_json(m, RESULTS_DIR / f"metrics_{model_id}.json")
    print(f"[{model_id}] test acc={m['accuracy']:.4f}  macro-F1={m['macro_f1']:.4f}  "
          f"95%CI={[round(x, 4) for x in m['ci95_macro_f1']]}  ECE={m.get('ece', 0):.4f}")
    return m
def measure_latency(predict_fn, texts, n: int = 50) -> float:
    sample = list(texts)[:n]
    t = time.perf_counter()
    for s in sample:
        predict_fn([s])
    return (time.perf_counter() - t) / max(1, len(sample)) * 1000.0
