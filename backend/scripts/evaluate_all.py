"""Leaderboard, per-language table, McNemar tests, fair LLM-subset comparison, ablations, figures, dashboard.json.
Run: python -m scripts.evaluate_all"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from src.config import FIG_DIR, MODELS_DIR, RESULTS_DIR  # noqa: E402
from src.data import load_dataset, split  # noqa: E402
from src.utils import load_json, save_json  # noqa: E402
ABLATIONS = [
    ("Classifier on the same TF-IDF features", ["tfidf_lr", "tfidf_nb", "tfidf_bnb"]),
    ("Transformer vs best classical and recurrent models", ["muril_multi", "tfidf_lr", "bilstm_subword"]),
    ("Zero-shot vs 3-shot LLM", ["llm_gemini_zs", "llm_gemini_fs3", "llm_groq_zs", "llm_groq_fs3",
                                 "llm_ollama_zs", "llm_ollama_fs3"]),
]
def display_name(mid, m):
    p = MODELS_DIR / mid / "meta.json"
    if p.exists():
        return load_json(p).get("display", mid)
    if m.get("llm"):
        return f"LLM {m['llm']} ({'zero-shot' if not m.get('shots') else str(m['shots']) + '-shot'})"
    return mid
def mcnemar(a, b):
    n01, n10 = int(np.sum(a & ~b)), int(np.sum(~a & b))
    p = 1.0 if n01 + n10 == 0 else float(binomtest(n01, n01 + n10, 0.5).pvalue)
    return n01, n10, p
def score(spec, metrics):
    if "@" in spec:
        mid, lang = spec.split("@")
        return metrics.get(mid, {}).get("per_lang", {}).get(lang, {}).get("macro_f1")
    return metrics.get(spec, {}).get("macro_f1")
def safe_fig(fn):
    def wrap(*a, **k):
        try:
            name = fn(*a, **k)
            if name:
                print(f"  figure -> artifacts/figures/{name}")
            return name
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] {fn.__name__}: {e}")
            return None
        finally:
            plt.close("all")
    return wrap
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=150)
    return name

@safe_fig
def fig_f1(full):
    if full.empty:
        return None
    d = full.sort_values("macro_f1")
    lo = np.clip(d["macro_f1"] - d["ci_low"].fillna(d["macro_f1"]), 0, None)
    hi = np.clip(d["ci_high"].fillna(d["macro_f1"]) - d["macro_f1"], 0, None)
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.5))
    ax.barh(d["display"], d["macro_f1"], xerr=[lo, hi], capsize=3, color="#4f46e5")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1 on held-out test set (95% bootstrap CI)")
    return _save(fig, "fig_macro_f1.png")
@safe_fig
def fig_heatmap(pl):
    if pl.empty:
        return None
    vals = pl.values.astype(float)
    fig, ax = plt.subplots(figsize=(1.1 * pl.shape[1] + 4, 0.45 * pl.shape[0] + 1.8))
    im = ax.imshow(vals, cmap="viridis", vmin=0.3, vmax=1.0, aspect="auto")
    ax.set_xticks(range(pl.shape[1]), pl.columns)
    ax.set_yticks(range(pl.shape[0]), pl.index, fontsize=8)
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if not np.isnan(vals[i, j]):
                ax.text(j, i, f"{vals[i, j]:.2f}", ha="center", va="center", fontsize=7,
                        color="w" if vals[i, j] < 0.75 else "k")
    fig.colorbar(im, ax=ax, label="Macro-F1")
    return _save(fig, "fig_per_language_heatmap.png")
@safe_fig
def fig_confusions(metrics, ids):
    ids = [i for i in ids if "confusion_matrix" in metrics[i]][:6]
    if not ids:
        return None
    cols = min(3, len(ids))
    rows = int(np.ceil(len(ids) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, mid in zip(axes, ids):
        cm = np.array(metrics[mid]["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, str(v), ha="center", va="center")
        ax.set_xticks([0, 1], ["neg", "pos"])
        ax.set_yticks([0, 1], ["neg", "pos"])
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(mid, fontsize=9)
    for ax in axes[len(ids):]:
        ax.axis("off")
    return _save(fig, "fig_confusion_matrices.png")
@safe_fig
def fig_curves():

    files = sorted(RESULTS_DIR.glob("history_*.json"))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(8, 5))
    for f in files:
        h = load_json(f)
        ax.plot([x["epoch"] for x in h], [x["val_macro_f1"] for x in h], marker="o", label=f.stem[len("history_"):])
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation macro-F1")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    return _save(fig, "fig_training_curves.png")
@safe_fig
def fig_reliability(preds, ids):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0.5, 1], [0.5, 1], "k--", lw=1, label="perfect calibration")
    for mid in ids[:4]:
        p = preds.get(mid)
        if p is None:
            continue
        conf = p[["p_neg", "p_pos"]].max(axis=1).values
        acc = (p["y_pred"] == p["y_true"]).values
        edges = np.linspace(0.5, 1.0, 6)
        xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            m = (conf >= lo) & (conf <= hi)
            if m.sum() >= 5:
                xs.append(conf[m].mean())
                ys.append(acc[m].mean())
        ax.plot(xs, ys, marker="o", label=mid)
    ax.set_xlabel("mean confidence")
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=7)
    return _save(fig, "fig_reliability.png")
@safe_fig
def fig_efficiency(full):
    d = full.dropna(subset=["latency_ms"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(d["latency_ms"], d["macro_f1"], s=60, color="#0891b2")
    for r in d.itertuples():
        ax.annotate(r.model_id, (r.latency_ms, r.macro_f1), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("latency per review (ms, log scale)")
    ax.set_ylabel("test macro-F1")
    ax.grid(alpha=0.3)
    return _save(fig, "fig_efficiency.png")
@safe_fig
def fig_ablation(abl):
    items = [(g["group"], it["label"], it["macro_f1"]) for g in abl for it in g["items"]]
    if not items:

        return None
    groups = list(dict.fromkeys(g for g, _, _ in items))
    palette = plt.cm.tab10.colors
    fig, ax = plt.subplots(figsize=(10, 0.38 * len(items) + 1.5))
    ax.barh(range(len(items)), [v for *_, v in items],
            color=[palette[groups.index(g) % 10] for g, _, _ in items])
    ax.set_yticks(range(len(items)), [f"{g} | {l}" for g, l, _ in items], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Macro-F1")
    return _save(fig, "fig_ablations.png")
def fmt(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.4f}" if isinstance(v, (float, np.floating)) else str(v)
def md_table(df, cols):
    s = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        s += "| " + " | ".join(fmt(r[c]) for c in cols) + " |\n"
    return s
def main():
    metrics, preds = {}, {}
    for f in sorted(RESULTS_DIR.glob("metrics_*.json")):
        mid = f.stem[len("metrics_"):]
        metrics[mid] = load_json(f)
        p = RESULTS_DIR / f"preds_{mid}.csv"
        if p.exists():
            preds[mid] = pd.read_csv(p)
    if not metrics:
        raise SystemExit("No metrics_*.json found - train models first.")
    full_ids = set(split(load_dataset(), "test")["id"])
    rows = []
    for mid, m in metrics.items():
        ids = set(preds[mid]["id"]) if mid in preds else set()
        scope = "full" if ids == full_ids else ("llm-subset" if m.get("subset") else
                                               "lang:" + "+".join(sorted(m.get("per_lang", {}))))
        ci = m.get("ci95_macro_f1", [None, None])
        rows.append(dict(model_id=mid, display=display_name(mid, m), family=m.get("family", ""), scope=scope,
                         n_test=m.get("n_test"), accuracy=m.get("accuracy"), macro_f1=m.get("macro_f1"),
                         ci_low=ci[0], ci_high=ci[1], ece=m.get("ece"), n_params=m.get("n_params"),
                         train_time_s=m.get("train_time_s"), latency_ms=m.get("latency_ms")))
    lb = pd.DataFrame(rows).sort_values("macro_f1", ascending=False).reset_index(drop=True)
    lb.to_csv(RESULTS_DIR / "leaderboard.csv", index=False)
    full = lb[lb["scope"] == "full"]
    best = full.iloc[0]["model_id"] if len(full) else lb.iloc[0]["model_id"]
    print(f"Best full-test model: {best}")
    per_lang = {mid: {l: v.get("macro_f1") for l, v in m.get("per_lang", {}).items()} for mid, m in metrics.items()}
    pl = pd.DataFrame(per_lang).T
    pl = pl.loc[lb["model_id"]] if len(pl) else pl
    pl.to_csv(RESULTS_DIR / "per_language_macro_f1.csv")

    mc = []
    if best in preds:
        b = preds[best].set_index("id")
        for mid in full["model_id"]:
            if mid == best or mid not in preds:
                continue
            o = preds[mid].set_index("id")
            common = b.index.intersection(o.index)
            bc = (b.loc[common, "y_pred"] == b.loc[common, "y_true"]).values
            oc = (o.loc[common, "y_pred"] == o.loc[common, "y_true"]).values
            n01, n10, p = mcnemar(bc, oc)
            mc.append({"model_a": best, "model_b": mid, "n": int(len(common)), "a_right_b_wrong": n01,
                       "a_wrong_b_right": n10, "p_value": p, "significant_at_0.05": p < 0.05})
    pd.DataFrame(mc).to_csv(RESULTS_DIR / "mcnemar.csv", index=False)
    sub, smc = [], []
    sp = RESULTS_DIR / "llm_subset_ids.json"
    if sp.exists():
        sids = set(load_json(sp))
        for mid, p in preds.items():
            q = p[p["id"].isin(sids)]
            if len(q) == len(sids):
                sub.append({"model_id": mid, "display": display_name(mid, metrics[mid]), "n": int(len(q)),
                            "accuracy": float(accuracy_score(q["y_true"], q["y_pred"])),
                            "macro_f1": float(f1_score(q["y_true"], q["y_pred"], average="macro", labels=[0, 1], zero_division=0))})
        sub.sort(key=lambda r: -r["macro_f1"])
        llms = [r for r in sub if r["model_id"].startswith("llm_")]
        locs = [r for r in sub if not r["model_id"].startswith("llm_")]
        if llms and locs:
            a = preds[llms[0]["model_id"]].set_index("id")
            for r in locs[:3]:
                o = preds[r["model_id"]].set_index("id")
                common = a.index.intersection(o.index)
                n01, n10, p = mcnemar((a.loc[common, "y_pred"] == a.loc[common, "y_true"]).values,
                                      (o.loc[common, "y_pred"] == o.loc[common, "y_true"]).values)
                smc.append({"model_a": llms[0]["model_id"], "model_b": r["model_id"], "n": int(len(common)),
                            "a_right_b_wrong": n01, "a_wrong_b_right": n10, "p_value": p,
                            "significant_at_0.05": p < 0.05})
    abl = []
    for group, specs in ABLATIONS:
        items = []
        for s in specs:
            v = score(s, metrics)
            if v is None:
                continue
            mid = s.split("@")[0]
            lab = display_name(mid, metrics[mid]) + (f" [{s.split('@')[1]} test]" if "@" in s else "")
            items.append({"spec": s, "label": lab, "macro_f1": v})
        if len(items) >= 2:
            abl.append({"group": group, "items": items})
    figs = [fig_f1(full), fig_heatmap(pl), fig_confusions(metrics, list(full["model_id"])), fig_curves(),
            fig_reliability(preds, list(full["model_id"])), fig_efficiency(full), fig_ablation(abl)]
    dash = {"best_model": best, "leaderboard": lb.to_dict(orient="records"), "per_lang": per_lang,

            "mcnemar": mc, "subset_leaderboard": sub, "subset_mcnemar": smc, "ablations": abl,
            "figures": [f for f in figs if f]}
    for name in ("data_stats", "hinglish_label_quality", "aspects_eval", "aspect_examples"):
        p = RESULTS_DIR / f"{name}.json"
        if p.exists():
            dash[name] = load_json(p)
    save_json(dash, RESULTS_DIR / "dashboard.json")
    md = "# Results summary\n\n## Leaderboard\n" + md_table(
        lb, ["model_id", "scope", "n_test", "accuracy", "macro_f1", "ci_low", "ci_high", "ece", "n_params", "latency_ms"])
    if mc:
        md += "\n## McNemar (best vs others, full test)\n" + md_table(pd.DataFrame(mc), list(mc[0].keys()))
    if sub:
        md += "\n## Fair comparison on the LLM subset\n" + md_table(pd.DataFrame(sub), ["model_id", "n", "accuracy", "macro_f1"])
    (RESULTS_DIR / "summary.md").write_text(md, encoding="utf-8")
    print(lb[["model_id", "scope", "accuracy", "macro_f1", "ci_low", "ci_high"]].to_string(index=False))
    print("DONE: artifacts/results/dashboard.json, summary.md, leaderboard.csv and figures written")
if __name__ == "__main__":
    main()
