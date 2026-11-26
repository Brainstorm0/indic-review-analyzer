"""Zero-/few-shot LLM sentiment on a fixed stratified test subset.
Run: python -m scripts.eval_llm --shots 0   and   python -m scripts.eval_llm --shots 3"""
import argparse
import os
import time
import pandas as pd
from src.config import RESULTS_DIR, SEED
from src.data import load_dataset, parse_langs, split
from src.evaluation import write_eval
from src.llm import LLMClient, classify_texts, to_probs
from src.utils import load_json, save_json
def get_subset(te, n_per_lang):
    path = RESULTS_DIR / "llm_subset_ids.json"
    if path.exists():
        ids = load_json(path)
        if set(ids) <= set(te["id"]):
            print(f"  reusing fixed subset ({len(ids)} ids). Delete {path.name} to resample.")
            return te[te["id"].isin(ids)]
    parts = [g.sample(min(len(g), n_per_lang), random_state=SEED) for _, g in te.groupby("lang")]
    sub = pd.concat(parts)
    save_json(sub["id"].tolist(), path)
    return sub
def few_shot_pool(tr, lang, k):
    g = tr[(tr["lang"] == lang) & (tr["text"].str.len() < 220)]
    pos = g[g["label"] == "positive"]
    neg = g[g["label"] == "negative"]
    pick = pd.concat([pos.sample(min(len(pos), (k + 1) // 2), random_state=SEED),
                      neg.sample(min(len(neg), k // 2), random_state=SEED)])
    return [{"text": r.text, "label": r.label, "lang": lang} for r in pick.itertuples()]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--providers", default=os.getenv("LLM_EVAL_PROVIDERS", "gemini"))
    ap.add_argument("--n-per-lang", type=int, default=40)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--langs", default="all")
    args = ap.parse_args()
    df = load_dataset(parse_langs(args.langs))

    tr, te = split(df, "train"), split(df, "test")
    sub = get_subset(te, args.n_per_lang)
    for prov in [p.strip() for p in args.providers.split(",") if p.strip()]:
        try:
            client = LLMClient(prov)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {prov}: {e}")
            continue
        print(f"== {client.name}  shots={args.shots}  n={len(sub)}")
        t0 = time.time()
        ordered, results = [], []
        for lang, g in sub.groupby("lang", sort=True):
            fs = few_shot_pool(tr, lang, args.shots) if args.shots else None
            results.extend(classify_texts(client, g["text"].tolist(), [lang] * len(g), args.batch_size, fs))
            ordered.append(g)
        sub_o = pd.concat(ordered).reset_index(drop=True)
        invalid = sum(r["label"] is None for r in results) / max(1, len(results))
        if invalid > 0.5:
            print(f"  [skip] {prov}: {invalid:.0%} unparseable answers - check the model name / quota.")
            continue
        model_id = f"llm_{prov}_{'zs' if args.shots == 0 else f'fs{args.shots}'}"
        write_eval(model_id, sub_o, to_probs(results), extra={
            "family": "llm", "llm": client.name, "shots": args.shots, "subset": True,
            "invalid_rate": invalid, "wall_time_s": time.time() - t0})
if __name__ == "__main__":
    main()
