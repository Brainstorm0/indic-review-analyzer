"""T8.8: evaluate LLM aspect extraction against the IndicSentiment ASPECTS field.
Run: python -m scripts.eval_aspects --n 30"""
import argparse
import re
import pandas as pd
from scipy.optimize import linear_sum_assignment
from src.config import RESULTS_DIR, SEED
from src.data import load_dataset, split
from src.llm import LLMClient, extract_aspects
from src.utils import load_json, save_json
def parse_aspects(s):
    if not isinstance(s, str):
        return set()
    return {a.strip().lower() for a in re.split(r"[,;/|]+", s)
            if a.strip() and a.strip().lower() not in ("nan", "none", "na")}
def count_matches(gold, predicted):
    if not gold or not predicted:
        return 0
    matches = [[int(g in p or p in g) for p in sorted(predicted)] for g in sorted(gold)]
    gold_indices, pred_indices = linear_sum_assignment(matches, maximize=True)
    return sum(matches[i][j] for i, j in zip(gold_indices, pred_indices))
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)

    ap.add_argument("--provider", default=None)
    args = ap.parse_args()
    te = split(load_dataset(include_hinglish=False), "test")
    te = te[te["aspects"].map(lambda s: len(parse_aspects(s)) > 0)]
    if te.empty:
        raise SystemExit("No ASPECTS values found in test data - see Phase 3 (aspect format).")
    per = max(1, args.n // te["lang"].nunique())
    sample = pd.concat([g.sample(min(len(g), per), random_state=SEED) for _, g in te.groupby("lang")])
    vp = RESULTS_DIR / "aspect_vocab.json"
    vocab = load_json(vp) if vp.exists() else None
    client = LLMClient(args.provider)
    tp = fp = fn = sent_ok = n_ok = 0
    examples = []
    for r in sample.itertuples():
        try:
            res = extract_aspects(client, r.text, r.lang, vocab)
        except Exception as e:  # noqa: BLE001
            print("  [warn]", e)
            continue
        pred = {str(a.get("aspect", "")).strip().lower() for a in res.get("aspects", []) if a.get("aspect")}
        pred.discard("")
        gold = parse_aspects(r.aspects)
        matched = count_matches(gold, pred)
        tp += matched
        fn += len(gold) - matched
        fp += len(pred) - matched
        sent_ok += int(str(res.get("overall", "")).lower() == r.label)
        n_ok += 1
        if len(examples) < 10:
            examples.append({"lang": r.lang, "text": r.text, "gold_aspects": sorted(gold), "llm": res})
    if not n_ok:
        raise SystemExit("No aspect requests succeeded. Check the provider and retry.")
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    out = {"llm": client.name, "n": n_ok, "aspect_precision": P, "aspect_recall": R, "aspect_f1": F,
           "overall_sentiment_accuracy": sent_ok / max(1, n_ok),
           "matching": "one-to-one case-insensitive substring matching"}
    save_json(out, RESULTS_DIR / "aspects_eval.json")
    save_json(examples, RESULTS_DIR / "aspect_examples.json")
    print(out)
if __name__ == "__main__":
    main()
