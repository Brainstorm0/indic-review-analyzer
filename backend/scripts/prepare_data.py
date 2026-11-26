"""Clean IndicSentiment, group-aware stratified split (no cross-lingual leakage), statistics.
Run: python -m scripts.prepare_data"""
import argparse
import hashlib
import json
import re
from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_PROC, DATA_RAW, LABEL2ID, LABELS, LANGS, RESULTS_DIR, SEED
from src.text import normalize_text
from src.utils import save_json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    args = ap.parse_args()
    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    rows = []
    for lang in langs:
        for sp in ("validation", "test"):
            f = DATA_RAW / "indicsentiment" / "data" / sp / f"{lang}.json"
            if not f.exists():
                raise SystemExit(f"Missing {f}. Run: python -m scripts.download_data")
            with open(f, encoding="utf-8") as fh:
                for i, line in enumerate(fh):
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    rows.append({"src_id": f"{lang}-{sp}-{i}", "lang": lang, "text": d.get("INDIC REVIEW"),
                                 "english": d.get("ENGLISH REVIEW"), "label_raw": d.get("LABEL"),
                                 "aspects": d.get("ASPECTS"), "aspect_combo": d.get("ASPECT COMBO"),
                                 "category": d.get("CATEGORY"), "generic_category": d.get("GENERIC CATEGORIES"),
                                 "product": d.get("PRODUCT"), "brand": d.get("BRAND")})
    df = pd.DataFrame(rows)
    n_raw = len(df)
    df["label"] = df["label_raw"].fillna("").astype(str).str.strip().str.lower()
    print("Raw label distribution:", df["label"].value_counts().to_dict())
    df = df[df["label"].isin(LABELS)].copy()
    df["text"] = df["text"].fillna("").map(normalize_text)
    df = df[df["text"].str.len() >= 3].drop_duplicates(subset=["lang", "text"]).copy()
    df["english"] = df["english"].fillna("").map(normalize_text)
    df["group"] = [hashlib.md5(e.lower().encode("utf-8")).hexdigest()[:16] if e else f"solo-{sid}"
                   for e, sid in zip(df["english"], df["src_id"])]
    g = df.groupby("group")["label"].agg(lambda s: s.mode().iloc[0]).reset_index()
    tr, tmp = train_test_split(g, test_size=0.30, stratify=g["label"], random_state=SEED)
    va, te = train_test_split(tmp, test_size=0.50, stratify=tmp["label"], random_state=SEED)
    smap = {**{x: "train" for x in tr["group"]}, **{x: "val" for x in va["group"]}, **{x: "test" for x in te["group"]}}
    df["split"] = df["group"].map(smap)

    df["y"] = df["label"].map(LABEL2ID).astype(int)
    df["id"] = df["src_id"]
    cols = ["id", "lang", "text", "english", "label", "y", "split", "group", "aspects", "aspect_combo",
            "category", "generic_category", "product", "brand"]
    df = df[cols].reset_index(drop=True)
    df.to_csv(DATA_PROC / "all.csv", index=False, encoding="utf-8")
    leak = df.groupby("group")["split"].nunique().max() == 1
    stats = {
        "n_raw": n_raw, "n_clean": len(df),
        "by_lang_split": df.groupby(["lang", "split"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "label_by_lang": df.groupby(["lang", "label"]).size().unstack(fill_value=0).to_dict(orient="index"),
        "avg_words_by_lang": df.assign(n=df["text"].str.split().str.len()).groupby("lang")["n"].mean().round(1).to_dict(),
        "leakage_check_passed": bool(leak),
    }
    save_json(stats, RESULTS_DIR / "data_stats.json")
    asp = df[df["split"] == "train"]["aspects"].dropna().astype(str)
    c = Counter(a.strip().lower() for s in asp for a in re.split(r"[,;/|]+", s)
                if a.strip() and a.strip().lower() not in ("nan", "none", "na"))
    save_json([a for a, _ in c.most_common(60)], RESULTS_DIR / "aspect_vocab.json")
    print(f"Raw rows={n_raw}  clean rows={len(df)}")
    print(pd.DataFrame(stats["by_lang_split"]).T.to_string())
    print("Leakage check (each English-review group in exactly one split):", "PASSED" if leak else "FAILED")
    print("DONE: data/processed/all.csv written")
if __name__ == "__main__":
    main()
