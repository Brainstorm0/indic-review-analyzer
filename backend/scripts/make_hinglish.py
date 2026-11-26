"""T8.5: LLM-generated Hinglish reviews + LLM labels (as in the PDF), with a gold-label audit (Cohen's kappa).
Run: python -m scripts.make_hinglish --n 500"""
import argparse
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from src.config import DATA_PROC, LABEL2ID, RESULTS_DIR, SEED
from src.data import load_dataset
from src.llm import LLMClient, classify_texts, to_hinglish

from src.text import normalize_text
from src.utils import save_json
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--provider", default=None, help="gemini | groq | ollama (default: LLM_PROVIDER)")
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--hingcorpus", default=None, help="optional path to a .txt sample of L3Cube-HingCorpus")
    ap.add_argument("--hingcorpus-n", type=int, default=200)
    args = ap.parse_args()
    df = load_dataset(include_hinglish=False)
    src = df[df["english"].fillna("").str.len() > 5].drop_duplicates("group")
    parts = []
    for sp, g in src.groupby("split"):
        k = max(1, round(args.n * len(g) / len(src)))
        parts.append(g.sample(min(k, len(g)), random_state=SEED))
    sample = pd.concat(parts).reset_index(drop=True)
    client = LLMClient(args.provider)
    print(f"Generating {len(sample)} Hinglish reviews with {client.name} ...")
    hing = to_hinglish(client, sample["english"].tolist(), args.batch_size)
    out = sample[["group", "split", "english", "label", "aspects", "aspect_combo", "category",
                  "generic_category", "product", "brand"]].copy()
    out["text"] = [normalize_text(h) if h else "" for h in hing]
    out = out[out["text"].str.len() >= 3].reset_index(drop=True)
    if out.empty:
        raise SystemExit("The provider produced no usable Hinglish reviews. Check the provider and retry.")
    print("Labelling Hinglish reviews with an independent LLM call ...")
    llm = classify_texts(client, out["text"].tolist(), ["hinglish"] * len(out), args.batch_size)
    out["label_gold"] = out["label"]
    out["label_llm"] = [r["label"] or "" for r in llm]
    valid = out["label_llm"].isin(list(LABEL2ID))
    if not valid.any():
        raise SystemExit("The provider returned no usable Hinglish sentiment labels. Check the provider and retry.")
    kappa = float(cohen_kappa_score(out.loc[valid, "label_gold"], out.loc[valid, "label_llm"])) if valid.sum() > 1 else None
    agree = float((out.loc[valid, "label_gold"] == out.loc[valid, "label_llm"]).mean()) if valid.any() else None
    # LLM labels for train/val (weak supervision, as in the PDF); gold labels for test (clean evaluation)
    out["label"] = [lg if s == "test" else (ll if ll in LABEL2ID else lg)
                    for s, lg, ll in zip(out["split"], out["label_gold"], out["label_llm"])]
    if args.hingcorpus:
        with open(args.hingcorpus, encoding="utf-8") as corpus:
            lines = pd.Series((normalize_text(line) for line in corpus if len(line.strip()) > 10),
                              dtype="str").drop_duplicates()
        hc = lines.sample(min(args.hingcorpus_n, len(lines)), random_state=SEED).tolist()
        lab = classify_texts(client, hc, ["hinglish"] * len(hc), args.batch_size)
        extra = pd.DataFrame({"text": hc, "label": [r["label"] for r in lab]}).dropna()
        extra = extra.assign(split="train", group=[f"hc-{i}" for i in range(len(extra))], english="",
                             label_gold="", label_llm=extra["label"])
        out = pd.concat([out, extra], ignore_index=True)
    out["lang"] = "hinglish"
    out["y"] = out["label"].map(LABEL2ID).astype(int)
    out["id"] = [f"hinglish-{i}" for i in range(len(out))]
    cols = ["id", "lang", "text", "english", "label", "y", "split", "group", "aspects", "aspect_combo",
            "category", "generic_category", "product", "brand", "label_gold", "label_llm"]
    out.reindex(columns=cols).to_csv(DATA_PROC / "hinglish.csv", index=False, encoding="utf-8")

    q = {"llm": client.name, "n_generated": int(len(out)), "llm_label_valid_rate": float(valid.mean()),
         "agreement_with_gold": agree, "cohen_kappa": kappa,
         "split_counts": out["split"].value_counts().to_dict()}
    save_json(q, RESULTS_DIR / "hinglish_label_quality.json")
    print(q)
    print("DONE: data/processed/hinglish.csv written")
if __name__ == "__main__":
    main()
