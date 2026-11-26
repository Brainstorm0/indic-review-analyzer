"""Classical models on TF-IDF (char + word n-grams) features.
Run: python -m scripts.train_tfidf                 # Logistic Regression (Unit II)
     python -m scripts.train_tfidf --clf nb        # Multinomial Naive Bayes (Unit III)
     python -m scripts.train_tfidf --clf bnb       # Bernoulli Naive Bayes, presence/absence (Unit III)
"""
import argparse
import time
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from src.config import MODELS_DIR, SEED
from src.data import load_dataset, parse_langs, split
from src.evaluation import measure_latency, write_eval
from src.text import word_tokenize
from src.utils import save_json, set_seed

# clf -> (default model id, display name, hyperparameter name, grid, factory)
CLASSIFIERS = {
    "lr": ("tfidf_lr", "TF-IDF + Logistic Regression", "C", (0.25, 0.5, 1, 2, 4, 8),
           lambda v: LogisticRegression(C=v, max_iter=3000, class_weight="balanced", random_state=SEED)),
    "nb": ("tfidf_nb", "TF-IDF + Multinomial Naive Bayes", "alpha", (0.01, 0.03, 0.1, 0.3, 1.0),
           lambda v: MultinomialNB(alpha=v)),
    # binarize=0.0 turns every non-zero TF-IDF value into 1: each n-gram is simply present or absent.
    "bnb": ("tfidf_bnb", "TF-IDF n-grams + Bernoulli Naive Bayes (presence/absence)", "alpha", (0.01, 0.03, 0.1, 0.3, 1.0),
            lambda v: BernoulliNB(alpha=v, binarize=0.0)),
}


def build(clf, value):
    feats = FeatureUnion([
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, max_features=200_000)),
        ("word", TfidfVectorizer(tokenizer=word_tokenize, token_pattern=None, lowercase=False,
                                 ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", CLASSIFIERS[clf][4](value))])


def count_params(model) -> int:
    """Learned numbers in the fitted classifier (weights or log-probabilities)."""
    if hasattr(model, "coef_"):
        return int(model.coef_.size)
    if hasattr(model, "feature_log_prob_"):
        return int(model.feature_log_prob_.size)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clf", choices=list(CLASSIFIERS), default="lr")
    ap.add_argument("--langs", default="all")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()
    set_seed(SEED)
    model_id_default, display, hp, grid, _ = CLASSIFIERS[args.clf]
    model_id = args.model_id or model_id_default
    langs = parse_langs(args.langs)
    df = load_dataset(langs)
    tr, va, te = split(df, "train"), split(df, "val"), split(df, "test")
    print(f"[{model_id}] train={len(tr)} val={len(va)} test={len(te)}")
    t0 = time.time()
    best_v, best_f1 = None, -1.0
    for v in grid:
        pipe = build(args.clf, v).fit(tr["text"], tr["y"])
        f1 = f1_score(va["y"], pipe.predict(va["text"]), average="macro")
        print(f"  {hp}={v!s:<6} val macro-F1={f1:.4f}")
        if f1 > best_f1:
            best_v, best_f1 = v, f1
    pipe = build(args.clf, best_v).fit(tr["text"], tr["y"])
    train_time = time.time() - t0
    out = MODELS_DIR / model_id
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out / "model.joblib")
    save_json({"type": "tfidf", "model_id": model_id, "clf": args.clf, "langs": langs, hp: best_v,
               "display": display}, out / "meta.json")
    lat = measure_latency(lambda xs: pipe.predict_proba(xs), te["text"].tolist())
    write_eval(model_id, te, pipe.predict_proba(te["text"]),
               extra={"family": "classical", "n_params": count_params(pipe.named_steps["clf"]),
                      "train_time_s": train_time, "latency_ms": lat, "best_val_macro_f1": best_f1,
                      "hparams": {hp: best_v}})


if __name__ == "__main__":
    main()
