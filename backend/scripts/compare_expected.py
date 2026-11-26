"""Phase 3: expected vs actual comparison. Run: python -m scripts.compare_expected"""
import fnmatch
import pandas as pd
from src.config import RESULTS_DIR
# Approximate engineering expectations for ~1.1k reviews/language (NOT published numbers).
EXPECTED = {
    "tfidf_lr": (0.70, 0.90),
    "tfidf_nb": (0.65, 0.90),
    "tfidf_bnb": (0.60, 0.88),
    "bilstm_subword": (0.62, 0.88),
    "muril_multi": (0.78, 0.96),
    "llm_ollama_*": (0.65, 0.95),
    "llm_*": (0.80, 0.98),
}
REASON_LOW = ("Check: training collapsed to one class (see confusion matrix), lr too high/low, "
              "keep_accents missing, too few epochs (quick mode), or LLM unparseable answers.")
REASON_HIGH = "Check for leakage (leakage_check_passed in data_stats.json) or a tiny test subset."
def main():

    lb = pd.read_csv(RESULTS_DIR / "leaderboard.csv")
    rows = []
    for r in lb.itertuples():
        rng = next((v for k, v in EXPECTED.items() if fnmatch.fnmatch(r.model_id, k)), None)
        if rng is None:
            continue
        lo, hi = rng
        f = float(r.macro_f1)
        status = "PASS" if lo <= f <= hi else ("BELOW" if f < lo else "ABOVE")
        rows.append({"model": r.model_id, "scope": r.scope, "expected_macro_f1": f"{lo:.2f}-{hi:.2f}",
                     "actual_macro_f1": f"{f:.4f}", "status": status,
                     "action": "" if status == "PASS" else (REASON_LOW if status == "BELOW" else REASON_HIGH)})
    get = {r.model_id: float(r.macro_f1) for r in lb.itertuples()}
    tol = 0.02
    checks = []
    def chk(name, cond):
        checks.append({"expected_behaviour": name, "observed": "YES" if cond else "NO"})
    if all(k in get for k in ("rnn_subword", "lstm_subword", "bilstm_subword")):
        chk("LSTM >= vanilla RNN (gating helps long reviews)", get["lstm_subword"] >= get["rnn_subword"] - tol)
        chk("BiLSTM >= LSTM (bidirectional context)", get["bilstm_subword"] >= get["lstm_subword"] - tol)
    rnn_best = max([v for k, v in get.items() if k.endswith("_subword")], default=None)
    if rnn_best is not None and "muril_multi" in get:
        chk("Pretrained MuRIL >= best RNN-family model", get["muril_multi"] >= rnn_best - tol)
    if "indicbert_multi" in get and "abl_indicbert_frozen" in get:
        chk("Full fine-tuning > frozen linear probe", get["indicbert_multi"] > get["abl_indicbert_frozen"])
    md = "# Expected vs actual\n\n| " + " | ".join(rows[0].keys()) + " |\n|" + "---|" * len(rows[0]) + "\n" if rows else "# Expected vs actual\n\n"
    for r in rows:
        md += "| " + " | ".join(str(v) for v in r.values()) + " |\n"
    md += "\n## Qualitative expectations\n| expected behaviour | observed |\n|---|---|\n"
    for c in checks:
        md += f"| {c['expected_behaviour']} | {c['observed']} |\n"
    (RESULTS_DIR / "expected_vs_actual.md").write_text(md, encoding="utf-8")
    print(md)
if __name__ == "__main__":
    main()
