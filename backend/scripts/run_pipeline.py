"""Runs the whole experiment end to end (TF-IDF classifiers, BiLSTM, MuRIL + LLM).
  python -m scripts.run_pipeline            # full run
  python -m scripts.run_pipeline --quick    # fast smoke run (fewer epochs)
  python -m scripts.run_pipeline --skip-llm # no LLM steps
  python -m scripts.run_pipeline --only bilstm,evaluate
"""
import argparse
import os
import subprocess
import sys
import time

from src.config import RESULTS_DIR
from src.utils import save_json
def steps(quick, local_subwords=False):
    er = ["--epochs", "5"] if quick else []
    et = ["--epochs", "1", "--max-train", "300"] if quick else ["--epochs", "4", "--freeze-embeddings"]
    subword = ["--tokenizer", "subword-local"] if local_subwords else []
    return [
        ("download", ["scripts.download_data"], False),
        ("prepare", ["scripts.prepare_data"], False),
        ("hinglish", ["scripts.make_hinglish", "--n", "500"], True),
        ("tfidf", ["scripts.train_tfidf"], False),
        ("nb", ["scripts.train_tfidf", "--clf", "nb"], False),
        ("bnb", ["scripts.train_tfidf", "--clf", "bnb"], False),
        ("bilstm", ["scripts.train_rnn", "--cell", "lstm", "--bidirectional", "--model-id", "bilstm_subword"] + subword + er, False),
        ("muril_multi", ["scripts.train_transformer", "--model-name", "google/muril-base-cased", "--model-id", "muril_multi", "--display", "MuRIL multilingual (all languages + Hinglish)", "--max-len", "128"] + et, False),
        ("llm_zs", ["scripts.eval_llm", "--shots", "0"], True),
        ("llm_fs", ["scripts.eval_llm", "--shots", "3"], True),
        ("aspects", ["scripts.eval_aspects", "--n", "30"], True),
        ("evaluate", ["scripts.evaluate_all"], False),
        ("expected", ["scripts.compare_expected"], False),
    ]
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-llm", action="store_true")
    ap.add_argument("--local-subwords", action="store_true", help="fit the recurrent tokenizer on training text")
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
    log = []
    for name, cmd, is_llm in steps(args.quick, args.local_subwords):
        if only and name not in only:
            continue
        if is_llm and args.skip_llm:
            print(f"--- skip {name} (LLM)")
            continue
        print(f"\n===== STEP {name}: python -m {' '.join(cmd)}")
        t = time.time()
        rc = subprocess.run([sys.executable, "-u", "-m", *cmd], env=env).returncode
        log.append({"step": name, "returncode": rc, "seconds": round(time.time() - t, 1)})

        if rc != 0:
            if is_llm:
                print(f"!!! LLM step {name} failed (non-fatal) - see Phase 3. Continuing.")
            else:
                save_json(log, RESULTS_DIR / "pipeline_log.json")
                raise SystemExit(f"Step {name} failed - fix it (Phase 3) and re-run with --only {name},...")
    save_json(log, RESULTS_DIR / "pipeline_log.json")
    print("\nPIPELINE FINISHED. Summary in artifacts/results/summary.md and expected_vs_actual.md")
if __name__ == "__main__":
    main()
