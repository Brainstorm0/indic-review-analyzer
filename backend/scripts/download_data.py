"""Downloads IndicSentiment raw JSON-lines + pretrained models. Run: python -m scripts.download_data"""
import argparse
from huggingface_hub import hf_hub_download, snapshot_download
from src.config import DATA_RAW, HF_DATASET, LANGS, MURIL
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default=",".join(LANGS))
    ap.add_argument("--skip-models", action="store_true")
    args = ap.parse_args()
    for lang in [x.strip() for x in args.langs.split(",") if x.strip()]:
        for split in ("validation", "test"):
            p = hf_hub_download(repo_id=HF_DATASET, repo_type="dataset",
                                filename=f"data/{split}/{lang}.json",
                                local_dir=str(DATA_RAW / "indicsentiment"))
            print(f"  downloaded {lang}/{split} -> {p}")
    if not args.skip_models:
        for m in (MURIL,):
            path = snapshot_download(m, allow_patterns=["*.json", "*.model", "*.txt", "*.bin", "*.safetensors"])
            print(f"  cached model {m} -> {path}")
    print("DONE: data + models downloaded")
if __name__ == "__main__":
    main()
