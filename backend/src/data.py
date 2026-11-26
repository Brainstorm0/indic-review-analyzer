"""Dataset loading helpers (processed CSVs)."""
import pandas as pd
from .config import DATA_PROC
def load_dataset(langs=None, include_hinglish: bool = True) -> pd.DataFrame:
    path = DATA_PROC / "all.csv"
    if not path.exists():
        raise FileNotFoundError("data/processed/all.csv missing. Run: python -m scripts.prepare_data")
    df = pd.read_csv(path)
    h = DATA_PROC / "hinglish.csv"
    if include_hinglish and h.exists():
        df = pd.concat([df, pd.read_csv(h)], ignore_index=True)
    if langs:
        df = df[df["lang"].isin(langs)]
    df["text"] = df["text"].astype(str)
    df["y"] = df["y"].astype(int)
    return df.reset_index(drop=True)
def split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    return df[df["split"] == name].reset_index(drop=True)
def parse_langs(arg):
    if not arg or arg == "all":
        return None
    return [x.strip() for x in arg.split(",") if x.strip()]
