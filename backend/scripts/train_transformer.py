"""Fine-tune MuRIL / IndicBERT / MarathiBERT (custom PyTorch loop, early stopping, AMP on CUDA).
Examples:
  python -m scripts.train_transformer --model-id indicbert_multi
  python -m scripts.train_transformer --langs hi --model-id indicbert_hi
  python -m scripts.train_transformer --model-name l3cube-pune/marathi-bert-v2 --langs mr --model-id marathibert_mr
"""
import argparse
import contextlib
import math
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from src.config import INDICBERT, MODELS_DIR, RESULTS_DIR, SEED
from src.data import load_dataset, parse_langs, split
from src.evaluation import write_eval
from src.utils import get_device, save_json, set_seed
def load_tokenizer(name, slow):
    if slow:  # ALBERT/IndicBERT: keep_accents=True keeps Indic vowel signs
        return AutoTokenizer.from_pretrained(name, keep_accents=True, use_fast=False)
    return AutoTokenizer.from_pretrained(name)
@torch.no_grad()
def predict(model, dl, device):
    model.eval()
    out = []
    for b in dl:
        b.pop("labels", None)
        b = {k: v.to(device) for k, v in b.items()}
        out.append(torch.softmax(model(**b).logits.float(), -1).cpu().numpy())
    return np.concatenate(out)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-name", default=INDICBERT)
    ap.add_argument("--model-id", default="indicbert_multi")
    ap.add_argument("--display", default=None)
    ap.add_argument("--langs", default="all")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)

    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--patience", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=0, help="subsample train set (quick mode); 0 = all")
    ap.add_argument("--freeze-encoder", action="store_true", help="ablation: linear probe")
    ap.add_argument("--freeze-embeddings", action="store_true",
                    help="freeze the input embedding table for lower-memory fine-tuning")
    args = ap.parse_args()
    set_seed(SEED)
    device = get_device()
    langs = parse_langs(args.langs)
    df = load_dataset(langs)
    tr, va, te = split(df, "train"), split(df, "val"), split(df, "test")
    if args.max_train and len(tr) > args.max_train:
        tr = tr.sample(args.max_train, random_state=SEED).reset_index(drop=True)
    print(f"[{args.model_id}] {args.model_name} on langs={langs or 'all'} train={len(tr)} val={len(va)} test={len(te)} device={device}")
    slow = "indic-bert" in args.model_name
    tok = load_tokenizer(args.model_name, slow)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, id2label={0: "negative", 1: "positive"}, label2id={"negative": 0, "positive": 1})
    if args.freeze_encoder:
        for p in model.base_model.parameters():
            p.requires_grad = False
    elif args.freeze_embeddings:
        model.get_input_embeddings().requires_grad_(False)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  parameters total={n_params:,} trainable={n_train:,}")
    def loader(d, shuffle, bs):
        data = list(zip(d["text"].tolist(), d["y"].astype(int).tolist()))
        def collate(b):
            enc = tok([t for t, _ in b], padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
            enc["labels"] = torch.tensor([y for _, y in b], dtype=torch.long)
            return enc
        return DataLoader(data, batch_size=bs, shuffle=shuffle, collate_fn=collate)
    tl, vl, tel = loader(tr, True, args.batch_size), loader(va, False, 64), loader(te, False, 64)
    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr["y"].values)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float).to(device))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = math.ceil(len(tl) / args.grad_accum) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(args.warmup * total_steps), total_steps)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    history, best_f1, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        opt.zero_grad()
        run_loss = 0.0
        for step, batch in enumerate(tqdm(tl, desc=f"epoch {ep}", leave=False)):

            y = batch.pop("labels").to(device)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else contextlib.nullcontext()
            with ctx:
                logits = model(**batch).logits
            loss = crit(logits.float(), y) / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            run_loss += loss.item() * args.grad_accum
            if (step + 1) % args.grad_accum == 0 or step + 1 == len(tl):
                if use_amp:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, 1.0)
                if use_amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad()
                sched.step()
        vp = predict(model, vl, device)
        vf1 = f1_score(va["y"], vp.argmax(1), average="macro")
        vacc = accuracy_score(va["y"], vp.argmax(1))
        history.append({"epoch": ep, "train_loss": run_loss / len(tl), "val_macro_f1": vf1, "val_accuracy": vacc})
        print(f"  epoch {ep} loss={run_loss / len(tl):.4f} val_acc={vacc:.4f} val_macroF1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, bad = vf1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    out = MODELS_DIR / args.model_id
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    display = args.display or ("IndicBERT multilingual fine-tune" if slow else args.model_name)
    save_json({"type": "transformer", "model_id": args.model_id, "base_model": args.model_name,
               "display": display, "slow_tokenizer": slow, "max_len": args.max_len, "langs": langs,
               "frozen_encoder": args.freeze_encoder, "frozen_embeddings": args.freeze_embeddings},
              out / "meta.json")
    save_json(history, RESULTS_DIR / f"history_{args.model_id}.json")
    tp = predict(model, tel, device)
    model.eval()
    sample = te["text"].tolist()[:30]
    t = time.perf_counter()
    with torch.no_grad():
        for s in sample:
            enc = tok([s], truncation=True, max_length=args.max_len, return_tensors="pt").to(device)
            model(**enc)
    lat = (time.perf_counter() - t) / max(1, len(sample)) * 1000
    write_eval(args.model_id, te, tp, extra={

        "family": "transformer", "n_params": n_params, "n_trainable": n_train, "train_time_s": train_time,
        "latency_ms": lat, "best_val_macro_f1": best_f1, "epochs_run": len(history),
        "hparams": {k: v for k, v in vars(args).items()}})
if __name__ == "__main__":
    main()
