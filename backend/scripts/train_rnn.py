"""Vanilla RNN / LSTM / BiLSTM training with early stopping.
Examples:
  python -m scripts.train_rnn --cell rnn
  python -m scripts.train_rnn --cell lstm
  python -m scripts.train_rnn --cell lstm --bidirectional
"""
import argparse
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from src.config import MODELS_DIR, RESULTS_DIR, SEED
from src.data import load_dataset, parse_langs, split
from src.evaluation import write_eval
from src.rnn_models import RNNClassifier, pad_batch
from src.text import TOKENIZERS, Vocab
from src.utils import save_json, set_seed
DISPLAY = {("rnn", False): "Vanilla RNN", ("lstm", False): "LSTM", ("lstm", True): "BiLSTM", ("rnn", True): "Bi-RNN"}

def make_loader(token_lists, ys, vocab, max_len, bs, shuffle):
    data = [(vocab.encode(t, max_len), int(y)) for t, y in zip(token_lists, ys)]
    def collate(batch):
        ids, lens = pad_batch([x for x, _ in batch])
        return ids, lens, torch.tensor([y for _, y in batch], dtype=torch.long)
    return DataLoader(data, batch_size=bs, shuffle=shuffle, collate_fn=collate)
@torch.no_grad()
def predict_probs(model, loader, device):
    model.eval()
    out = [torch.softmax(model(ids.to(device), lens), -1).cpu().numpy() for ids, lens, _ in loader]
    return np.concatenate(out)
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", choices=["rnn", "lstm"], default="lstm")
    ap.add_argument("--bidirectional", action="store_true")
    ap.add_argument("--tokenizer", choices=[*TOKENIZERS, "subword-local"], default="subword-local",
                    help="subword-local fits SentencePiece on the training partition")
    ap.add_argument("--pooling", choices=["maxmean", "last"], default="maxmean")
    ap.add_argument("--emb", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--max-len", type=int, default=None)
    ap.add_argument("--min-freq", type=int, default=2)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--langs", default="all")
    ap.add_argument("--model-id", default=None)
    args = ap.parse_args()
    set_seed(SEED)
    max_len = args.max_len or {"subword": 128, "subword-local": 128, "word": 100, "char": 400}[args.tokenizer]
    model_id = args.model_id or f"{'bi' if args.bidirectional else ''}{args.cell}_{args.tokenizer}"
    display = f"{DISPLAY[(args.cell, args.bidirectional)]} ({args.tokenizer} tokens, {args.pooling} pooling, {args.layers} layer)"
    langs = parse_langs(args.langs)
    df = load_dataset(langs)
    tr, va, te = split(df, "train"), split(df, "val"), split(df, "test")
    tokenizer_model = None
    if args.tokenizer == "subword-local":
        import io
        import sentencepiece as spm

        buffer = io.BytesIO()
        spm.SentencePieceTrainer.train(
            sentence_iterator=iter(tr["text"]), model_writer=buffer, vocab_size=8000,
            hard_vocab_limit=False, normalization_rule_name="identity",
            shuffle_input_sentence=False, num_threads=1, minloglevel=2,
        )
        tokenizer_model = buffer.getvalue()
        processor = spm.SentencePieceProcessor(model_proto=tokenizer_model)
        tok = processor.EncodeAsPieces
    else:
        tok = TOKENIZERS[args.tokenizer]
    print(f"[{model_id}] tokenising {len(tr)}/{len(va)}/{len(te)} reviews with '{args.tokenizer}' ...")
    tr_t, va_t, te_t = [tok(x) for x in tr["text"]], [tok(x) for x in va["text"]], [tok(x) for x in te["text"]]
    vocab = Vocab.build(tr_t, min_freq=args.min_freq)
    print(f"  vocab size = {len(vocab)}")
    tl = make_loader(tr_t, tr["y"], vocab, max_len, args.batch_size, True)
    vl = make_loader(va_t, va["y"], vocab, max_len, 128, False)
    tel = make_loader(te_t, te["y"], vocab, max_len, 128, False)
    device = "cuda" if torch.cuda.is_available() else "cpu"  # packed RNNs are unreliable on MPS
    model = RNNClassifier(len(vocab), args.emb, args.hidden, args.layers, args.cell, args.bidirectional,
                          args.dropout, pooling=args.pooling).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters = {n_params:,}  device = {device}")

    cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=tr["y"].values)
    crit = nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float, device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
    history, best_f1, best_state, bad = [], -1.0, None, 0
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for ids, lens, y in tl:
            ids, y = ids.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(ids, lens), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(y)
            n += len(y)
        vp = predict_probs(model, vl, device)
        vf1 = f1_score(va["y"], vp.argmax(1), average="macro")
        vacc = accuracy_score(va["y"], vp.argmax(1))
        sched.step(vf1)
        history.append({"epoch": ep, "train_loss": tot / n, "val_macro_f1": vf1, "val_accuracy": vacc})
        print(f"  epoch {ep:02d} loss={tot / n:.4f} val_acc={vacc:.4f} val_macroF1={vf1:.4f}")
        if vf1 > best_f1:
            best_f1, best_state, bad = vf1, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.patience:
                print("  early stopping")
                break
    train_time = time.time() - t0
    model.load_state_dict(best_state)
    out = MODELS_DIR / model_id
    out.mkdir(parents=True, exist_ok=True)
    if tokenizer_model is not None:
        (out / "tokenizer.model").write_bytes(tokenizer_model)
    torch.save({k: v.cpu() for k, v in model.state_dict().items()}, out / "model.pt")
    meta = {"type": "rnn", "model_id": model_id, "display": display, "cell": args.cell,
            "bidirectional": args.bidirectional, "tokenizer": args.tokenizer, "pooling": args.pooling,
            "emb": args.emb, "hidden": args.hidden, "layers": args.layers, "dropout": args.dropout,
            "max_len": max_len, "langs": langs, "vocab_size": len(vocab)}
    save_json(meta, out / "meta.json")
    save_json(vocab.itos, out / "vocab.json")
    save_json(history, RESULTS_DIR / f"history_{model_id}.json")
    tp = predict_probs(model, tel, device)
    model.eval()
    sample = te_t[:50]
    t = time.perf_counter()
    with torch.no_grad():
        for s in sample:
            ids, lens = pad_batch([vocab.encode(s, max_len)])
            model(ids.to(device), lens)
    lat = (time.perf_counter() - t) / max(1, len(sample)) * 1000
    write_eval(model_id, te, tp, extra={"family": "rnn", "n_params": n_params, "train_time_s": train_time,
                                        "latency_ms": lat, "best_val_macro_f1": best_f1,
                                        "epochs_run": len(history), "hparams": meta})

if __name__ == "__main__":
    main()
