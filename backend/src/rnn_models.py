"""Vanilla RNN / LSTM / BiLSTM text classifiers (PyTorch)."""
import torch
import torch.nn as nn
def pad_batch(seqs):
    lens = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    ids = torch.zeros(len(seqs), int(lens.max()), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, lens
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, emb_dim=128, hidden=128, num_layers=1, cell="lstm",
                 bidirectional=False, dropout=0.3, num_classes=2, pooling="maxmean"):
        super().__init__()

        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        kw = dict(input_size=emb_dim, hidden_size=hidden, num_layers=num_layers, batch_first=True,
                  bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        if cell == "rnn":
            self.rnn = nn.RNN(nonlinearity="tanh", **kw)
        elif cell == "lstm":
            self.rnn = nn.LSTM(**kw)
        else:
            raise ValueError(f"unknown cell {cell}")
        out_dim = hidden * (2 if bidirectional else 1)
        self.pooling = pooling
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim * 2 if pooling == "maxmean" else out_dim, num_classes)
    def forward(self, ids, lengths):
        x = self.drop(self.emb(ids))
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu().clamp(min=1), batch_first=True,
                                                   enforce_sorted=False)
        out, h = self.rnn(packed)
        if self.pooling == "last":
            if isinstance(h, tuple):  # LSTM -> (h_n, c_n)
                h = h[0]
            feat = torch.cat([h[-2], h[-1]], dim=1) if self.rnn.bidirectional else h[-1]
        else:
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=ids.size(1))
            lens = lengths.to(ids.device)
            mask = (torch.arange(ids.size(1), device=ids.device)[None, :] < lens[:, None]).unsqueeze(-1).float()
            mean = (out * mask).sum(1) / mask.sum(1).clamp(min=1)
            mx = out.masked_fill(mask == 0, -1e9).max(1).values
            feat = torch.cat([mean, mx], dim=1)
        return self.fc(self.drop(feat))
