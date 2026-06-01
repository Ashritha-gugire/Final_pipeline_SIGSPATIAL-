"""
models/lmtad_scorer.py
=======================
LM-TAD — Language Model for Trajectory Anomaly Detection.
Adapted from: Mbuya et al., ACM SIGSPATIAL 2024.

A causal GPT-like Transformer trained on normal agent stay sequences.
Anomaly score = mean negative log-likelihood (perplexity) of location tokens.
Higher perplexity = more anomalous.

NUMOSIM Adaptation:
  - Tokens     : geo_bin IDs (POI identifiers from stay sequences)
  - Vocabulary : top-N most frequent geo_bins from training + special tokens
  - Novel geo_bins (unseen or rare) → <NOVEL> token
  - This aligns naturally with Type 2 anomalies (inserted novel locations)
    which produce high perplexity under a model trained on normal routines.

Special tokens:
  <PAD>   = 0   padding to max_len
  <UNK>   = 1   geo_bins present in train but below frequency threshold
  <NOVEL> = 2   geo_bins never seen in training (novel locations)
  <BOS>   = 3   beginning of sequence
"""

import logging
import pickle
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from models.anomaly_scorer import AnomalyScorer

log = logging.getLogger(__name__)

# Special token IDs
PAD_ID   = 0
UNK_ID   = 1
NOVEL_ID = 2
BOS_ID   = 3
N_SPECIAL = 4   # number of special tokens


# ─────────────────────────────────────────────────────────────────────────────
# Sequence construction
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(
    train_stays: pd.DataFrame,
    max_vocab:   int = 5_000,
) -> Dict[int, int]:
    """
    Build geo_bin → token_id vocabulary from training stays.
    Top-max_vocab geo_bins by frequency get unique IDs.
    Everything else maps to UNK_ID.

    Returns:
        vocab: dict mapping geo_bin → token_id (>= N_SPECIAL)
    """
    counts = train_stays["geo_bin"].value_counts()
    top_bins = counts.head(max_vocab).index.tolist()
    vocab = {geo_bin: idx + N_SPECIAL
             for idx, geo_bin in enumerate(top_bins)}
    log.info(
        f"LM-TAD vocab: {len(vocab):,} geo_bins  "
        f"(coverage={counts.head(max_vocab).sum() / len(train_stays):.1%})"
    )
    return vocab


def build_lmtad_sequences(
    stay_df:    pd.DataFrame,
    vocab:      Dict[int, int],
    train_bins: set,
    max_len:    int = 60,
) -> Tuple[List[np.ndarray], List]:
    """
    Convert stay DataFrame into per-agent token sequences.

    Tokenisation:
      - geo_bin in vocab          → vocab[geo_bin]
      - geo_bin in train but rare → UNK_ID
      - geo_bin never in train    → NOVEL_ID  (key anomaly signal)

    Returns:
        sequences : list of 1-D int arrays, one per agent
        agent_ids : list of agent IDs in same order
    """
    sort_cols = ["agent", "timestamp"] if "timestamp" in stay_df.columns \
        else ["agent"]
    df = stay_df.sort_values(sort_cols).copy()

    sequences, agent_ids = [], []

    for agent, grp in df.groupby("agent"):
        tokens = []
        for gb in grp["geo_bin"].values:
            if gb in vocab:
                tokens.append(vocab[gb])
            elif gb in train_bins:
                tokens.append(UNK_ID)
            else:
                tokens.append(NOVEL_ID)   # novel location

        # Prepend BOS, truncate to max_len
        tokens = [BOS_ID] + tokens
        tokens = tokens[-max_len:]
        sequences.append(np.array(tokens, dtype=np.int64))
        agent_ids.append(agent)

    return sequences, agent_ids


def _pad_token_sequences(
    sequences: List[np.ndarray],
    max_len:   int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Pad sequences to max_len. Returns (tokens, attention_mask).
    tokens         : (N, max_len) int64
    attention_mask : (N, max_len) bool  True = real token
    """
    N = len(sequences)
    tokens = np.full((N, max_len), PAD_ID, dtype=np.int64)
    mask   = np.zeros((N, max_len), dtype=bool)
    for i, seq in enumerate(sequences):
        L = min(len(seq), max_len)
        tokens[i, :L] = seq[:L]
        mask[i,   :L] = True
    return tokens, mask


# ─────────────────────────────────────────────────────────────────────────────
# Causal Transformer (GPT-like)
# ─────────────────────────────────────────────────────────────────────────────

def _build_transformer(vocab_size: int, embed_dim: int,
                       n_heads: int, n_layers: int, max_len: int,
                       dropout: float = 0.1):
    import torch
    import torch.nn as nn

    class CausalTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_emb = nn.Embedding(vocab_size, embed_dim,
                                          padding_idx=PAD_ID)
            self.pos_emb   = nn.Embedding(max_len, embed_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, nhead=n_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout, batch_first=True,
            )
            self.transformer = nn.TransformerEncoder(enc_layer,
                                                     num_layers=n_layers)
            self.head    = nn.Linear(embed_dim, vocab_size)
            self.max_len = max_len

        def _causal_mask(self, L, device):
            """Upper-triangular mask — each position only attends to past."""
            mask = torch.triu(torch.ones(L, L, device=device), diagonal=1)
            return mask.bool()

        def forward(self, x, pad_mask=None):
            # x: (B, L)
            B, L  = x.shape
            device = x.device
            pos   = torch.arange(L, device=device).unsqueeze(0)  # (1, L)
            emb   = self.token_emb(x) + self.pos_emb(pos)        # (B, L, D)
            cmask = self._causal_mask(L, device)                  # (L, L)
            # pad_mask: True where padding — invert for transformer
            src_key_padding_mask = ~pad_mask if pad_mask is not None else None
            out   = self.transformer(emb,
                                     mask=cmask,
                                     src_key_padding_mask=src_key_padding_mask)
            return self.head(out)   # (B, L, vocab_size)

    return CausalTransformer()


# ─────────────────────────────────────────────────────────────────────────────
# Main scorer
# ─────────────────────────────────────────────────────────────────────────────

class LMTADScorer(AnomalyScorer):
    """
    LM-TAD anomaly scorer for agent stay sequences.

    Trains a causal Transformer on normal agent geo_bin sequences.
    Anomaly score = mean per-token negative log-likelihood (perplexity).
    Novel locations get the <NOVEL> token → naturally high perplexity.

    Args:
        max_vocab:  Maximum vocabulary size (top-N geo_bins). Default 5000.
        embed_dim:  Token embedding dimension. Default 64.
        n_heads:    Attention heads. Default 4.
        n_layers:   Transformer layers. Default 2.
        max_len:    Max sequence length. Default 60.
        epochs:     Training epochs. Default 50.
        lr:         Learning rate. Default 1e-3.
        batch_size: Batch size. Default 128.
        dropout:    Dropout rate. Default 0.1.
        random_state: Seed. Default 42.
    """

    def __init__(
        self,
        max_vocab:    int   = 5_000,
        embed_dim:    int   = 64,
        n_heads:      int   = 4,
        n_layers:     int   = 2,
        max_len:      int   = 60,
        epochs:       int   = 50,
        lr:           float = 1e-3,
        batch_size:   int   = 128,
        dropout:      float = 0.1,
        random_state: int   = 42,
    ):
        self.max_vocab    = max_vocab
        self.embed_dim    = embed_dim
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.max_len      = max_len
        self.epochs       = epochs
        self.lr           = lr
        self.batch_size   = batch_size
        self.dropout      = dropout
        self.random_state = random_state

        self._model      = None
        self._vocab      = None       # geo_bin → token_id
        self._train_bins = None       # set of all geo_bins seen in training
        self._vocab_size = None
        self._device     = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(
        self,
        sequences:  List[np.ndarray],
        vocab:      Dict[int, int],
        train_bins: set,
    ) -> None:
        """
        Train on normal agent token sequences.

        Args:
            sequences:  List of 1-D int64 arrays (from build_lmtad_sequences)
            vocab:      geo_bin → token_id mapping
            train_bins: set of all geo_bins seen in training stays
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(self.random_state)
        self._vocab      = vocab
        self._train_bins = train_bins
        self._vocab_size = self.max_vocab + N_SPECIAL
        self._device     = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        log.info(
            f"Training LM-TAD on {len(sequences):,} agents, "
            f"vocab_size={self._vocab_size}, "
            f"embed_dim={self.embed_dim}, "
            f"n_layers={self.n_layers}, "
            f"epochs={self.epochs}, "
            f"device={self._device}"
        )

        tokens, mask = _pad_token_sequences(sequences, self.max_len)
        t_tokens = torch.tensor(tokens, dtype=torch.long)
        t_mask   = torch.tensor(mask,   dtype=torch.bool)

        loader = DataLoader(
            TensorDataset(t_tokens, t_mask),
            batch_size=self.batch_size,
            shuffle=True,
        )

        model  = _build_transformer(
            self._vocab_size, self.embed_dim,
            self.n_heads, self.n_layers, self.max_len, self.dropout,
        ).to(self._device)

        opt     = torch.optim.Adam(model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_ID)

        model.train()
        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for (x, m) in loader:
                x = x.to(self._device)
                m = m.to(self._device)

                # Causal LM: predict x[t] from x[0..t-1]
                inp  = x[:, :-1]   # input  tokens (drop last)
                tgt  = x[:, 1:]    # target tokens (drop first)
                m_in = m[:, :-1]

                logits = model(inp, pad_mask=m_in)
                # logits: (B, L-1, vocab_size) → flatten for loss
                loss = loss_fn(
                    logits.reshape(-1, self._vocab_size),
                    tgt.reshape(-1),
                )
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                epoch_loss += loss.item()

            if epoch % 10 == 0 or epoch == 1:
                log.info(
                    f"  Epoch {epoch:>3}/{self.epochs}  "
                    f"loss={epoch_loss / len(loader):.4f}"
                )

        self._model = model
        log.info("LM-TAD training complete.")

    # ── score ─────────────────────────────────────────────────────────────────

    def score(self, sequences: List[np.ndarray]) -> np.ndarray:
        """
        Compute per-agent mean NLL (perplexity proxy).
        Higher = more anomalous.
        """
        if self._model is None:
            raise RuntimeError("LMTADScorer not fitted. Call fit() first.")

        import torch
        import torch.nn.functional as F

        tokens, mask = _pad_token_sequences(sequences, self.max_len)
        t_tokens = torch.tensor(tokens, dtype=torch.long)
        t_mask   = torch.tensor(mask,   dtype=torch.bool)

        self._model.eval()
        scores = []

        with torch.no_grad():
            for i in range(0, len(t_tokens), self.batch_size):
                x = t_tokens[i: i + self.batch_size].to(self._device)
                m = t_mask[i:   i + self.batch_size].to(self._device)

                inp    = x[:, :-1]
                tgt    = x[:, 1:]
                m_tgt  = m[:, 1:]

                logits = self._model(inp, pad_mask=m[:, :-1])
                log_p  = F.log_softmax(logits, dim=-1)

                # Gather log-prob of the actual next token
                nll = -log_p.gather(
                    2, tgt.unsqueeze(2)
                ).squeeze(2)   # (B, L-1)

                # Mask padding, compute mean NLL per agent
                nll_masked = nll * m_tgt.float()
                n_tokens   = m_tgt.float().sum(dim=1).clamp(min=1)
                mean_nll   = (nll_masked.sum(dim=1) / n_tokens).cpu().numpy()
                scores.append(mean_nll)

        out = np.concatenate(scores)
        log.info(
            f"LM-TAD scored {len(sequences):,} agents. "
            f"Range: [{out.min():.4f}, {out.max():.4f}]"
        )
        return out

    # ── save / load ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        import torch
        d = Path(path)
        d.mkdir(parents=True, exist_ok=True)
        torch.save(self._model.state_dict(), d / "lmtad_state_dict.pt")
        cfg = dict(
            max_vocab=self.max_vocab, embed_dim=self.embed_dim,
            n_heads=self.n_heads, n_layers=self.n_layers,
            max_len=self.max_len, epochs=self.epochs,
            lr=self.lr, batch_size=self.batch_size,
            dropout=self.dropout, random_state=self.random_state,
            vocab_size=self._vocab_size,
        )
        with open(d / "lmtad_config.pkl", "wb") as fh:
            pickle.dump(cfg, fh)
        with open(d / "lmtad_vocab.pkl", "wb") as fh:
            pickle.dump({"vocab": self._vocab,
                         "train_bins": self._train_bins}, fh)
        log.info(f"LMTADScorer saved to {d}")

    def load(self, path: str) -> None:
        import torch
        d = Path(path)
        with open(d / "lmtad_config.pkl", "rb") as fh:
            cfg = pickle.load(fh)
        for k, v in cfg.items():
            if k != "vocab_size":
                setattr(self, k, v)
        self._vocab_size = cfg["vocab_size"]
        with open(d / "lmtad_vocab.pkl", "rb") as fh:
            vdata = pickle.load(fh)
        self._vocab      = vdata["vocab"]
        self._train_bins = vdata["train_bins"]
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._model = _build_transformer(
            self._vocab_size, self.embed_dim,
            self.n_heads, self.n_layers, self.max_len, self.dropout,
        ).to(self._device)
        self._model.load_state_dict(
            torch.load(d / "lmtad_state_dict.pt",
                       map_location=self._device)
        )
        log.info(f"LMTADScorer loaded from {d}")