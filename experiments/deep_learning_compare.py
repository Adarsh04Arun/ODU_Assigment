"""
Optional deep learning comparison: LSTM Autoencoder vs classical
models on the EPS numeric scoring layer.

Scope: EPS subsystem only (battery_voltage, solar_current, charge_rate).
This is a standalone, report-only experiment. It does not touch the fusion
layer, hard-limit layer, or operator UI — the production pipeline continues
to use the Isolation Forest chosen in Step 5.

Approach:
  - Train a single LSTM-Autoencoder on nominal-only EPS time-series passes
    (same training data used for IF/OC-SVM in Step 5.2).
  - Score test passes by reconstruction error, percentile-ranked against the
    training distribution so scores sit on the same [0,1] scale as classical.
  - Evaluate precision/recall/F1 for binary EPS anomaly detection across all
    three models (IF, OC-SVM, LSTM-AE) in one directly comparable table.

Expected result: the autoencoder is unlikely to outperform IF on a small
synthetic dataset. If it does not, this turns PDD Section 6.5's literature-
based argument into an empirically validated one using our own data.

Usage:
    python experiments/deep_learning_compare.py
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.numeric_scoring import load_models, score_pass

# ---------------------------------------------------------------------------
# EPS channels used as raw time-series input to the autoencoder.
# Three physically meaningful channels — voltage, solar current, charge rate.
# ---------------------------------------------------------------------------
EPS_CHANNELS = ["battery_voltage", "solar_current", "charge_rate"]
SEQ_LEN = 300       # matches READINGS_PER_PASS in synthetic_generator.py
N_FEATURES = len(EPS_CHANNELS)


# ===========================================================================
# Raw EPS sequence extraction
# ===========================================================================

def extract_eps_sequence(pass_data):
    """Return raw EPS time series as a (SEQ_LEN, N_FEATURES) float32 array."""
    return np.stack(
        [np.array(pass_data["eps"][ch], dtype=np.float32) for ch in EPS_CHANNELS],
        axis=1,
    )


# ===========================================================================
# Per-channel min-max normalisation (fit on nominal training data only)
# ===========================================================================

def fit_normalizer(sequences):
    """Return (ch_min, ch_max) arrays fitted on the concatenated training sequences."""
    all_data = np.concatenate(sequences, axis=0)   # (N * SEQ_LEN, N_FEATURES)
    return all_data.min(axis=0), all_data.max(axis=0)


def normalize(seq, ch_min, ch_max):
    """Min-max normalise a sequence to [0, 1] per channel."""
    denom = np.where((ch_max - ch_min) == 0, 1.0, ch_max - ch_min)
    return (seq - ch_min) / denom


# ===========================================================================
# LSTM Autoencoder: encoder → bottleneck → decoder
# ===========================================================================

class _Encoder(nn.Module):
    def __init__(self, n_features, hidden):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return h[-1]                       # (batch, hidden)


class _Decoder(nn.Module):
    def __init__(self, hidden, n_features, seq_len):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.proj = nn.Linear(hidden, n_features)

    def forward(self, z):
        z_seq = z.unsqueeze(1).expand(-1, self.seq_len, -1)   # broadcast across time
        out, _ = self.lstm(z_seq)
        return self.proj(out)              # (batch, seq_len, n_features)


class LSTMAutoencoder(nn.Module):
    """LSTM-Autoencoder for EPS anomaly detection via reconstruction error.

    Hidden size of 16 is intentionally small: the bottleneck forces the
    encoder to compress only the structure of a nominal pass, so anomalous
    passes reconstruct poorly and score high.
    """

    def __init__(self, n_features=N_FEATURES, hidden=16, seq_len=SEQ_LEN):
        super().__init__()
        self.encoder = _Encoder(n_features, hidden)
        self.decoder = _Decoder(hidden, n_features, seq_len)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ===========================================================================
# Training
# ===========================================================================

def train_autoencoder(nominal_sequences, epochs=50, batch_size=16, lr=1e-3, seed=42):
    """Train the LSTM-AE on nominal EPS sequences.

    Args:
        nominal_sequences: list of (SEQ_LEN, N_FEATURES) float32 arrays.

    Returns:
        model:               trained LSTMAutoencoder (eval mode)
        train_errors_sorted: sorted per-pass MSE array (for percentile scoring)
        ch_min, ch_max:      normalisation parameters
    """
    torch.manual_seed(seed)

    ch_min, ch_max = fit_normalizer(nominal_sequences)
    normed = np.stack(
        [normalize(s, ch_min, ch_max) for s in nominal_sequences], axis=0
    )                                                   # (N, SEQ_LEN, N_FEATURES)

    X = torch.tensor(normed, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = LSTMAutoencoder()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for (batch,) in loader:
            opt.zero_grad()
            loss = criterion(model(batch), batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(batch)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch:3d}/{epochs}  MSE={total_loss / len(normed):.6f}")

    # Compute per-pass reconstruction errors over the training set.
    # Sorted so percentile lookup is O(log N) via np.searchsorted.
    model.eval()
    with torch.no_grad():
        recon = model(X)
        train_errors = ((X - recon) ** 2).mean(dim=(1, 2)).numpy()

    return model, np.sort(train_errors), ch_min, ch_max


# ===========================================================================
# Scoring
# ===========================================================================

def score_autoencoder(pass_data, model, train_errors_sorted, ch_min, ch_max):
    """Anomaly score for one pass via percentile rank of reconstruction MSE.

    Percentile 0 (below all training errors) → score 0.0  (looks normal).
    Percentile 1 (above all training errors) → score 1.0  (anomalous).

    This is the same percentile-rank normalisation used in numeric_scoring.py
    for IF and OC-SVM, so the three scores sit on a comparable [0, 1] scale.
    """
    seq = extract_eps_sequence(pass_data)
    normed = normalize(seq, ch_min, ch_max)
    x = torch.tensor(normed[np.newaxis], dtype=torch.float32)   # (1, SEQ_LEN, N_FEAT)

    model.eval()
    with torch.no_grad():
        mse = ((x - model(x)) ** 2).mean().item()

    percentile = np.searchsorted(train_errors_sorted, mse) / len(train_errors_sorted)
    return float(np.clip(percentile, 0.0, 1.0))


# ===========================================================================
# Main: run training, score test set, print comparison table
# ===========================================================================

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "generated")

    with open(os.path.join(data_dir, "nominal_passes.json")) as f:
        nominal_passes = json.load(f)
    with open(os.path.join(data_dir, "labelled_passes.json")) as f:
        labelled_passes = json.load(f)
    with open(os.path.join(data_dir, "labels.json")) as f:
        labels = json.load(f)

    classical = load_models(os.path.join(data_dir, "trained_models.pkl"))

    print("=" * 65)
    print("Step 9.3 — Deep Learning Comparison (EPS Subsystem Only)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Train LSTM-AE on same nominal data used for IF / OC-SVM in Step 5
    # ------------------------------------------------------------------
    print(f"\nTraining LSTM-Autoencoder on {len(nominal_passes)} nominal EPS passes...")
    nominal_seqs = [extract_eps_sequence(p) for p in nominal_passes]
    ae_model, train_errors, ch_min, ch_max = train_autoencoder(nominal_seqs)
    print(f"Training complete. "
          f"Train MSE range: [{train_errors.min():.6f}, {train_errors.max():.6f}]\n")

    # ------------------------------------------------------------------
    # Score every test pass with all three models on EPS subsystem only.
    # Ground truth: pass is an EPS anomaly iff subsystem==eps and type!=none.
    # Binary prediction: score >= 0.5 → anomaly predicted.
    # ------------------------------------------------------------------
    THRESHOLD = 0.5

    y_true, y_if, y_oc, y_ae = [], [], [], []
    records = []

    for p, l in zip(labelled_passes, labels):
        is_eps_anom = (l["anomaly_type"] != "none" and l.get("subsystem") == "eps")
        y_true.append(int(is_eps_anom))

        if_score = score_pass(p, classical, "iforest")["eps"]["score"]
        oc_score = score_pass(p, classical, "ocsvm")["eps"]["score"]
        ae_score = score_autoencoder(p, ae_model, train_errors, ch_min, ch_max)

        y_if.append(int(if_score >= THRESHOLD))
        y_oc.append(int(oc_score >= THRESHOLD))
        y_ae.append(int(ae_score >= THRESHOLD))

        records.append({
            "pass_id":      p["pass_id"],
            "anomaly_type": l["anomaly_type"],
            "eps_anomaly":  is_eps_anom,
            "if_score":     round(if_score, 4),
            "oc_score":     round(oc_score, 4),
            "ae_score":     round(ae_score, 4),
        })

    # ------------------------------------------------------------------
    # Comparison table (Step 9.3 deliverable — one row per model)
    # ------------------------------------------------------------------
    print("--- EPS Anomaly Detection: Precision / Recall / F1 ---")
    print(f"  Binary threshold = {THRESHOLD}  |  "
          f"EPS anomaly passes: {sum(y_true)} / {len(y_true)} total\n")
    print(f"  {'Model':<30} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("  " + "-" * 62)

    results = {}
    for name, y_pred in [
        ("Isolation Forest (primary)", y_if),
        ("One-Class SVM (comparison)", y_oc),
        ("LSTM Autoencoder (Step 9.3)", y_ae),
    ]:
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="binary", zero_division=0
        )
        results[name] = (prec, rec, f1)
        print(f"  {name:<30} {prec:>10.3f} {rec:>10.3f} {f1:>10.3f}")

    # ------------------------------------------------------------------
    # Concrete before/after examples (Step 9.4 requirement)
    # ------------------------------------------------------------------
    print("\n--- Concrete Examples (Step 9.4) ---")

    eps_examples = [r for r in records if r["eps_anomaly"]][:4]
    nom_examples = [r for r in records if not r["eps_anomaly"]
                    and "NOM" in r["pass_id"]][:3]

    hdr = f"  {'Pass ID':22s} {'Type':14s} {'IF':>8} {'OCSVM':>8} {'LSTM-AE':>8}"
    sep = "  " + "-" * 64

    print("\nEPS Anomaly Passes (all three scores should be high):")
    print(hdr)
    print(sep)
    for r in eps_examples:
        print(f"  {r['pass_id']:22s} {r['anomaly_type']:14s} "
              f"{r['if_score']:>8.3f} {r['oc_score']:>8.3f} {r['ae_score']:>8.3f}")

    print("\nNominal Passes (all three scores should be near 0):")
    print(hdr)
    print(sep)
    for r in nom_examples:
        print(f"  {r['pass_id']:22s} {r['anomaly_type']:14s} "
              f"{r['if_score']:>8.3f} {r['oc_score']:>8.3f} {r['ae_score']:>8.3f}")

    # ------------------------------------------------------------------
    # Interpretation (written into output so it can go straight into report)
    # ------------------------------------------------------------------
    f1_if = results["Isolation Forest (primary)"][2]
    f1_ae = results["LSTM Autoencoder (Step 9.3)"][2]

    print("\n--- Interpretation ---")
    if f1_ae > f1_if + 0.05:
        verdict = (
            "LSTM-AE outperforms Isolation Forest by a meaningful margin. "
            "Consider switching the EPS primary model and note this in the report."
        )
    elif abs(f1_ae - f1_if) <= 0.05:
        verdict = (
            "LSTM-AE ties Isolation Forest (within 0.05 F1). "
            "The classical model is defensible at this data scale — "
            "this empirically validates PDD Section 6.5's argument that "
            "deep learning does not add value when training data is small "
            "and synthetic. Isolation Forest remains the production model."
        )
    else:
        verdict = (
            "LSTM-AE underperforms Isolation Forest. "
            "The classical model is clearly preferable at this data scale — "
            "this empirically validates PDD Section 6.5. "
            "Isolation Forest remains the production model."
        )
    print(f"  {verdict}")
    print()


if __name__ == "__main__":
    main()
