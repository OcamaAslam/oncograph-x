"""
evaluate.py  —  Clinical Twin GNN · Full Evaluation Suite
==========================================================
Computes metrics for both model heads and writes a formatted
report to evaluation_results.txt in the project root.

Usage (from project root):
    python evaluate.py
"""

import os
import sys
import datetime
import warnings
warnings.filterwarnings("ignore")

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.model_selection import StratifiedKFold

# ── project imports ──────────────────────────────────────────────────────────
from src.data_loader import load_and_preprocess_data
from src.model import ClinicalTwinGNN


# ── helpers ──────────────────────────────────────────────────────────────────
def sep(char="=", width=62):
    return char * width


def fmt(label, value, width=36):
    """Right-align a value next to a label."""
    return f"  {label:<{width}} {value}"


# ── evaluation ───────────────────────────────────────────────────────────────
def evaluate(model_path="models/clinical_twin_gnn.pth", data_dir="data"):
    print("Loading data …")
    data, df, feature_names = load_and_preprocess_data(data_dir)
    n_nodes   = data.num_nodes
    n_edges   = data.num_edges
    n_feats   = data.num_features

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data   = data.to(device)

    # ── load model ──────────────────────────────────────────────────────────
    model = ClinicalTwinGNN(in_channels=n_feats, hidden_channels=64).to(device)
    if not os.path.exists(model_path):
        sys.exit(f"[ERROR] Model checkpoint not found at '{model_path}'.\n"
                 "Please run `python -m src.train` first.")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    # ── inference ────────────────────────────────────────────────────────────
    print("Running inference …")
    with torch.no_grad():
        out_rec, out_sur = model(data.x, data.edge_index)

    y_rec_true  = data.y_recurrence.cpu().numpy()           # int  {0, 1}
    y_sur_true  = data.y_survival.cpu().numpy().flatten()   # float [0, 1]

    rec_probs   = torch.softmax(out_rec, dim=-1).cpu().numpy()
    rec_prob1   = rec_probs[:, 1]                           # P(recurrence)
    rec_pred    = np.argmax(rec_probs, axis=1)              # hard labels

    sur_pred    = out_sur.cpu().numpy().flatten()

    # ── class imbalance ───────────────────────────────────────────────────
    pos_count = int(y_rec_true.sum())
    neg_count = int(len(y_rec_true) - pos_count)

    # ── Task A: Recurrence Classification ────────────────────────────────
    acc   = accuracy_score(y_rec_true, rec_pred)
    prec  = precision_score(y_rec_true, rec_pred, zero_division=0)
    rec_s = recall_score(y_rec_true, rec_pred, zero_division=0)
    f1    = f1_score(y_rec_true, rec_pred, zero_division=0)
    auc   = roc_auc_score(y_rec_true, rec_prob1)
    auprc = average_precision_score(y_rec_true, rec_prob1)
    cm    = confusion_matrix(y_rec_true, rec_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    mcc_denom   = np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    mcc         = ((tp*tn - fp*fn) / mcc_denom) if mcc_denom > 0 else 0.0
    bal_acc     = (rec_s + specificity) / 2.0

    clf_report = classification_report(
        y_rec_true, rec_pred, target_names=["No Recurrence", "Recurrence"]
    )

    # ── Task B: Survival Regression ────────────────────────────────────
    mse   = mean_squared_error(y_sur_true, sur_pred)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(y_sur_true, sur_pred)
    r2    = r2_score(y_sur_true, sur_pred)
    ss_res = np.sum((y_sur_true - sur_pred) ** 2)
    ss_tot = np.sum((y_sur_true - y_sur_true.mean()) ** 2)
    ev    = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0  # explained variance

    # Concordance-style: % of pairs where ranking is correct
    n_samp   = len(y_sur_true)
    sample_n = min(n_samp, 2000)          # sample for speed
    rng      = np.random.default_rng(42)
    idx      = rng.choice(n_samp, sample_n, replace=False)
    yt_s     = y_sur_true[idx]
    yp_s     = sur_pred[idx]
    pairs    = [(i, j) for i in range(sample_n) for j in range(i+1, sample_n)
                if yt_s[i] != yt_s[j]]
    concordant = sum(1 for i, j in pairs
                     if (yt_s[i] > yt_s[j]) == (yp_s[i] > yp_s[j]))
    c_index  = concordant / len(pairs) if pairs else 0.5

    # ── per-class survival error ─────────────────────────────────────────
    living_mask = (df["survival_status"].values == "living")
    dead_mask   = ~living_mask
    mae_living  = mean_absolute_error(y_sur_true[living_mask], sur_pred[living_mask]) \
                  if living_mask.sum() > 0 else float("nan")
    mae_dead    = mean_absolute_error(y_sur_true[dead_mask],   sur_pred[dead_mask]) \
                  if dead_mask.sum()   > 0 else float("nan")

    # ── build report text ────────────────────────────────────────────────
    now   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    lines = [
        sep(),
        "  OncoGraph X -- GNN EVALUATION REPORT",
        f"  Generated : {now}",
        f"  Model     : models/clinical_twin_gnn.pth",
        f"  Device    : {str(device).upper()}",
        sep(),
        "",
        sep("-"),
        "  GRAPH STATISTICS",
        sep("-"),
        fmt("Patients (nodes):",         f"{n_nodes:,}"),
        fmt("Clinical-twin edges:",      f"{n_edges:,}"),
        fmt("Input feature dimensions:", f"{n_feats}"),
        fmt("Trainable parameters:",     f"{n_params:,}"),
        "",
        sep("-"),
        "  CLASS DISTRIBUTION  --  Task A: Recurrence",
        sep("-"),
        fmt("No Recurrence  (label 0):", f"{neg_count:,}  ({100*neg_count/n_nodes:.1f}%)"),
        fmt("Recurrence     (label 1):", f"{pos_count:,}  ({100*pos_count/n_nodes:.1f}%)"),
        fmt("Imbalance ratio (neg/pos):",f"{neg_count/max(pos_count,1):.2f}"),
        "",
        sep("="),
        "  TASK A -- RECURRENCE CLASSIFICATION  (Binary)",
        sep("="),
        "",
        fmt("Accuracy:",                 f"{acc:.4f}   ({acc*100:.2f}%)"),
        fmt("Balanced Accuracy:",        f"{bal_acc:.4f}"),
        fmt("Precision  (PPV):",         f"{prec:.4f}"),
        fmt("Recall     (Sensitivity):", f"{rec_s:.4f}"),
        fmt("Specificity (TNR):",        f"{specificity:.4f}"),
        fmt("F1 Score:",                 f"{f1:.4f}"),
        fmt("Matthews Corr. Coef.:",     f"{mcc:.4f}"),
        fmt("ROC-AUC:",                  f"{auc:.4f}"),
        fmt("PR-AUC (Avg. Precision):",  f"{auprc:.4f}"),
        "",
        "  Confusion Matrix:",
        f"  {'':>20}  Predicted 0    Predicted 1",
        f"  {'Actual 0 (No Rec.)':<20}  {tn:^13,}  {fp:^13,}",
        f"  {'Actual 1 (Recur.)':<20}  {fn:^13,}  {tp:^13,}",
        "",
        "  Per-class Classification Report:",
        *[f"  {l}" for l in clf_report.strip().split("\n")],
        "",
        sep("="),
        "  TASK B -- SURVIVAL SCORE REGRESSION  (0-1 scale)",
        sep("="),
        "",
        fmt("Mean Squared Error (MSE):",   f"{mse:.6f}"),
        fmt("Root MSE (RMSE):",            f"{rmse:.6f}"),
        fmt("Mean Absolute Error (MAE):",  f"{mae:.6f}"),
        fmt("R² Score:",                   f"{r2:.4f}"),
        fmt("Explained Variance Score:",   f"{ev:.4f}"),
        fmt("C-Index (concordance est.):", f"{c_index:.4f}"),
        "",
        "  Per-status MAE breakdown:",
        fmt("  MAE — Living patients:",    f"{mae_living:.6f}"),
        fmt("  MAE — Deceased patients:",  f"{mae_dead:.6f}"),
        "",
        sep("-"),
        "  INTERPRETATION GUIDE",
        sep("-"),
        "  ROC-AUC  > 0.80  → Good discrimination",
        "  PR-AUC          → More informative under class imbalance",
        "  MCC             → Best single metric for imbalanced binary tasks",
        "  C-Index > 0.70  → Acceptable survival ranking ability",
        "  R²      > 0.60  → Meaningful regression fit",
        "",
        sep(),
        "  END OF REPORT",
        sep(),
    ]

    report_text = "\n".join(lines) + "\n"

    # ── print to stdout ──────────────────────────────────────────────────
    print("\n" + report_text)

    # ── write to file ────────────────────────────────────────────────────
    out_path = "evaluation_results.txt"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[✓] Results saved to  →  {os.path.abspath(out_path)}")

    return {
        "accuracy": acc, "f1": f1, "roc_auc": auc, "pr_auc": auprc,
        "rmse": rmse, "mae": mae, "r2": r2, "c_index": c_index,
    }


if __name__ == "__main__":
    evaluate()
