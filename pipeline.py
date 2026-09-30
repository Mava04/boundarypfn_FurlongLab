"""Motif-pair feature selection with TabPFN and SHAP.

All analysis code lives in this file. The scripts run_analysis.py and
permutation_test.py only call the functions below.

Terminology
-----------
dataset 1   inserted boundary pairs (IDs contain "TADA" or "TADB", the
            insertion site). Used for preselection, cross-validation (CV),
            feature selection and threshold tuning.
dataset 2   genomic boundary pairs. Used only for the final evaluation.
feature     one oriented motif pair, e.g. "convergent_CISBP.pita_AND_CISBP.pita".
            Its value is the count of that motif pair in a boundary pair.
label       Boundary = 1 (pairing / working) or 0 (non-pairing).

Steps (run_pipeline)
--------------------
1. Preselection: keep motif pairs with a high working rate in dataset 1 and
   enough occurrences in the genome (preselect).
2. CV on dataset 1 with all preselected features. In each training fold,
   drop low-variance and correlated features, fit TabPFN, and compute SHAP
   values on the held-out fold.
3. Rank features by the median over folds of mean |SHAP|. Keep the top N.
4. Refit on the same folds with the selected features. Tune the decision
   threshold to maximize mean F1 across folds.
5. Fit the final model on all of dataset 1. Evaluate it on dataset 2.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score, auc, average_precision_score, f1_score,
    precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)
from sklearn.model_selection import StratifiedKFold

import warnings

# SHAP passes NumPy arrays to models fit on DataFrames. Column order is
# identical
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", message="X has feature names, but")

# ======================================================================
# Configuration
# ======================================================================
@dataclass
class Config:
    """All parameters of one analysis. Loaded from a YAML file."""

    # Input tables (CSV) and output folder
    cv_file: str          # dataset 1
    ext_pos_file: str     # dataset 2, positives
    ext_neg_file: str     # dataset 2, negatives
    metrics_file: str     # motif-pair table with the genome columns
    out_dir: str = "results/run"

    # Preselection thresholds
    min_workA: float = 0.8           # working rate at the TADA site
    min_workB: float = 0.8           # working rate at the TADB site
    min_rep_ins: int = 5             # working pairs carrying the motif pair (TADA + TADB)
    min_freq_genome_or: float = 0.25
    min_rep_genome_or: int = 10
    min_rep_genome_anyor: int = 0

    # Feature filtering inside each training fold
    var_threshold: float = 1e-3      # drop features with variance <= this
    corr_threshold: float = 0.9      # drop one of each pair with |r| > this

    # Model and selection
    n_features: int = 40             # number of motif pairs kept after SHAP ranking
    n_splits: int = 5
    random_state: int = 42
    shap_evals_per_feature: int = 3        # SHAP budget during CV
    final_shap_evals_per_feature: int = 10  # SHAP budget for the final model
    n_optuna_trials: int = 50
    device: str = "auto"             # "auto", "cuda" or "cpu"


def load_config(path: str, **overrides) -> Config:
    """Read a YAML config. Overrides that are None are ignored."""
    with open(path) as fh:
        d = yaml.safe_load(fh) or {}
    d.update({k: v for k, v in overrides.items() if v is not None})
    return Config(**d)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ======================================================================
# Data loading and preselection
# ======================================================================
def load_data(cfg: Config) -> dict:
    """Load both datasets with all motif-pair columns."""
    d1 = pd.read_csv(cfg.cv_file)
    d2 = pd.concat([pd.read_csv(cfg.ext_pos_file), pd.read_csv(cfg.ext_neg_file)],
                   ignore_index=True)
    return {
        "X1": d1.drop(columns=["ID", "Boundary"]),
        "y1": d1["Boundary"].to_numpy(),
        "ids1": d1["ID"].astype(str).to_numpy(),
        "X2": d2.drop(columns=["ID", "Boundary"]),
        "y2": d2["Boundary"].to_numpy(),
    }


def working_rates(X: pd.DataFrame, y, ids) -> pd.DataFrame:
    """Working rate of each motif pair at each insertion site.

    A motif pair is present in a boundary pair when its count is > 0.
    For site in (TADA, TADB), using only rows whose ID contains the site:
        N_work_<site>    = working pairs (y == 1) carrying the motif pair
        Freq_work_<site> = N_work_<site> / pairs carrying the motif pair

    These values depend on the labels, so the permutation test recomputes
    them for every permutation.
    """
    ids = pd.Series(ids).astype(str)
    present = X.reset_index(drop=True).gt(0)
    works = np.asarray(y) == 1
    out = pd.DataFrame(index=X.columns)
    for site in ("TADA", "TADB"):
        rows = ids.str.contains(site).to_numpy()
        n_work = present[rows & works].sum(axis=0)
        n_present = present[rows].sum(axis=0)
        out[f"N_work_{site}"] = n_work
        out[f"Freq_work_{site}"] = n_work / n_present.replace(0, np.nan)
    return out


def preselect(cfg: Config, data: dict, y1=None) -> list[str]:
    """Motif pairs passing all preselection thresholds.

    Working rates are computed from the dataset 1 labels (y1, default: the
    real labels). Genome columns are read from cfg.metrics_file.
    """
    y1 = data["y1"] if y1 is None else y1
    wr = working_rates(data["X1"], y1, data["ids1"])
    genome = pd.read_csv(cfg.metrics_file).set_index("Motif_pair_wOr")[
       ["Freq_genome_Or", "N_genome_Or", "N_genome_anyOr"]]
    m = wr.join(genome, how="inner")

    keep = (
        (m["Freq_work_TADA"] >= cfg.min_workA)
        & (m["Freq_work_TADB"] >= cfg.min_workB)
        & ((m["N_work_TADA"] + m["N_work_TADB"]) >= cfg.min_rep_ins)
        & (m["Freq_genome_Or"] >= cfg.min_freq_genome_or)
        & (m["N_genome_Or"] >= cfg.min_rep_genome_or)
        & (m["N_genome_anyOr"] >= cfg.min_rep_genome_anyor)
    )
    return m.index[keep].tolist()


def check_preselection(cfg: Config, data: dict) -> dict:
    """Compare recomputed working rates with those stored in metrics_file.

    Use this once on the real data: the recomputed preselection must match
    the one used in the manuscript.
    """
    stored = pd.read_csv(cfg.metrics_file).set_index("Motif_pair_wOr")
    wr = working_rates(data["X1"], data["y1"], data["ids1"])
    common = wr.index.intersection(stored.index)
    report = {f"max_abs_diff_{c}": float((wr.loc[common, c] - stored.loc[common, c])
                                         .abs().max()) for c in wr.columns}

    # preselection with the stored working rates (as in the notebook)
    s = stored
    old = s.index[(s["Freq_work_TADA"] >= cfg.min_workA)
                  & (s["Freq_work_TADB"] >= cfg.min_workB)
                  & ((s["N_work_TADA"] + s["N_work_TADB"]) >= cfg.min_rep_ins)
                  & (s["Freq_genome_Or"] >= cfg.min_freq_genome_or)
                  & (s["N_genome_Or"] >= cfg.min_rep_genome_or)
                  & (s["N_genome_anyOr"] >= cfg.min_rep_genome_anyor)]
    old = set(data["X1"].columns.intersection(old))
    new = set(preselect(cfg, data))
    report.update(n_stored=len(old), n_recomputed=len(new), identical=old == new,
                  only_stored=sorted(old - new), only_recomputed=sorted(new - old))
    return report


# ======================================================================
# Model helpers
# ======================================================================
def filter_features(X_train, X_test, var_threshold, corr_threshold):
    """Drop low-variance and highly correlated features.

    Thresholds are applied to the training fold only; the same columns are
    then dropped from the test fold. For each correlated pair, the later
    column is removed. Returns the filtered data and a table of removed
    correlated features.
    """
    var = X_train.var(axis=0)
    low_var = var[var <= var_threshold].index
    X_train, X_test = X_train.drop(columns=low_var), X_test.drop(columns=low_var)

    corr = X_train.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    removed, records = [], []
    for col in upper.columns:
        hits = upper[col][upper[col] > corr_threshold]
        if len(hits):
            removed.append(col)
            records += [{"kept_feature": k, "removed_feature": col, "correlation": r}
                        for k, r in hits.items()]
    return (X_train.drop(columns=removed), X_test.drop(columns=removed),
            pd.DataFrame(records))


def new_model(device):
    """TabPFN classifier. balance_probabilities corrects for class imbalance."""
    from tabpfn import TabPFNClassifier
    return TabPFNClassifier(device=device, balance_probabilities=True)

def shap_values(model, X, max_evals):
    """SHAP values for class 1 (permutation explainer from tabpfn-extensions).

    shap's automatic choice switches to the exact explainer for <= 10
    features, which does not accept max_evals. Forcing the permutation
    explainer keeps the method identical to runs with more features.
    """
    from tabpfn_extensions import interpretability
    sv = interpretability.shap.get_shap_values(
        model, X, max_evals=max_evals, algorithm="permutation")
    return sv.values[:, :, 1]

def classification_metrics(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "aupr": average_precision_score(y_true, y_prob),
    }


# ======================================================================
# Pipeline
# ======================================================================
def run_pipeline(cfg: Config, data: dict, y1=None, seed=None,
                 save=True, verbose=True) -> dict:
    """Run steps 1 to 5 (see module docstring).

    y1    dataset 1 labels to use (default: real labels; the permutation
          test passes permuted labels). Dataset 2 labels are never changed.
    seed  seed for threshold tuning (default: cfg.random_state).
    save  write tables to cfg.out_dir/tables.

    Returns a dict with a one-line summary and the objects needed for plots.
    Raises ValueError if no motif pair passes preselection.
    """
    import optuna

    log = print if verbose else (lambda *a, **k: None)
    device = resolve_device(cfg.device)
    y1 = data["y1"] if y1 is None else y1
    y2 = data["y2"]
    seed = cfg.random_state if seed is None else seed
    tables = os.path.join(cfg.out_dir, "tables")
    if save:
        os.makedirs(tables, exist_ok=True)

    # ---- Step 1: preselection ----
    pre = preselect(cfg, data, y1)
    if not pre:
        raise ValueError("no motif pair passed preselection")
    X1, X2 = data["X1"][pre], data["X2"][pre]
    log(f"Dataset 1: {len(y1)} pairs | dataset 2: {len(y2)} pairs | "
        f"{len(pre)} preselected motif pairs")

    folds = list(StratifiedKFold(cfg.n_splits, shuffle=True,
                                 random_state=cfg.random_state).split(X1, y1))

    # ---- Step 2: CV with all preselected features, SHAP per fold ----
    abs_shap, signed_shap, fold_metrics = [], [], []
    for k, (tr, te) in enumerate(folds, 1):
        log(f"  fold {k}/{cfg.n_splits}: fit and SHAP")
        X_tr, X_te, removed = filter_features(X1.iloc[tr], X1.iloc[te],
                                              cfg.var_threshold, cfg.corr_threshold)
        if X_tr.shape[1] == 0:
            raise ValueError(f"no feature left after filtering in fold {k}")
        if save:
            removed.to_csv(f"{tables}/removed_correlated_fold{k}.csv", index=False)
        model = new_model(device).fit(X_tr, y1[tr])
        fold_metrics.append({"fold": k, "model": "all_features",
                             **classification_metrics(
                                 y1[te], model.predict_proba(X_te)[:, 1], 0.5)})
        sv = shap_values(model, X_te, cfg.shap_evals_per_feature * X_te.shape[1])
        # features removed in this fold get importance 0
        abs_shap.append(pd.Series(np.abs(sv).mean(0), X_te.columns)
                        .reindex(X1.columns, fill_value=0.0))
        signed_shap.append(pd.Series(sv.mean(0), X_te.columns)
                           .reindex(X1.columns, fill_value=0.0))

    # ---- Step 3: rank by median |SHAP| and keep the top N ----
    ranking = pd.concat(abs_shap, axis=1).median(axis=1).sort_values(ascending=False)
    selected = ranking.index[:cfg.n_features].tolist()
    log(f"Selected {len(selected)} motif pairs")

    # ---- Step 4: CV with selected features, tune threshold on F1 ----
    cv_probs = []
    for tr, te in folds:
        m = new_model(device).fit(X1.iloc[tr][selected].values, y1[tr])
        cv_probs.append((te, m.predict_proba(X1.iloc[te][selected].values)[:, 1]))

    def mean_f1(trial):
        t = trial.suggest_float("threshold", 0.05, 0.95)
        return np.mean([f1_score(y1[te], p >= t, zero_division=0)
                        for te, p in cv_probs])

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(mean_f1, n_trials=cfg.n_optuna_trials)
    threshold = study.best_params["threshold"]
    for k, (te, p) in enumerate(cv_probs, 1):
        fold_metrics.append({"fold": k, "model": "selected_features",
                             **classification_metrics(y1[te], p, threshold)})

    # ---- Step 5: final model on dataset 1, evaluation on dataset 2 ----
    final_model = new_model(device).fit(X1[selected].values, y1)
    y2_prob = final_model.predict_proba(X2[selected].values)[:, 1]
    external = classification_metrics(y2, y2_prob, threshold)
    log(f"Threshold {threshold:.3f} | dataset 2: AUROC {external['roc_auc']:.3f}, "
        f"F1 {external['f1']:.3f}")

    cv_sel = pd.DataFrame([m for m in fold_metrics if m["model"] == "selected_features"])
    summary = {
        "n_preselected": len(pre), "n_selected": len(selected), "threshold": threshold,
        "cv_roc_auc": cv_sel["roc_auc"].mean(), "cv_aupr": cv_sel["aupr"].mean(),
        "cv_f1": cv_sel["f1"].mean(),
        **{f"ext_{k}": v for k, v in external.items()},
    }

    if save:
        pd.Series(pre, name="motif_pair").to_csv(f"{tables}/preselected.csv", index=False)
        pd.Series(selected, name="motif_pair").to_csv(f"{tables}/selected.csv", index=False)
        ranking.rename("median_abs_shap").to_csv(f"{tables}/shap_ranking_cv.csv")
        pd.DataFrame(fold_metrics + [{"fold": "dataset2", "model": "final", **external}]) \
            .to_csv(f"{tables}/metrics.csv", index=False)
        pd.Series(summary).to_csv(f"{tables}/summary.csv", header=False)

    return {"summary": summary, "selected": selected, "final_model": final_model,
            "X1": X1, "y1": y1, "y2": y2, "signed_shap": signed_shap,
            "cv_probs": cv_probs, "y2_prob": y2_prob}


# ======================================================================
# Figures
# ======================================================================
def _bar_top_posneg(values: pd.Series, xlabel, title, path):
    """Horizontal bars: 10 most positive and 10 most negative values."""
    top = pd.concat([values.nlargest(10), values.nsmallest(10)])
    top = top[~top.index.duplicated()].sort_values()
    plt.figure(figsize=(8, 6))
    plt.barh(top.index, top.values, color=["red" if v > 0 else "blue" for v in top])
    plt.axvline(0, color="black", lw=0.8)
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def make_figures(cfg: Config, res: dict):
    """SHAP bar plots, ROC and precision-recall curves (CV vs dataset 2)."""
    img = os.path.join(cfg.out_dir, "images")
    os.makedirs(img, exist_ok=True)
    sel, X1, y1, y2 = res["selected"], res["X1"], res["y1"], res["y2"]

    # SHAP: median signed value across CV folds
    _bar_top_posneg(pd.concat(res["signed_shap"], axis=1).median(axis=1),
                    "Median SHAP value (class 1)", "SHAP across CV folds",
                    f"{img}/shap_cv.png")

    # SHAP: final model on dataset 1
    sv = shap_values(res["final_model"], X1[sel],
                     cfg.final_shap_evals_per_feature * len(sel))
    final = pd.Series(sv.mean(0), index=sel)
    final.rename("mean_shap").to_csv(f"{cfg.out_dir}/tables/shap_final_model.csv")
    _bar_top_posneg(final, "Mean SHAP value (class 1)", "SHAP, final model",
                    f"{img}/shap_final_model.png")

    # ROC: mean over CV folds (band = 1 SD) and dataset 2
    grid = np.linspace(0, 1, 100)
    tprs = [np.interp(grid, *roc_curve(y1[te], p)[:2]) for te, p in res["cv_probs"]]
    mean, sd = np.mean(tprs, 0), np.std(tprs, 0)
    fpr2, tpr2, _ = roc_curve(y2, res["y2_prob"])
    plt.figure(figsize=(10, 6))
    plt.plot(grid, mean, label=f"CV mean (AUC = {auc(grid, mean):.2f})")
    plt.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), alpha=0.2)
    plt.plot(fpr2, tpr2, color="red", label=f"Dataset 2 (AUC = {auc(fpr2, tpr2):.2f})")
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(f"{img}/roc.png", dpi=300)
    plt.close()

    # Precision-recall: same layout; dashed lines = class prevalence
    precs = []
    for te, p in res["cv_probs"]:
        prec, rec, _ = precision_recall_curve(y1[te], p)
        order = np.argsort(rec)
        precs.append(np.interp(grid, rec[order], prec[order]))
    mean, sd = np.mean(precs, 0), np.std(precs, 0)
    prec2, rec2, _ = precision_recall_curve(y2, res["y2_prob"])
    plt.figure(figsize=(10, 6))
    plt.plot(grid, mean, label=f"CV mean (AUPR = {auc(grid, mean):.2f})")
    plt.fill_between(grid, np.clip(mean - sd, 0, 1), np.clip(mean + sd, 0, 1), alpha=0.2)
    plt.plot(rec2, prec2, color="red", label=f"Dataset 2 (AUPR = {auc(rec2, prec2):.2f})")
    plt.axhline(np.mean(y1), ls="--", color="blue", label="CV prevalence")
    plt.axhline(np.mean(y2), ls="--", color="red", label="Dataset 2 prevalence")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    plt.tight_layout(rect=[0, 0, 0.82, 1])
    plt.savefig(f"{img}/precision_recall.png", dpi=300)
    plt.close()
