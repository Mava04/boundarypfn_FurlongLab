#!/usr/bin/env python
"""Label-permutation baseline.

Dataset 1 labels are shuffled (class proportions kept) and the whole pipeline
is rerun, including preselection, since working rates depend on the labels.
Dataset 2 labels are never shuffled. Permutation 0 is the real-label run.

Usage:
    # run permutations 0..200 (split into chunks for parallel jobs, see pbs/)
    python permutation_test.py run config/demo.yaml --start 0 --n 201
    # empirical p-values and null distribution plot
    python permutation_test.py summarize config/demo.yaml

p-value = (1 + number of permutations >= observed) / (1 + number of permutations).
If no motif pair passes preselection in a permutation, no model is built and
the permutation counts as below the observed value.
"""
import argparse
import glob
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import pipeline as pl

METRICS = ["cv_roc_auc", "cv_aupr", "cv_f1",
           "ext_roc_auc", "ext_aupr", "ext_f1", "ext_accuracy"]
SEED = 12345  # permutation i uses seed SEED + i


def run(args):
    cfg = pl.load_config(args.config, device=args.device)
    data = pl.load_data(cfg)
    outdir = os.path.join(cfg.out_dir, "permutations")
    os.makedirs(outdir, exist_ok=True)

    for i in range(args.start, args.start + args.n):
        path = os.path.join(outdir, f"perm_{i:05d}.json")
        if os.path.exists(path):
            continue  # already done: chunks can be resubmitted safely
        y1 = data["y1"] if i == 0 else \
            np.random.default_rng(SEED + i).permutation(data["y1"])
        t0 = time.time()
        rec = {"perm": i}
        try:
            res = pl.run_pipeline(cfg, data, y1=y1, seed=SEED + i,
                                  save=False, verbose=False)
            rec.update(res["summary"], selected=res["selected"], error="")
        except ValueError as e:  # nothing passed preselection
            rec.update({m: None for m in METRICS}, error=str(e))
        rec["seconds"] = time.time() - t0
        with open(path, "w") as fh:
            json.dump(rec, fh)
        result = rec["error"] or f"dataset 2 AUROC {rec['ext_roc_auc']:.3f}"
        print(f"perm {i}: {result} ({rec['seconds']:.0f} s)", flush=True)


def summarize(args):
    cfg = pl.load_config(args.config)
    files = sorted(glob.glob(os.path.join(cfg.out_dir, "permutations", "perm_*.json")))
    df = pd.DataFrame([json.load(open(f)) for f in files])
    if 0 not in set(df.perm):
        raise SystemExit("Permutation 0 (real labels) is missing: run with --start 0.")
    obs, null = df[df.perm == 0].iloc[0], df[df.perm > 0]
    n = len(null)

    rows = []
    for m in METRICS:
        v = null[m].astype(float)
        exceed = (v.fillna(-np.inf) >= obs[m]).sum()
        rows.append({"metric": m, "observed": obs[m], "null_mean": v.mean(),
                     "null_q95": v.quantile(0.95), "p_value": (1 + exceed) / (1 + n)})
    stats = pd.DataFrame(rows)

    os.makedirs(f"{cfg.out_dir}/tables", exist_ok=True)
    os.makedirs(f"{cfg.out_dir}/images", exist_ok=True)
    df.drop(columns="selected", errors="ignore") \
      .to_csv(f"{cfg.out_dir}/tables/permutations.csv", index=False)
    stats.to_csv(f"{cfg.out_dir}/tables/permutation_pvalues.csv", index=False)
    print(stats.to_string(index=False, float_format="%.4f"))
    print(f"\n{n} permutations, {int(null['error'].astype(bool).sum())} without a "
          f"model, mean {null['seconds'].mean():.0f} s each")

    show = ["cv_roc_auc", "ext_roc_auc", "ext_f1"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, m in zip(axes, show):
        ax.hist(null[m].dropna().astype(float), bins=30, color="grey")
        ax.axvline(obs[m], color="red", lw=2)
        p = stats.set_index("metric").loc[m, "p_value"]
        ax.set_title(f"{m}   p = {p:.3g}", fontsize=9)
    axes[0].set_ylabel("Permutations")
    plt.tight_layout()
    plt.savefig(f"{cfg.out_dir}/images/permutation_null.png", dpi=300)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run permutations")
    r.add_argument("config")
    r.add_argument("--start", type=int, default=0, help="first permutation (0 = real labels)")
    r.add_argument("--n", type=int, default=201, help="number of permutations in this job")
    r.add_argument("--device")
    s = sub.add_parser("summarize", help="p-values and plot")
    s.add_argument("config")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else summarize(args)


if __name__ == "__main__":
    main()
