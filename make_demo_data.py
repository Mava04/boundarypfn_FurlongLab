#!/usr/bin/env python
"""Generate a small simulated dataset with the same format as the real inputs.

Writes to data/demo/:
  demo_dataset1.csv            dataset 1 (CV / selection), IDs contain TADA or TADB
  demo_dataset2_pos.csv        dataset 2 positives (external test)
  demo_dataset2_neg.csv        dataset 2 negatives (external test)
  demo_motifpair_metrics.csv   preselection metrics table
Columns: ID, Boundary (1 = pairing, 0 = non-pairing), one count column per
oriented motif pair (<orientation>_<motifA>_AND_<motifB>).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from pipeline import working_rates  # noqa: E402

rng = np.random.default_rng(0)
OUT = os.path.join(ROOT, "data", "demo")
os.makedirs(OUT, exist_ok=True)

orients = ["convergent", "divergent", "tandemplus", "tandemminus"]
motifs = [f"MOTIF{i:02d}" for i in range(12)]
feats = []
while len(feats) < 60:
    a, b = sorted(rng.choice(motifs, 2))
    f = f"{rng.choice(orients)}_{a}_AND_{b}"
    if f not in feats:
        feats.append(f)
informative = feats[:12]


def simulate(n_pos, n_neg, ids):
    y = np.r_[np.ones(n_pos, int), np.zeros(n_neg, int)]
    X = rng.poisson(0.5, size=(len(y), len(feats))).astype(float)
    for j, f in enumerate(feats):
        if f in informative:
            X[:, j] += rng.poisson(1.5, len(y)) * y  # enriched in pairing
    df = pd.DataFrame(X, columns=feats)
    df.insert(0, "Boundary", y)
    df.insert(0, "ID", ids)
    return df


# dataset 1: half of the insertions in TADA, half in TADB
ids1 = [f"ins_{'TADA' if i % 2 == 0 else 'TADB'}_{i:04d}" for i in range(120)]
d1 = simulate(60, 60, ids1).sample(frac=1, random_state=1)
d1.to_csv(f"{OUT}/demo_dataset1.csv", index=False)

d2 = simulate(150, 150, [f"genome_{i:04d}" for i in range(300)])
d2[d2.Boundary == 1].to_csv(f"{OUT}/demo_dataset2_pos.csv", index=False)
d2[d2.Boundary == 0].to_csv(f"{OUT}/demo_dataset2_neg.csv", index=False)

# working rates from dataset 1 labels (as in the real table); genome columns simulated
wm = working_rates(d1[feats], d1["Boundary"], d1["ID"])
n = len(feats)
metrics = pd.DataFrame({
    "Motif_pair_wOr": feats,
    "Freq_work_TADA": wm["Freq_work_TADA"].to_numpy(),
    "Freq_work_TADB": wm["Freq_work_TADB"].to_numpy(),
    "N_work_TADA": wm["N_work_TADA"].to_numpy(),
    "N_work_TADB": wm["N_work_TADB"].to_numpy(),
    "Freq_genome_Or": rng.uniform(0.3, 0.6, n),
    "N_genome_Or": rng.integers(10, 50, n),
    "N_genome_anyOr": rng.integers(10, 80, n),
})
metrics.to_csv(f"{OUT}/demo_motifpair_metrics.csv", index=False)
print(f"Demo data written to {OUT}")
