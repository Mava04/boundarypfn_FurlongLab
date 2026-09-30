# boundarypfn

Selection of oriented motif pairs that predict boundary pairing, using TabPFN
and SHAP. Motif pairs are selected on dataset 1 (inserted boundary pairs) with
5-fold cross-validation and tested on dataset 2 (genomic boundary pairs).

Code for the manuscript "Heterotypic directional motifs contribute to TAD boundary function in *Drosophila*".

## Repository contents

| File or folder | Purpose |
|---|---|
| `pipeline.py` | All analysis code (preselection, CV, SHAP, selection, evaluation, figures) |
| `run_analysis.py` | Runs the analysis for one config file |
| `permutation_test.py` | Label-permutation baseline |
| `make_demo_data.py` | Generates the simulated demo data |
| `config/` | `demo.yaml` (demo) and `manuscript.yaml` (manuscript settings) |
| `data/demo/` | Simulated demo data |
| `data/processed/` | Place the manuscript input tables here |
| `notebooks/feature_selection.ipynb` | Same analysis as an annotated notebook |
| `pbs/permutations.pbs` | PBS array job for the permutation test |
| `environment.yml` | Conda environment |
| `environment_exact.txt` | Full export of the environment used (reference only) |

## 1. System requirements

**Operating system:** Linux (x86_64). Tested on CentOS Linux 7 (Core).

**Software:** Python 3.11 and the packages in `environment.yml`: tabpfn 2.1.0,
tabpfn-extensions 0.1.1, torch 2.6.0 (CUDA 11.8 build), shap 0.48.0,
optuna 4.4.0, scikit-learn 1.6.1, numpy 2.2.6, pandas 2.3.1,
matplotlib 3.10.3, pyyaml 6.0.2.

**Tested on:** CentOS Linux 7 (Core), NVIDIA Tesla T4 GPU (32 GB), CUDA 11.8.

**Hardware:** an NVIDIA GPU is recommended. The code also runs on CPU, more
slowly. Memory use is a few GB of GPU memory and under 8 GB of system memory.

**Pretrained model:** the TabPFN weights are downloaded automatically on first
use, which needs internet access once. They are distributed by Prior Labs
under their own license (https://github.com/PriorLabs/TabPFN).

## 2. Installation

```bash
git clone https://github.com/Mava04/boundarypfn
cd boundarypfn
conda env create -f environment.yml
conda activate boundarypfn
```

For CPU only, edit the torch line in `environment.yml` as described in its
header.

Typical install time on a standard desktop computer: 5-10 min.

## 3. Demo

All commands are run from the repository root. The demo data (120 pairs in
dataset 1, 300 in dataset 2, 60 motif pairs of which 12 are informative) are
included. `python make_demo_data.py` regenerates them.

```bash
python run_analysis.py config/demo.yaml
```

Expected output in `results/demo/`:

- `tables/selected.csv`: the 10 selected motif pairs, mostly among the 12
  simulated informative ones
- `tables/summary.csv`: summary metrics; dataset 2 AUROC well above 0.5
- `tables/metrics.csv`, `tables/shap_ranking_cv.csv`, `tables/preselected.csv`
- `images/shap_cv.png`, `images/shap_final_model.png`, `images/roc.png`,
  `images/precision_recall.png`

Expected run time: ~4 min with a GPU.

Permutation test on the demo (real labels + 30 permutations):

```bash
python permutation_test.py run config/demo.yaml --start 0 --n 31
python permutation_test.py summarize config/demo.yaml
```

Output: `tables/permutation_pvalues.csv`, `tables/permutations.csv`,
`images/permutation_null.png`.

## 4. Instructions for use

### Input tables (CSV)

- **Dataset 1** and **dataset 2** (positives and negatives in separate files):
  columns `ID`, `Boundary` (1 = pairing, 0 = non-pairing), and one column per
  oriented motif pair, named `<orientation>_<motifA>_AND_<motifB>`, holding
  its count. In dataset 1, each `ID` contains `TADA` or `TADB` (insertion site).
- **Motif-pair metrics**: columns `Motif_pair_wOr`, `Freq_genome_Or`,
  `N_genome_Or`, `N_genome_anyOr` (genome occurrence). The table may also
  contain `Freq_work_*` and `N_work_*`; these are recomputed from dataset 1
  (see below) and used only by `--check`.

### Running on your data

Copy `config/demo.yaml`, edit the paths and thresholds, and run
`python run_analysis.py my_config.yaml`. All parameters and their defaults are
listed in `pipeline.py` (class `Config`).

### Method

1. **Preselection.** For each motif pair and insertion site, the working rate
   is the fraction of dataset 1 pairs carrying the motif pair (count > 0) that
   are working (`Boundary == 1`). Motif pairs are kept if the working rate is
   at least 0.8 at both sites, at least 5 working pairs carry them, and they
   pass the genome thresholds.
2. **CV with all preselected motif pairs.** Dataset 1 is split into 5
   stratified folds (seed 42). In each training fold, features with variance
   at most 1e-3 are removed, and one feature of each pair with |Pearson r| > 0.9.
   TabPFN is fit and SHAP values (class 1) are computed on the held-out fold.
3. **Selection.** Motif pairs are ranked by the median over folds of mean
   |SHAP|; the top 40 are kept.
4. **Threshold.** TabPFN is refit on the same folds with the 40 motif pairs.
   The decision threshold maximizing mean F1 across folds is chosen with Optuna
   (50 trials).
5. **Evaluation.** The final model is fit on all of dataset 1 and evaluated on
   dataset 2 with the chosen threshold.

Dataset 2 is used only in step 5. Steps 1, 3 and 4 use the dataset 1 labels,
so CV metrics are optimistic; dataset 2 metrics are the unbiased estimate.

**Permutation test.** Dataset 1 labels are shuffled (class proportions kept)
and steps 1 to 5 are rerun, including the working rates. Dataset 2 labels are
not shuffled. The p-value is (1 + number of permutations with a value at least
as high as observed) / (1 + number of permutations). If no motif pair passes
preselection, no model is built and the permutation counts as lower than
observed.

## 5. Reproducing the manuscript results

1. Put the input tables in `data/processed/` (see `data/processed/README.md`).
2. Check that recomputed working rates match the stored ones
   (`identical: True`):
   ```bash
   python run_analysis.py config/manuscript.yaml --check
   ```
3. Main analysis (~4 min on one GPU):
   ```bash
   python run_analysis.py config/manuscript.yaml
   ```
4. Permutation test (200 permutations as 10 PBS jobs, ~1 min per
   permutation), then summary:
   ```bash
   qsub pbs/permutations.pbs
   python permutation_test.py summarize config/manuscript.yaml
   ```

| Output | File in `results/[your_name]/` |
|---|---|
| Figure: SHAP | `images/shap_cv.png`, `images/shap_final_model.png` |
| Figure: ROC, PR | `images/roc.png`, `images/precision_recall.png` |
| Table: metrics | `tables/metrics.csv`, `tables/summary.csv` |
| Figure: permutation | `images/permutation_null.png` |

## License

Code: MIT (see `LICENSE`). TabPFN weights: see the TabPFN license.

## Citation

[Citation after publication]
