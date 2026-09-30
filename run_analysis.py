#!/usr/bin/env python
"""Main analysis: feature selection, evaluation on dataset 2, and figures.

Usage:
    python run_analysis.py config/demo.yaml
    python run_analysis.py config/manuscript.yaml --check   # compare preselection
                                                             # with the metrics file
"""
import argparse
import time

import pipeline as pl


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="YAML config file")
    ap.add_argument("--check", action="store_true",
                    help="only compare recomputed and stored preselection, then exit")
    ap.add_argument("--device", help="auto, cuda or cpu (overrides config)")
    args = ap.parse_args()

    cfg = pl.load_config(args.config, device=args.device)
    data = pl.load_data(cfg)

    report = pl.check_preselection(cfg, data)
    if args.check:
        for k, v in report.items():
            print(f"{k}: {v}")
        return
    if not report["identical"]:
        print("WARNING: recomputed preselection differs from metrics_file. "
              "Run with --check for details.")

    t0 = time.time()
    res = pl.run_pipeline(cfg, data)
    pl.make_figures(cfg, res)
    print(f"Done in {(time.time() - t0) / 60:.1f} min. Output: {cfg.out_dir}")


if __name__ == "__main__":
    main()
