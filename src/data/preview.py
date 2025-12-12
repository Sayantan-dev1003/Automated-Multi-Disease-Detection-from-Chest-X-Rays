#!/usr/bin/env python3
"""
src/data/preview.py
Quick preview and sanity checks for a labels or manifest CSV.
"""
import argparse
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(args):
    p = Path(args.csv)
    assert p.exists(), f"{p} not found"
    df = pd.read_csv(p)
    logging.info("Loaded CSV: %s (rows=%d, cols=%d)", p, len(df), len(df.columns))
    print("Columns:", list(df.columns))
    print("\nFirst 10 rows:")
    print(df.head(10))
    # basic label detection
    label_cols = [c for c in df.columns if c.lower() in ("labels", "finding labels", "finding_labels")]
    if not label_cols:
        # try detect numeric label columns (0/1)
        candidate = [c for c in df.columns if df[c].dropna().isin([0, 1]).all() and df[c].nunique() <= 3]
        if candidate:
            logging.info("Detected numeric label columns: %s", candidate)
    else:
        logging.info("Detected label column(s): %s", label_cols)
        print(df[label_cols[0]].value_counts().head(30))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to labels or manifest CSV")
    args = parser.parse_args()
    main(args)