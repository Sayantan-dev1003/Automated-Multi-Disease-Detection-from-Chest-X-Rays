#!/usr/bin/env python3
"""
src/data/make_splits.py
Make patient-wise train/val/test splits from a manifest CSV.
Saves train.csv, val.csv, test.csv into out_dir.
"""
import argparse
from pathlib import Path
import pandas as pd
import logging
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(args):
    manifest = Path(args.manifest)
    assert manifest.exists(), f"{manifest} not found"
    df = pd.read_csv(manifest)
    logging.info("Manifest loaded: rows=%d", len(df))

    # detect patient id column
    patient_col = None
    for c in df.columns:
        if "patient" in c.lower():
            patient_col = c
            break
    if patient_col is None and "Patient ID" in df.columns:
        patient_col = "Patient ID"

    if patient_col:
        groups = df[patient_col].astype(str)
        logging.info("Using patient id column for grouping: %s", patient_col)
    else:
        # fallback: group by filename prefix before first underscore or the stem
        groups = df["image_path"].astype(str).apply(lambda p: Path(p).stem.split("_")[0])
        logging.info("No patient id found. Grouping by image filename prefix.")

    # first split off test
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(df, groups=groups))
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)

    # split train into train + val
    val_fraction = args.val_size / (1.0 - args.test_size)
    splitter2 = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=args.seed)
    train_idx2, val_idx2 = next(splitter2.split(df_train, groups=df_train[groups.name] if hasattr(groups, 'name') else df_train.index))
    df_train_final = df_train.iloc[train_idx2].reset_index(drop=True)
    df_val_final = df_train.iloc[val_idx2].reset_index(drop=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_train_final.to_csv(out_dir / "train.csv", index=False)
    df_val_final.to_csv(out_dir / "val.csv", index=False)
    df_test.to_csv(out_dir / "test.csv", index=False)
    logging.info("Wrote splits to %s (train=%d val=%d test=%d)", out_dir, len(df_train_final), len(df_val_final), len(df_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="Path to manifest CSV")
    parser.add_argument("--out_dir", required=True, help="Output directory to save splits")
    parser.add_argument("--test_size", type=float, default=0.15)
    parser.add_argument("--val_size", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    main(args)