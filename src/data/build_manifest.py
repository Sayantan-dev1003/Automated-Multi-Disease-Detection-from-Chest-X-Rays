#!/usr/bin/env python3
"""
build_manifest.py

Builds a manifest CSV mapping image paths to labels.
Designed for NIH ChestXray14-style datasets.

Output columns:
- image_path   (relative path, e.g. images/00000001_000.png)
- labels       (pipe-separated string, e.g. 'Effusion|Infiltration')
"""

import argparse
from pathlib import Path
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main(args):
    labels_csv = Path(args.labels_csv)
    images_root = Path(args.images_root)
    out_path = Path(args.out)

    # -------- checks --------
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels CSV not found: {labels_csv}")
    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")

    logging.info("Reading labels CSV: %s", labels_csv)
    df = pd.read_csv(labels_csv)

    # -------- expected NIH columns --------
    if "Image Index" not in df.columns:
        raise ValueError("Expected column 'Image Index' not found in labels CSV")
    if "Finding Labels" not in df.columns:
        raise ValueError("Expected column 'Finding Labels' not found in labels CSV")

    # -------- build image filename mapping --------
    logging.info("Scanning image directory: %s", images_root)
    image_files = {p.name for p in images_root.rglob("*") if p.is_file()}
    logging.info("Found %d image files", len(image_files))

    # -------- filter rows with existing images --------
    df["image_name"] = df["Image Index"].astype(str)
    df = df[df["image_name"].isin(image_files)].copy()

    if len(df) == 0:
        raise RuntimeError("No images from CSV were found in images_root")

    logging.info("Matched %d images between CSV and image folder", len(df))

    # -------- build manifest --------
    manifest = pd.DataFrame({
        "image_path": df["image_name"].apply(lambda x: f"images/{x}"),
        "labels": df["Finding Labels"].fillna("").astype(str)
    })

    # -------- save --------
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_path, index=False)

    logging.info("Manifest written to: %s", out_path)
    logging.info("Manifest shape: %s", manifest.shape)
    logging.info("Sample rows:")
    logging.info("\n%s", manifest.head(5))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build image-label manifest CSV")
    parser.add_argument("--labels_csv", required=True, help="Path to labels CSV")
    parser.add_argument("--images_root", required=True, help="Path to image directory")
    parser.add_argument("--out", required=True, help="Output manifest CSV path")
    args = parser.parse_args()

    main(args)