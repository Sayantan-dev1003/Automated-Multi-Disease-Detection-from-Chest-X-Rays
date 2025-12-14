#!/usr/bin/env python3
"""
src/data/data_pipeline.py
tf.data pipeline for image loading from a CSV manifest.

Manifest expected columns:
- image_path : path relative to manifest file parent (e.g. images/img001.png)
- labels OR one/more binary label columns
"""
import argparse
from pathlib import Path
import pandas as pd
import tensorflow as tf
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _parse_pipe_labels(series, label_names=None):
    all_sets = [set(str(x).split("|")) if str(x) else set() for x in series]
    if label_names is None:
        label_names = sorted({v for s in all_sets for v in s if v and v != 'nan'})
    label_to_idx = {l: i for i, l in enumerate(label_names)}
    arr = np.zeros((len(series), len(label_names)), dtype=np.float32)
    for i, s in enumerate(all_sets):
        for lab in s:
            if lab in label_to_idx:
                arr[i, label_to_idx[lab]] = 1.0
    return arr, label_names


def preprocess_image_bytes(image_bytes, image_size=(192, 192), channels=3):
    img = tf.io.decode_image(image_bytes, channels=channels, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size)
    return img


def build_dataset(manifest_csv, images_root, batch_size=16, image_size=(192, 192), shuffle=True, repeat=False):
    manifest_csv = Path(manifest_csv)
    assert manifest_csv.exists(), f"{manifest_csv} not found"
    df = pd.read_csv(manifest_csv)
    images_root = Path(images_root)

    if "image_path" not in df.columns:
        raise ValueError("manifest must contain 'image_path' column pointing to images relative to the manifest file")

    image_paths = [str(images_root / p) for p in df["image_path"].tolist()]

    # Determine labels
    if "labels" in df.columns:
        # pipe-separated strings -> convert to multi-hot numpy array
        labels_arr, label_names = _parse_pipe_labels(df["labels"])
        num_classes = labels_arr.shape[1]
        labels_ds = labels_arr
    else:
        # find the first non-image column and assume it's numeric label(s)
        label_cols = [c for c in df.columns if c != "image_path"]
        if label_cols:
            labels_df = df[label_cols].fillna(0)
            labels_ds = labels_df.values.astype(np.float32)
            if labels_ds.ndim == 1:
                labels_ds = labels_ds.reshape(-1, 1)
            num_classes = labels_ds.shape[1]
            label_names = label_cols
        else:
            # no labels -> zero vector
            labels_ds = np.zeros((len(df), 1), dtype=np.float32)
            num_classes = 1
            label_names = ["dummy"]

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels_ds))
    if shuffle:
        ds = ds.shuffle(buffer_size=min(len(image_paths), 2000))
    def _load(path, label):
        image_bytes = tf.io.read_file(path)
        image = preprocess_image_bytes(image_bytes, image_size=image_size, channels=3)
        return image, label
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    # return dataset and some meta info
    meta = {"num_examples": len(image_paths), "num_classes": num_classes, "label_names": label_names}
    return ds, meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, nargs=2, default=(192, 192))
    args = parser.parse_args()
    ds, meta = build_dataset(args.manifest, batch_size=args.batch_size, image_size=tuple(args.image_size))
    logging.info("Dataset built: %s", meta)
    for images, labels in ds.take(1):
        logging.info("Batch shapes: images=%s labels=%s", images.shape, labels.shape)