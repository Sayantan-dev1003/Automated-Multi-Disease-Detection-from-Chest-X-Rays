#!/usr/bin/env python3
"""
src/eval.py
Load a saved Keras model and run inference on a test manifest. Save predictions to preds.csv.
"""
import argparse
from pathlib import Path
import tensorflow as tf
import pandas as pd
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_image_as_tensor(path, image_size=(224, 224)):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size)
    return img.numpy()


def main(args):
    model_path = Path(args.model)
    assert model_path.exists(), f"{model_path} not found"
    model = tf.keras.models.load_model(str(model_path), compile=False)
    test_manifest = Path(args.test_manifest)
    df = pd.read_csv(test_manifest)
    base = test_manifest.parent

    preds = []
    for _, row in df.iterrows():
        img_path = base / row["image_path"]
        img = load_image_as_tensor(img_path, image_size=(args.image_size, args.image_size))
        x = np.expand_dims(img, 0)
        pred = model.predict(x)[0]
        preds.append(pred.tolist() if getattr(pred, "shape", None) and len(np.atleast_1d(pred)) > 1 else float(np.ravel(pred)[0]))

    df_out = df.copy()
    # attach preds (if multi-dim preds, expand columns)
    if isinstance(preds[0], list):
        pred_arr = np.array(preds)
        for i in range(pred_arr.shape[1]):
            df_out[f"pred_{i}"] = pred_arr[:, i]
    else:
        df_out["pred"] = preds

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_dir / "preds.csv", index=False)
    logging.info("Saved preds.csv to %s", out_dir / "preds.csv")

    # try compute simple metrics if labels numeric present
    label_cols = [c for c in df.columns if c != "image_path"]
    numeric_label_cols = []
    for c in label_cols:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_label_cols.append(c)
        except Exception:
            pass

    if numeric_label_cols:
        try:
            from sklearn.metrics import roc_auc_score
            y_true = df[numeric_label_cols].values
            if isinstance(preds[0], list):
                y_score = np.array(preds)
            else:
                y_score = np.array(preds).reshape(-1, 1)
            rocs = {}
            for i, c in enumerate(numeric_label_cols):
                try:
                    score = roc_auc_score(y_true[:, i], y_score[:, i])
                    rocs[c] = float(score)
                except Exception as e:
                    rocs[c] = None
            with open(out_dir / "metrics.json", "w") as f:
                json.dump({"auroc_per_label": rocs}, f, indent=2)
            logging.info("Wrote metrics.json with AUROC per label")
        except Exception:
            logging.warning("sklearn not available or metric computation failed; skipping ROC computation")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=224)
    args = parser.parse_args()
    main(args)