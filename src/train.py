#!/usr/bin/env python3
"""
src/train.py
Keras training script that consumes CSV manifests created earlier.
"""

import argparse
import json
from pathlib import Path
import logging
import tensorflow as tf
import os
import time
from data_pipeline import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def build_simple_cnn(input_shape=(224, 224, 1), num_classes=1):
    inputs = tf.keras.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(inputs)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.Conv2D(32, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    outputs = tf.keras.layers.Dense(
        num_classes if num_classes > 1 else 1,
        activation="sigmoid"
    )(x)

    model = tf.keras.Model(inputs, outputs)
    loss = "binary_crossentropy"

    return model, loss


def classification_metrics():
    """
    Full classification metrics for binary + multi-label tasks
    """
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
    ]


def save_run_metadata(out_dir: Path, args, extra=None):
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "args": vars(args),
        "tf_version": tf.__version__,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES", None)
    }
    if extra:
        meta.update(extra)

    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Datasets
    train_ds, train_meta = build_dataset(
        args.train_manifest,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        repeat=True
    )

    val_ds, val_meta = build_dataset(
        args.val_manifest,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=False,
        repeat=False
    )

    num_classes = train_meta["num_classes"]
    logging.info("Dataset meta: train=%s val=%s", train_meta, val_meta)

    # Model
    model, loss = build_simple_cnn(
        input_shape=(args.image_size, args.image_size, 1),
        num_classes=num_classes
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.learning_rate),
        loss=loss,
        metrics=classification_metrics()
    )

    # Callbacks
    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(out_dir / "model.h5"),
        monitor="val_auc_roc",
        mode="max",
        save_best_only=True,
        verbose=1
    )

    tb_cb = tf.keras.callbacks.TensorBoard(log_dir=str(out_dir / "logs"))

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_roc",
        mode="max",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    steps_per_epoch = max(1, train_meta["num_examples"] // args.batch_size)
    validation_steps = max(1, val_meta["num_examples"] // args.batch_size)

    # Resume if provided
    if args.resume_checkpoint and Path(args.resume_checkpoint).exists():
        logging.info("Resuming from checkpoint: %s", args.resume_checkpoint)
        model.load_weights(args.resume_checkpoint)

    save_run_metadata(
        out_dir,
        args,
        extra={
            "num_train": train_meta["num_examples"],
            "num_val": val_meta["num_examples"],
            "num_classes": num_classes
        }
    )

    # Train
    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        callbacks=[ckpt_cb, tb_cb, es_cb]
    )

    # Save final metrics (includes val_auc_roc & val_auc_pr)
    final_metrics = {k: float(v[-1]) for k, v in history.history.items()}

    with open(out_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    logging.info("Training finished. Artifacts saved to %s", out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default=None)
    args = parser.parse_args()
    main(args)