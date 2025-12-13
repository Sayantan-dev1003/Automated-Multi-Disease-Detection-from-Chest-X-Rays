#!/usr/bin/env python3
"""
src/train.py
DenseNet121 training for Chest X-ray multi-label classification
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")

import argparse
import json
from pathlib import Path
import logging
import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("mixed_float16")
import time
from data_pipeline import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

def build_densenet121(input_shape, num_classes, fine_tune=False):
    base_model = tf.keras.applications.DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    if fine_tune:
        base_model.trainable = True
        # Freeze early layers, fine-tune last blocks only
        for layer in base_model.layers[:-50]:
            layer.trainable = False
    else:
        base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=fine_tune)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)

    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    model = tf.keras.Model(inputs, outputs)
    return model

def classification_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
    ]

def save_run_metadata(out_dir, args, extra=None):
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()),
        "args": vars(args),
        "tf_version": tf.__version__,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")
    }
    if extra:
        meta.update(extra)

    with open(out_dir / "run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds, train_meta = build_dataset(
        args.train_manifest,
        images_root=args.images_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=True,
        repeat=True
    )

    val_ds, val_meta = build_dataset(
        args.val_manifest,
        images_root=args.images_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=False,
        repeat=False
    )

    model = build_densenet121(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=train_meta["num_classes"],
        fine_tune=args.fine_tune
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.learning_rate),
        loss="binary_crossentropy",
        metrics=classification_metrics()
    )

    if args.resume_checkpoint:
        logging.info("Loading checkpoint: %s", args.resume_checkpoint)
        model.load_weights(args.resume_checkpoint)

    ckpt_cb = tf.keras.callbacks.ModelCheckpoint(
        str(out_dir / "model.h5"),
        monitor="val_auc_roc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1
    )

    es_cb = tf.keras.callbacks.EarlyStopping(
        monitor="val_auc_roc",
        mode="max",
        patience=5,
        restore_best_weights=True
    )

    steps_per_epoch = train_meta["num_examples"] // args.batch_size
    val_steps = val_meta["num_examples"] // args.batch_size

    save_run_metadata(out_dir, args)

    history = model.fit(
        train_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=val_steps,
        callbacks=[ckpt_cb, es_cb]
    )

    with open(out_dir / "metrics.json", "w") as f:
        json.dump({k: float(v[-1]) for k, v in history.history.items()}, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--resume_checkpoint", default=None)
    parser.add_argument("--fine_tune", action="store_true")
    args = parser.parse_args()
    main(args)