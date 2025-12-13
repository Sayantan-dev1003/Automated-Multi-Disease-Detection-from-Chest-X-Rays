#!/usr/bin/env python3
"""
src/test.py
Evaluate trained DenseNet121 model on test.csv
"""

import argparse
import json
from pathlib import Path
import tensorflow as tf
from data_pipeline import build_dataset


def build_densenet121(input_shape, num_classes):
    base_model = tf.keras.applications.DenseNet121(
        weights=None,
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.densenet.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

    return tf.keras.Model(inputs, outputs)


def classification_metrics():
    return [
        tf.keras.metrics.BinaryAccuracy(name="accuracy"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.AUC(name="auc_roc", curve="ROC"),
        tf.keras.metrics.AUC(name="auc_pr", curve="PR"),
    ]


def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build test dataset
    test_ds, test_meta = build_dataset(
        args.test_manifest,
        images_root=args.images_root,
        batch_size=args.batch_size,
        image_size=(args.image_size, args.image_size),
        shuffle=False,
        repeat=False
    )

    # Build model
    model = build_densenet121(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=test_meta["num_classes"]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=classification_metrics()
    )

    # Load trained weights
    model.load_weights(args.weights)
    print(f"Loaded weights from: {args.weights}")

    # Evaluate
    results = model.evaluate(test_ds, verbose=1)
    metrics = dict(zip(model.metrics_names, results))

    # Save metrics
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n===== TEST RESULTS =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_manifest", required=True)
    parser.add_argument("--images_root", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=192)
    args = parser.parse_args()

    main(args)