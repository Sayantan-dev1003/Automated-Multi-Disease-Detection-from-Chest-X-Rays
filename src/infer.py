#!/usr/bin/env python3
"""
src/infer.py
Inference script for Chest X-ray multi-label classification
"""

import argparse
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

def build_model(input_shape, num_classes):
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


def preprocess_image(path, image_size):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (image_size, image_size))
    return img


def main(args):
    # Load labels
    with open(args.labels, "r") as f:
        label_data = json.load(f)

    label_names = label_data["labels"]
    num_classes = label_data["num_classes"]

    # Build + load model
    model = build_model(
        input_shape=(args.image_size, args.image_size, 3),
        num_classes=num_classes
    )
    model.load_weights(args.weights)

    # Collect images
    inp = Path(args.input)
    images = [inp] if inp.is_file() else sorted(inp.glob("*"))

    results = []

    for img_path in images:
        img = preprocess_image(img_path, args.image_size)
        img = tf.expand_dims(img, 0)

        preds = model.predict(img, verbose=0)[0]

        results.append({
            "image": img_path.name,
            "predictions": {
                label_names[i]: float(preds[i])
                for i in range(num_classes)
            }
        })

    # Save / print output
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved predictions to {args.output_json}")
    else:
        for r in results:
            print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Image or folder")
    parser.add_argument("--weights", required=True, help="model.weights.h5")
    parser.add_argument("--labels", required=True, help="label_names.json")
    parser.add_argument("--image_size", type=int, default=192)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()

    main(args)