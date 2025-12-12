#!/usr/bin/env python3
"""
src/infer.py
Simple CLI to run inference on one image or all images in a folder.
"""
import argparse
from pathlib import Path
import tensorflow as tf
import numpy as np
import logging
import json

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def preprocess(path, image_size=(224, 224)):
    img = tf.io.read_file(str(path))
    img = tf.image.decode_image(img, channels=1, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, image_size)
    return img.numpy()


def main(args):
    model = tf.keras.models.load_model(args.model, compile=False)
    inp = Path(args.input)
    if inp.is_dir():
        images = sorted([p for p in inp.iterdir() if p.is_file()])
    else:
        images = [inp]

    out = []
    for p in images:
        img = preprocess(p, image_size=(args.image_size, args.image_size))
        x = np.expand_dims(img, 0)
        pred = model.predict(x)[0]
        if hasattr(pred, "__len__") and len(pred) > 1:
            out.append({"image": str(p), "pred": [float(x) for x in pred]})
        else:
            out.append({"image": str(p), "pred": float(np.ravel(pred)[0])})

    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(out, f, indent=2)
        logging.info("Wrote results to %s", args.output_json)
    else:
        for r in out:
            print(r)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to saved Keras model (.h5)")
    parser.add_argument("--input", required=True, help="Image file or directory")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    main(args)