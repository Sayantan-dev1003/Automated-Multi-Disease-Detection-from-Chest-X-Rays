#!/usr/bin/env python3
"""
src/app.py
FastAPI inference service for Chest X-ray multi-label classification
"""

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import io
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import JSONResponse
from gradcam import (
    make_gradcam_heatmaps,
    overlay_heatmap_on_image,
    encode_image_to_base64
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"

WEIGHTS_PATH = ARTIFACTS_DIR / "final_model.weights.h5"
LABELS_PATH = ARTIFACTS_DIR / "label_map.json"
IMAGE_SIZE = 192

# Load label names
with open(LABELS_PATH, "r") as f:
    label_map = json.load(f)

LABEL_NAMES = label_map["labels"]
NUM_CLASSES = label_map["num_classes"]

# Build model (same as training)
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


# Load model once (startup)
model = build_model(
    input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
    num_classes=NUM_CLASSES
)
model.load_weights(WEIGHTS_PATH)

# FastAPI app
app = FastAPI(title="Chest X-ray Multi-Disease Detection API")

# Utils
def preprocess_image_bytes(image_bytes: bytes):
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = tf.expand_dims(img, axis=0)
    return img

# Health check
@app.get("/")
def health():
    return {"status": "ok", "model_loaded": True}

# Prediction endpoint
@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    explain: bool = Query(False, description="Generate Grad-CAM visualizations")
):
    image_bytes = await file.read()
    img = preprocess_image_bytes(image_bytes)

    THRESHOLD = 0.3
    preds = model.predict(img, verbose=0)[0]

    detected = [
        {
            "index": i,
            "disease": LABEL_NAMES[i],
            "probability": float(preds[i])
        }
        for i in range(NUM_CLASSES)
        if preds[i] >= THRESHOLD
    ]

    detected = sorted(
        detected,
        key=lambda x: x["probability"],
        reverse=True
    )

    response = {
        "filename": file.filename,
        "threshold": THRESHOLD,
        "num_detected": len(detected),
        "detections": [
            {
                "disease": d["disease"],
                "probability": d["probability"]
            }
            for d in detected
        ]
    }

    if explain and detected:
        class_indices = [d["index"] for d in detected]

        heatmaps = make_gradcam_heatmaps(
            model,
            img,
            class_indices=class_indices
        )

        gradcam_results = {}

        for d in detected:
            heatmap = heatmaps.get(d["index"])
            if heatmap is not None:
                overlay = overlay_heatmap_on_image(img.numpy(), heatmap)
                gradcam_results[d["disease"]] = encode_image_to_base64(overlay)

        response["gradcam"] = gradcam_results

    return JSONResponse(response)