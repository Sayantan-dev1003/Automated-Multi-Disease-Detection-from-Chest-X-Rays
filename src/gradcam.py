import tensorflow as tf
import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image

IMAGE_SIZE = 192


def make_gradcam_heatmaps(model, img_array, class_indices):
    heatmaps = {}

    # Get DenseNet backbone
    backbone = model.get_layer("densenet121")

    for idx in class_indices:
        with tf.GradientTape() as tape:
            tape.watch(img_array)

            # Forward pass through backbone
            conv_outputs = backbone(img_array, training=False)

            # Forward pass through head
            x = model.layers[-4](conv_outputs)   # GAP
            x = model.layers[-3](x)               # Dense
            x = model.layers[-2](x, training=False)  # Dropout
            preds = model.layers[-1](x)            # Output

            loss = preds[:, idx]

        # Gradients
        grads = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_outputs[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)

        heatmap = tf.maximum(heatmap, 0)
        heatmap /= tf.reduce_max(heatmap) + 1e-8

        heatmaps[idx] = heatmap.numpy()

    return heatmaps


def overlay_heatmap_on_image(image_array, heatmap, alpha=0.4):
    image = (image_array[0] * 255).astype(np.uint8)

    heatmap = cv2.resize(heatmap, (IMAGE_SIZE, IMAGE_SIZE))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay


def encode_image_to_base64(image_array):
    image = Image.fromarray(image_array)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")