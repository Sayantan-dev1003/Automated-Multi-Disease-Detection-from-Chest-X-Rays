# Automated Multi-Disease Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

A production-grade deep learning solution for automated detection of 14 thoracic diseases from chest X-ray images. This project leverages **DenseNet121**, a state-of-the-art convolutional neural network, to perform multi-label classification, identifying pathologies such as Pneumonia, Effusion, and Infiltration with high precision.

The system is designed for end-to-end usage, featuring a robust training pipeline, comprehensive evaluation scripts, a CLI for batch inference, and a high-performance **FastAPI** service for real-time deployment.

---

## 🌟 Key Features

*   **Advanced Model Architecture**: Uses `DenseNet121` pretrained on ImageNet and fine-tuned for medical imaging domain.
*   **Multi-Label Classification**: Simultaneously detects up to 14 distinct chest pathologies.
*   **Optimized Data Pipeline**: Implements `tf.data` for high-performance, asynchronous data loading and preprocessing.
*   **Production Inference**:
    *   **REST API**: Built with FastAPI for low-latency real-time prediction.
    *   **CLI Tool**: Script for batch processing of local image directories.
*   **Comprehensive Metrics**: Tracks AUC-ROC, AUC-PR, Precision, Recall, and Accuracy during training and evaluation.
*   **Configurable**: Easy-to-adjust hyperparameters via YAML configuration or command-line arguments.

---

## 📂 Project Structure

```text
.
├── artifacts/             # Storage for trained model checkpoints and label maps
├── config/                # Configuration files (e.g., hyperparameters)
├── data/                  # Layout for dataset storage
├── src/                   # Core source code
│   ├── app.py             # FastAPI application for model serving
│   ├── data_pipeline.py   # TensorFlow data loading and augmentation pipeline
│   ├── infer.py           # Command-line interface for inference
│   ├── test.py            # Evaluation script for testing model performance
│   └── train.py           # Main training loop with mixed-precision support
├── Dockerfile             # Container configuration (template)
├── requirements.txt       # Python dependency dependencies
└── README.md              # Project documentation
```

---

## 🚀 Getting Started

### Prerequisites

*   **Python 3.11+**
*   **CUDA-compatible GPU** (Highly recommended for training)

### Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chest-xray-detection.git
    cd "Automated Multi-Disease Detection from Chest X-Rays"
    ```

2.  **Create and activate a virtual environment**:
    ```bash
    # Linux/MacOS
    python3 -m venv venv
    source venv/bin/activate

    # Windows
    python -m venv venv
    venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Running the API**:
    ```bash
    uvicorn src.app:app --reload
    ```
---

## 📊 Data Preparation

This project expects a CSV manifest file for training and validation. The CSV can follow the **NIH Chest X-ray 14** format or a custom format.

**Required Columns:**
*   `image_path`: Relative path to the image file (e.g., `images/00000001_000.png`).
*   **Labels**:
    *   Option A: A `labels` column containing pipe-separated strings (e.g., `Infiltration|Pneumonia`).
    *   Option B: Separate binary columns for each disease (0/1).

**Example CSV (`train.csv`):**
```csv
image_path,labels
images/patient001.png,No Finding
images/patient002.png,Effusion|Atelectasis
images/patient003.png,Pneumonia
```

---

## 🛠️ Usage

### 1. Training

Train the model from scratch or fine-tune an existing checkpoint.

```bash
python src/train.py \
    --images_root /path/to/dataset/root \
    --train_manifest /path/to/train.csv \
    --val_manifest /path/to/val.csv \
    --output_dir ./artifacts/run_01 \
    --batch_size 16 \
    --image_size 192 \
    --epochs 10
```

**Key Arguments:**
*   `--fine_tune`: Unfreezes the top layers of the base model for fine-tuning.
*   `--resume_checkpoint`: Path to a `.h5` file to resume training.
*   `--learning_rate`: Set the initial learning rate (default: `1e-4`).

### 2. Evaluation

Evaluate a trained model on a held-out test set.

```bash
python src/test.py \
    --images_root /path/to/dataset/root \
    --test_manifest /path/to/test.csv \
    --weights ./artifacts/run_01/model.weights.h5 \
    --output_dir ./artifacts/run_01 \
    --batch_size 32
```
*Results including Accuracy, AUC-ROC, and Precision/Recall will be saved to `test_metrics.json`.*

### 3. Inference

#### 🖥️ CLI (Batch Processing)
Run predictions on a single image or an entire folder of images.

```bash
python src/infer.py \
    --input ./data/sample_xray.png \
    --weights ./artifacts/run_01/model.weights.h5 \
    --labels ./artifacts/run_01/run_metadata.json \
    --output_json predictions.json
```
*(Note: Ensure you provide the correct label mapping JSON usually generated during training or stored in artifacts).*

#### 🌐 REST API (Real-time)
Start the FastAPI server for HTTP-based inference.

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

**Test the API:**

*   **Health Check**: `GET /`
*   **Predict**: `POST /predict`

**Example cURL:**
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/chest.png"
```

---

## 🧠 Model Architecture

The solution is built on a robust deep learning foundation:

1.  **Backbone**: **DenseNet121**
    *   Selected for its parameter efficiency and strong feature propagation.
    *   Pretrained on ImageNet to leverage transfer learning.
2.  **Preprocessing**:
    *   Resize to **192x192** (Configurable).
    *   Normalization to [0,1].
3.  **Classification Head**:
    *   Global Average Pooling 2D.
    *   Dense Layer (512 units, ReLU).
    *   Dropout (0.4) for regularization.
    *   Output Layer (Sigmoid activation) for multi-label probability generation.

---

## 📄 License

This project is open-sourced under the MIT License. See the `LICENSE` file for details.
