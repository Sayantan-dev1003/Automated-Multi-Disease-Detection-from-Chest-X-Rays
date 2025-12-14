# Automated Multi-Disease Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.11%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 🩺 Detailed Description

Automated Multi-Disease Detection from Chest X-Rays is a robust, production-grade deep learning pipeline that detects 14 common thoracic diseases directly from chest radiographs (CXR). Designed for both researchers and engineers, the solution leverages a powerful DenseNet121 backbone—pretrained on ImageNet—for state-of-the-art performance on real-world clinical data.

#### The system supports:
- **Training** and **fine-tuning** with scalable data pipelines
- Extensive **evaluation metrics**
- High-speed **batch/CLI inference**
- Easy-to-deploy **FastAPI** for real-time and remote predictions

---

## 🩻 Problem Statement

Millions of chest X-rays are performed globally, but radiologist availability is limited, resulting in delays and missed pathologies. There is a critical need for automated, scalable, and accurate multi-disease analysis from CXR images to support radiologists and improve clinical outcomes through AI-powered screening.

---

## 🎯 Key Objectives

1. **Accurate Detection**: Simultaneous multi-label classification of 14 thoracic diseases.
2. **Production-Ready**: Provide deployment pipelines (API, CLI) suitable for hospitals and cloud inference.
3. **Easy Customization**: Enable users to quickly adapt/configure for new datasets.
4. **Medical Interpretability**: Output actionable, interpretable probability scores for each disease.
5. **Open Source**: Foster research and real-world adoption through transparent, reproducible practices.

---

## ✨ Key Features (1-liners)

| Feature                                      | Description                                                                |
|----------------------------------------------|----------------------------------------------------------------------------|
| 🚀 _Transfer Learning (DenseNet121)_         | State-of-the-art pretrained model for high accuracy                        |
| 🏷️  _Multi-label Output_                     | Predict all 14 NIH ChestX-ray pathologies per image                        |
| ⚡ _FastAPI Real-time Service_               | Instant, RESTful API for hospital/remote deployment                        |
| 🖥️  _Batch CLI Tool_                         | Inference on entire directories or single images                           |
| 📈 _Advanced Metrics & Logging_              | ROC, PR, accuracy, precision/recall metrics tracked per epoch              |
| ⚙️ _Configurable via YAML/CLI_               | Adjust image size, batch size, learning rate, and more easily              |
| 🔒 _Patient-wise Data Split Utility_         | Prevents data leakage during training/validation                           |
| 🗄️ _Sampleizable & Extensible Data Loaders_  | tf.data pipelines for robust, fast loading and augmentation                |
| 🏥 _Interpretability_                        | Directly output interpretable per-disease probabilities                    |

---

## 🧑‍⚕️ Medical Interpretation Strategy

The model outputs **independent probabilities (sigmoid)** for each disease.

| Probability Range | Interpretation                                 |
|-------------------|------------------------------------------------|
| `< 0.3`           | Likely normal / disease unlikely               |
| `0.3 – 0.5`       | Suspicious / possible early-stage abnormality  |
| `≥ 0.5`           | High confidence disease presence               |

> **Note:** This AI tool is for **screening and decision support only**—NOT for clinical diagnosis. Always confirm findings with a qualified radiologist.

---

## 🏗️ System Architecture

```text
+---------------------+
|  Input  (PNG, JPG)  |
+----------+----------+
           |
           v
+------------------------------+
|  tf.data Preprocessing       |
|  Resize, Normalize, Augment  |
+------------------------------+
           |
           v
+------------------------------+
|     DenseNet121 Backbone     |
| (Pretrained, fine-tunable)   |
+------------------------------+
           |
           v
| → GlobalAvgPool2D → Dense(512, ReLU) → Dropout(0.4) → Dense(14, Sigmoid)
|
+------------------------------+
|      Output Probabilities    |
|  (0–1 for each pathology)    |
+------------------------------+
           |
           v
+------------------------------+
|   Decision Support Layer     |
| (Threshold-based reporting,  |
|  JSON output, REST API, etc) |
+------------------------------+
```

---

## 📂 Project Repository Structure (Full)

```text
.
├── artifacts/                           # Model weights, label maps, metrics, run logs
│   ├── final_model.weights.h5           # Main trained weights (Keras)
│   └── label_map.json                   # List of disease classes
├── config/
│   └── default.yaml                     # Default experiment/training configuration
├── data/
│   └── sample/
│       ├── images/                      # Tiny sample images for testing
│       ├── sample_manifest.csv          # Example manifest
│       └── README.md                    # Sample data usage guidelines
├── src/
│   ├── app.py                           # FastAPI inference application (real-time)
│   ├── infer.py                         # Command line inference tool (batch/single)
│   ├── test.py                          # Model evaluation (testing) script
│   ├── train.py                         # End-to-end multi-label training loop
│   ├── data_pipeline.py                 # tf.data loading and augmentation utilities
│   └── data/
│       ├── build_manifest.py            # Build manifest from NIH/open datasets
│       ├── make_splits.py               # Patient-wise train/val/test splitting
│       └── preview.py                   # CSV preview/sanity checker
├── Dockerfile                           # (Optional) Container config
├── requirements.txt                     # All core Python dependencies
└── README.md                            # You are here
```

---

## ⚡️ Getting Started

### Prerequisites

- Python **3.11+**
- TensorFlow **2.x** (see `requirements.txt`)
- FastAPI, Uvicorn, Pandas, NumPy, Scikit-Learn, etc.
- CUDA-compatible GPU **(recommended for training)**
- NIH Chest X-ray 14 Dataset (official [download](https://nihcc.app.box.com/v/ChestXray-NIHCC))

---

### 🛠️ Installation & Environment Setup

#### 1. Clone the repository

```bash
git clone https://github.com/Sayantan-dev1003/Automated-Multi-Disease-Detection-from-Chest-X-Rays.git
cd "Automated Multi-Disease Detection from Chest X-Rays"
```

#### 2. Create & activate a virtual environment

**Linux/MacOS:**

```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows:**

```cmd
python -m venv venv
venv\Scripts\activate
```

#### 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

### 🚀 Running the API Server

```bash
uvicorn src.app:app --host 0.0.0.0 --port 8000
```

- The server will start at: http://localhost:8000

#### API Usage

- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Redoc:** [http://localhost:8000/redoc](http://localhost:8000/redoc)

**1. Health Check**  
`GET /`

Response:
```json
{
  "status": "ok",
  "model_loaded": true
}
```

**2. Prediction Endpoint**  
`POST /predict` (multipart/form-data, field: `file`)

Sample cURL:
```bash
curl -X POST "http://localhost:8000/predict" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/chest.png"
```

Swagger UI also enables interactive file upload testing!

Sample response:
```json
{
  "filename": "chest.png",
  "threshold": 0.3,
  "num_detected": 2,
  "detections": [
    {"disease": "Infiltration", "probability": 0.83},
    {"disease": "Atelectasis", "probability": 0.56}
  ]
}
```

---

## 🧮 Threshold Design Choice

- The default decision threshold for reporting a disease is **0.3** (sigmoid output).
- This value is chosen for high sensitivity in screening settings, lowering missed cases.
- For probabilities:
    - `< 0.3`: Normal/low risk; very unlikely presence of disease
    - `0.3–0.5`: Suspicious or borderline; early/intermediate risk
    - `≥ 0.5`: High confidence/strong disease evidence

**Clinical Rationale:**  
Screening tools must minimize false negatives. A lower threshold identifies subtle or early abnormalities, but results should be correlated with expert radiology review.

---

## 🧠 Complete Model Details

- **Backbone:** DenseNet121 (ImageNet pre-trained, optionally fine-tuned)
- **Input:** 192×192 RGB images (configurable)
- **Preprocessing:** tf.data pipeline, normalization, augmentation
- **Head:** GlobalAvgPool2D → Dense(512, ReLU) → Dropout(0.4) → Dense(14, Sigmoid)
- **Loss:** Binary cross-entropy (multi-label)
- **Metrics:** AUC-ROC, AUC-PR, accuracy, precision, recall
- **Checkpoints:** Saved during training by validation AUC-ROC

---

## 🏥 Training & Data Preparation

- Recommended to use the **NIH ChestXray14 Dataset**. Download from [official source](https://nihcc.app.box.com/v/ChestXray-NIHCC).
- If no local GPU, use **Kaggle Notebooks** or similar cloud GPU platforms.

### **Build Manifest & Data Splits**
```bash
python src/data/build_manifest.py --labels_csv /path/to/labels.csv --images_root /path/to/images/ --out data/manifest.csv

python src/data/make_splits.py --manifest data/manifest.csv --out_dir data/splits --test_size 0.15 --val_size 0.10
```

### **Sample data available:**  
See `data/sample/` for demo images, manifest, and usage.

---

## 🏃‍♂️ Training, Testing, Inference (CLI)

**Train:**
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

**Test:**
```bash
python src/test.py \
    --images_root /path/to/dataset/root \
    --test_manifest /path/to/test.csv \
    --weights ./artifacts/run_01/model.weights.h5 \
    --output_dir ./artifacts/run_01 \
    --batch_size 32
```

**CLI Inference:**
```bash
python src/infer.py \
    --input ./data/sample_xray.png \
    --weights ./artifacts/run_01/model.weights.h5 \
    --labels ./artifacts/run_01/run_metadata.json \
    --output_json predictions.json
```

---

## 📁 Refer to Data Folder

- `data/sample/images/`: Demo images for testing
- `data/sample/sample_manifest.csv`: Example manifest format (image_path, labels)
- Use provided Python scripts in `src/data/` for manifest building, preview, and splitting

---

## ⭐ Suggestions for Kaggle

If you lack a local GPU:
- Use **Kaggle Notebooks** for free GPU acceleration.
- Upload the NIH dataset and run the code and scripts exactly as above.
- Adjust `config/default.yaml` for Kaggle paths (see sample).

---

## ⚠️ Disclaimer

- This software is intended **solely for research and screening/decision support purposes**.
- It is **not a substitute for professional medical diagnosis** or a replacement for a board-certified radiologist.
- Users are responsible for interpreting results with appropriate clinical context.

---

## 📄 License

This project is distributed under the terms of the MIT License. See the `LICENSE` file for full details.

---

**For questions, contributions, or collaboration, please open an issue or pull request.**
