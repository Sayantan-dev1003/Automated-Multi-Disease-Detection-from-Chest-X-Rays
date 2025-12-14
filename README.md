# Automated Multi-Disease Detection from Chest X-Rays

![Python](https://img.shields.io/badge/Python-3.12%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-orange)
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

## ✨ Key Features

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
| 🟣 _Grad-CAM Visualization_                  | On-demand multi-class explainability via API (`/predict?explain=1`)         |

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

## 📗 Model Evaluation & Metrics

The model was evaluated on held-out **validation** and **test** datasets using standard metrics for multi-label medical image classification.  
These metrics assess both the **probabilistic quality** of predictions and their **clinical usefulness** in screening scenarios.

---

### Overall Performance

| Metric                     | Validation | Test   |
|----------------------------|------------|--------|
| **ROC–AUC (Macro)**        | 0.8624     | 0.8629 |
| **Precision**              | 0.6876     | 0.7023 |
| **Recall (Sensitivity)**   | 0.2661     | 0.2917 |
| **Loss**                   | 0.2044     | 0.1995 |  

The close alignment between validation and test performance indicates **stable learning behavior and good generalization**, with no significant signs of overfitting.

---

### ROC–AUC Analysis

ROC–AUC is treated as the **primary evaluation metric** for this task due to its suitability for medical imaging problems:

- It is **threshold-independent**, evaluating ranking quality rather than hard decisions  
- It is **robust to class imbalance**, which is common in chest X-ray datasets  
- It reflects the model’s ability to separate diseased from non-diseased cases across all classes  

A macro ROC–AUC of approximately **0.86** demonstrates strong discriminative capability across multiple thoracic disease categories, which is considered a solid performance for large-scale multi-label chest X-ray classification.

---

### Precision–Recall Trade-off

At the default decision threshold of **0.5**, the model exhibits **higher precision than recall**, indicating a conservative prediction behavior:

- **High precision** reduces false-positive alerts  
- **Lower recall** implies some subtle or early-stage pathologies may be missed  

Given the screening-oriented nature of this system, inference is performed using a **lower probability threshold of 0.35**, which increases sensitivity (recall) while maintaining acceptable precision. This adjustment aligns the model’s behavior with real-world screening requirements, where missing a pathology carries greater clinical risk than generating a false alert.

---

### Confusion Matrix (Conceptual Explanation)

In a multi-label classification setting, confusion matrices are computed **independently for each disease class**, rather than as a single aggregated matrix.

For each disease, predictions fall into four categories:

- **True Positive (TP):** Disease correctly identified  
- **False Positive (FP):** Disease predicted but not present  
- **False Negative (FN):** Disease present but missed (most critical error in screening)  
- **True Negative (TN):** Correctly predicted absence of disease  

From a clinical perspective, **false negatives are prioritized for minimization**, as undetected pathologies pose higher risk than false alarms. This directly motivates the use of a lower inference threshold.

---

### Clinical Interpretation & Limitations

This system is designed as a **screening and decision-support tool**, not a diagnostic system.  
Predictions with moderate confidence are intended to **flag cases for expert radiological review**, rather than provide definitive medical conclusions.

All outputs should be interpreted in conjunction with:
- Clinical history  
- Radiologist expertise  
- Additional diagnostic tests

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

- Python **3.12+**
- TensorFlow **2.18** (see `requirements.txt`)
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
- Optional explainability: Add `?explain=1` to obtain Grad-CAM heatmaps for all detected diseases.

Sample cURL:
```bash
curl -X POST "http://localhost:8000/predict?explain=1" \
    -H "accept: application/json" \
    -H "Content-Type: multipart/form-data" \
    -F "file=@/path/to/chest.png"
```

Swagger UI also enables interactive file upload/testing! If `explain=1` is added, the JSON response will include base64-encoded Grad-CAM heatmaps for each detected disease class, suitable for direct browser display or further analysis.

Sample response (
with explanation):
```json
{
  "filename": "chest.png",
  "threshold": 0.3,
  "num_detected": 2,
  "detections": [
    {"disease": "Infiltration", "probability": 0.83,
     "grad_cam": "data:image/png;base64,iVBORw0KGg..."},
    {"disease": "Atelectasis", "probability": 0.56,
     "grad_cam": "data:image/png;base64,iVBORw0KGg..."}
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

## 🔍 Model Interpretability with Multi-Class Grad-CAM

This API supports **on-demand multi-class Grad-CAM visualization** for all detected thoracic disease categories. Simply set the `explain` query parameter when calling `/predict`:

- If `?explain=1` is provided, each detected class in the response includes a Grad-CAM heatmap.
- Heatmaps are returned as base64 PNGs for immediate GUI/app display.
- Helps clinicians, researchers, and developers interpret which image regions most influenced each prediction.

> **Note:** Grad-CAM maps are for research/interpretation only—not a diagnostic. Always corroborate with radiologist review.

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

### **Stage 1: Training DenseNet121 (Backbone Frozen)**
```bash
# TRAINING DENSENET121 MODEL - STAGE 1
!python src/train.py \
  --train_manifest /path/to/train.csv \
  --val_manifest /path/to/val.csv \
  --images_root /path/to/images_root \
  --output_dir /path/to/model_stage1 \
  --epochs 5 \
  --batch_size 16 \
  --image_size 192 \
  --learning_rate 1e-4
```

### **Stage 2: Fine Tuning DenseNet121**
```bash
# FINE TUNING DENSENET121 MODEL - STAGE 2
!python src/train.py \
  --train_manifest /path/to/train.csv \
  --val_manifest /path/to/val.csv \
  --images_root /path/to/images_root \
  --output_dir /path/to/model_stage2 \
  --epochs 6 \
  --batch_size 16 \
  --image_size 192 \
  --learning_rate 1e-5 \
  --resume_checkpoint /path/to/model_stage1/model.weights.h5 \
  --fine_tune
```

### **Testing the Final Model**
```bash
# TESTING THE MODEL ON TEST.CSV
!python src/test.py \
  --test_manifest /path/to/test.csv \
  --images_root /path/to/images_root \
  --weights /path/to/model_stage2/model.weights.h5 \
  --output_dir /path/to/test_results \
  --batch_size 32 \
  --image_size 192
```

### **CLI Inference:**
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

## 🐳 Deploying with Docker

Containerization with Docker is supported to simplify runtime environments, accelerate deployment, and ensure reproducibility across diverse systems (including cloud and on-premise).

### **Build and Run Docker Image**

Assuming you have Docker installed:

```bash
docker build -t chest-xray-api .
docker run -d -p 8000:8000 chest-xray-api
```

- The API will now be available at [http://localhost:8000](http://localhost:8000).
- Edit or extend the provided `Dockerfile` as necessary for GPU or production deployment.

**Key Benefits:**
- Fully portable—runs identically anywhere Docker is supported
- Encapsulates all dependencies
- Facilitates cloud/serverless deployment and CI/CD pipelines

---

## ⚠️ Disclaimer

- This software is intended **solely for research and screening/decision support purposes**.
- It is **not a substitute for professional medical diagnosis** or a replacement for a board-certified radiologist.
- Users are responsible for interpreting results with appropriate clinical context.
- Grad-CAM visualizations are provided for interpretability purposes only and should not be used as a diagnostic tool.

---

## 📄 License

This project is distributed under the terms of the MIT License. See the `LICENSE` file for full details.

---

**For questions, contributions, or collaboration, please open an issue or pull request.**