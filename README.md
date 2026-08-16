#  AI-Powered Bank Document Classification & Information Extraction using CNN and OCR

---

## 1. Project Overview 

 This project builds an end-to-end Computer Vision + Document AI system that automatically classifies scanned banking documents (cheque, bank statement, deposit slip, withdrawal slip, KYC document, account opening form, card document, other financial documents) using CNN and Transfer Learning models, explains its predictions using Grad-CAM, and — for cheques — extracts structured fields (date, payee, amount, account number, IFSC) using OCR. It is served as a production-style REST API with Docker packaging.

---

## 2. Architecture 

```
Bank Document Images
        ↓
Data Preprocessing (denoise, deskew, auto-crop)
        ↓
Data Augmentation (rotation, translation, color jitter)
        ↓
CNN Baseline  +  Transfer Learning (MobileNetV2 / ResNet50 / EfficientNetB0)
        ↓
Model Evaluation (Accuracy, F1, Inference Time)
        ↓
Grad-CAM Explainability
        ↓
FastAPI  →  Docker
        ↓
OCR Phase 2 (cheque field extraction)
```

---

## 3. Folder Structure 

```
bank_document_classifier/
│
├── config.py                        # Central config: paths, class names, hyperparameters
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container build definition
├── docker-compose.yml               # One-command local Docker run
├── .dockerignore
├── run_pipeline.sh                  # Runs the ENTIRE pipeline in one command
│
├── data/
│   ├── generate_synthetic_data.py   # Creates synthetic training images (8 classes)
│   ├── raw/                         # Original images, organized by class folder
│   │   ├── cheque/
│   │   ├── bank_statement/
│   │   ├── card_document/
│   │   ├── deposit_slip/
│   │   ├── withdrawal_slip/
│   │   ├── account_opening_form/
│   │   ├── kyc_document/
│   │   └── other_financial_document/
│   └── processed/                   # Denoised / deskewed / cropped images (auto-generated)
│
├── src/
│   ├── data_preprocessing.py        # OpenCV: denoise, deskew, auto-crop, resize/normalize
│   ├── dataset.py                   # PyTorch Dataset, augmentation, train/val/test split
│   ├── train.py                     # Training loop (works for any of the 4 models)
│   ├── evaluate.py                  # Accuracy / F1 / inference-time comparison + confusion matrix
│   ├── gradcam.py                   # Grad-CAM heatmap generation
│   ├── ocr_extraction.py            # Phase 2: OCR field extraction for cheques
│   └── models/
│       ├── custom_cnn.py            # Baseline CNN built from scratch
│       └── transfer_models.py       # MobileNetV2 / ResNet50 / EfficientNetB0 wrappers
│
├── api/
│   ├── main.py                      # FastAPI app: /predict, /predict/gradcam, /predict/ocr
│   └── schemas.py                   # Pydantic response models
│
├── outputs/
│   ├── models/                      # Trained checkpoints (<model_name>.pt)
│   ├── logs/                        # Per-epoch training history (JSON)
│   ├── reports/                     # model_comparison.md/json, confusion matrices
│   └── gradcam/                     # Saved Grad-CAM heatmap images
│
└── tests/                           # (add your own unit tests here)
```

---

## 4. Prerequisites

- Python 3.10+
- pip
- Tesseract OCR engine installed on your system (for the OCR module)
  - Ubuntu/Debian: `sudo apt-get install tesseract-ocr`
  - Mac: `brew install tesseract`
  - Windows: install from https://github.com/UB-Mannheim/tesseract/wiki and add to PATH
- (Optional) NVIDIA GPU + CUDA for faster training. CPU works fine too, just slower.
- (Optional) Docker, if you want to run it as a container.


---

## 5. Step-by-Step: Run Locally 

### Step 1 — Install dependencies 

```bash
cd bank_document_classifier
pip install -r requirements.txt
```

If `pip` complains about externally-managed environments (common on Ubuntu 24+):
```bash
pip install -r requirements.txt --break-system-packages
```

### Step 2 — Add your data

**Option A — Use your own scanned documents (recommended for real results):**
Place images into `data/raw/<class_name>/` folders, matching the 8 class names above.
```
data/raw/cheque/my_cheque_001.jpg
data/raw/bank_statement/statement_001.jpg
...
```

**Option B — Generate synthetic data (to test the whole pipeline immediately):**
```bash
python3 data/generate_synthetic_data.py --per_class 60
```
This creates 60 synthetic images per class (480 total) with realistic layouts for cheques, statements, KYC forms, etc. Useful for validating the pipeline before you have a real dataset, or for a quick demo.

### Step 3 — Preprocess the images 

```bash
python3 src/data_preprocessing.py
```
This denoises, deskews (fixes scan rotation), and auto-crops every image in `data/raw/`, saving the result to `data/processed/`. Training automatically prefers `data/processed/` if it exists.

### Step 4 — Train the models

Train a single model:
```bash
python3 -m src.train --model custom_cnn --epochs 8
python3 -m src.train --model mobilenet_v2 --epochs 8
python3 -m src.train --model resnet50 --epochs 8
python3 -m src.train --model efficientnet_b0 --epochs 8
```

Or train all 4 in one go:
```bash
python3 -m src.train --model all --epochs 8
```

**No internet access for pretrained ImageNet weights?** Add `--no_pretrained` to train transfer-learning models from random initialization instead:
```bash
python3 -m src.train --model all --epochs 8 --no_pretrained
```

**No GPU?** Force CPU mode:
```bash
FORCE_CPU=1 python3 -m src.train --model custom_cnn --epochs 8
```

Checkpoints are saved to `outputs/models/<model_name>.pt`, and per-epoch history to `outputs/logs/<model_name>_history.json`.

### Step 5 — Evaluate & compare models

```bash
python3 -m src.evaluate
```
This produces:
- `outputs/reports/model_comparison.md` — the Accuracy / F1 / Inference-Time table
- `outputs/reports/confusion_matrix_<model_name>.png` — per-model confusion matrix
- Full precision/recall/F1 classification report printed to console

### Step 6 — Generate a Grad-CAM explanation

```bash
python3 -m src.gradcam --image data/raw/cheque/cheque_0000.png --model resnet50
```
Saves a heatmap overlay to `outputs/gradcam/` showing which regions of the document (e.g. signature, MICR line, amount box) drove the prediction.

### Step 7 — Extract cheque fields with OCR

```bash
python3 -m src.ocr_extraction --image data/raw/cheque/cheque_0000.png
```
Prints extracted date, payee, amount, account number, and IFSC code.

### Step 8 — Run everything in one command 

```bash
bash run_pipeline.sh --epochs 8 --per-class 60
```
This runs Steps 2 (synthetic data, if `data/raw` is empty) through 5 automatically.

### Step 9 — Start the API

```bash
uvicorn api.main:app --reload --port 8000
```
Open http://127.0.0.1:8000/docs for the interactive Swagger UI.

Test with curl:
```bash
# Classify a document
curl -X POST "http://127.0.0.1:8000/predict?model=resnet50" \
     -F "file=@data/raw/cheque/cheque_0000.png"

# Classify + Grad-CAM explanation
curl -X POST "http://127.0.0.1:8000/predict/gradcam?model=resnet50" \
     -F "file=@data/raw/cheque/cheque_0000.png"

# Classify + OCR field extraction (cheques)
curl -X POST "http://127.0.0.1:8000/predict/ocr?model=resnet50" \
     -F "file=@data/raw/cheque/cheque_0000.png"
```

Example response from `/predict`:
```json
{
  "document_type": "cheque",
  "confidence": 0.964,
  "model_used": "resnet50"
}
```

### Step 10 — Run with Docker 

```bash
docker compose up --build
```
The API will be available at http://localhost:8000. The `docker-compose.yml` mounts your local `outputs/` and `data/` folders into the container, so you don't need to retrain inside Docker — just train locally first, then launch the container to serve it.

run from frontend:
```bash
frontend/ python streamlit_app.py
```
