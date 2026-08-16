"""
main.py (FastAPI)
------------------
Serves the trained document classifier as a REST API + Web Dashboard.

Endpoints:
    GET  /                         -> Interactive Web Frontend Dashboard
    GET  /health                   -> Liveness check
    GET  /api/models               -> Available models & checkpoint status
    GET  /api/reports/comparison   -> Model benchmark comparison metrics
    GET  /api/reports/confusion-matrix/{model} -> Confusion matrix plot
    GET  /api/samples              -> Sample documents for each class
    GET  /api/samples/{cls}/{file} -> Get raw sample document
    POST /predict                  -> Classification + probabilities + latency
    POST /predict/gradcam          -> Classification + Grad-CAM heatmap overlay
    GET  /predict/gradcam/image    -> Retrieve saved Grad-CAM image
    POST /predict/ocr              -> Classification + OCR field extraction (cheques)
"""

import os
import sys
import time
import shutil
import tempfile
from typing import Dict, Any

import torch
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.models.transfer_models import build_model  # noqa: E402
from src.dataset import EVAL_TRANSFORMS  # noqa: E402
from src.gradcam import run_gradcam  # noqa: E402
from src.ocr_extraction import extract_cheque_fields  # noqa: E402
from api.schemas import PredictionResponse, GradCamResponse, OCRResponse  # noqa: E402

app = FastAPI(
    title="Bank Document Classification & Information Extraction API",
    description="CNN + Transfer Learning document classifier with Grad-CAM explainability, OCR extraction, and Web UI.",
    version="1.0.0",
)

# Enable CORS for local frontends (Angular / React / Streamlit / direct browser)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEVICE = torch.device(config.DEVICE)
_LOADED_MODELS: Dict[str, torch.nn.Module] = {}  # cache: model_name -> nn.Module
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "resnet50")


def load_model(model_name: str) -> torch.nn.Module:
    if model_name in _LOADED_MODELS:
        return _LOADED_MODELS[model_name]

    ckpt_path = os.path.join(config.MODELS_DIR, f"{model_name}.pt")
    if not os.path.exists(ckpt_path):
        raise HTTPException(
            status_code=503,
            detail=f"No trained checkpoint found for '{model_name}'. Train it first: "
                   f"python -m src.train --model {model_name}",
        )
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model = build_model(model_name, config.NUM_CLASSES, pretrained=False).to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    _LOADED_MODELS[model_name] = model
    return model


def save_upload_to_temp(upload_file: UploadFile) -> str:
    suffix = os.path.splitext(upload_file.filename or "upload.png")[1] or ".png"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    with tmp as f:
        shutil.copyfileobj(upload_file.file, f)
    return tmp.name


# ---------------------------------------------------------------------------
# Frontend Dashboard Routes
# ---------------------------------------------------------------------------
FRONTEND_DIR = os.path.join(config.BASE_DIR, "frontend")
if os.path.exists(FRONTEND_DIR):
    static_assets = os.path.join(FRONTEND_DIR, "static")
    if os.path.exists(static_assets):
        app.mount("/static", StaticFiles(directory=static_assets), name="static")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    index_file = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return HTMLResponse(
        content="""
        <html>
            <head><title>Bank Document Classifier</title></head>
            <body style="font-family:sans-serif; text-align:center; padding:50px;">
                <h1>🏦 Bank Document Classification API is Active</h1>
                <p>Swagger documentation available at <a href="/docs">/docs</a></p>
            </body>
        </html>
        """
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    trained_status = {
        m: os.path.exists(os.path.join(config.MODELS_DIR, f"{m}.pt"))
        for m in config.MODEL_NAMES
    }
    return {
        "status": "ok",
        "device": str(DEVICE),
        "classes": config.CLASS_NAMES,
        "models": config.MODEL_NAMES,
        "trained_models": trained_status,
    }


@app.get("/api/models")
def get_models():
    model_info = []
    specs = {
        "custom_cnn": {"desc": "Baseline 4-layer CNN built from scratch", "params": "~1.4M", "backbone": "Custom"},
        "mobilenet_v2": {"desc": "Lightweight mobile-optimized transfer learning", "params": "~2.2M", "backbone": "MobileNetV2"},
        "resnet50": {"desc": "Deep residual network with strong feature representation", "params": "~23.5M", "backbone": "ResNet50"},
        "efficientnet_b0": {"desc": "Compound-scaled architecture for high accuracy/FLOPs", "params": "~4.0M", "backbone": "EfficientNet-B0"},
    }
    for m in config.MODEL_NAMES:
        ckpt_path = os.path.join(config.MODELS_DIR, f"{m}.pt")
        is_trained = os.path.exists(ckpt_path)
        size_mb = round(os.path.getsize(ckpt_path) / (1024 * 1024), 2) if is_trained else 0
        model_info.append({
            "name": m,
            "trained": is_trained,
            "size_mb": size_mb,
            "specs": specs.get(m, {}),
        })
    return {"models": model_info, "default": DEFAULT_MODEL, "classes": config.CLASS_NAMES}


@app.get("/api/reports/comparison")
def get_model_comparison():
    json_path = os.path.join(config.REPORTS_DIR, "model_comparison.json")
    if os.path.exists(json_path):
        import json
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


@app.get("/api/reports/confusion-matrix/{model_name}")
def get_confusion_matrix(model_name: str):
    if model_name not in config.MODEL_NAMES:
        raise HTTPException(status_code=400, detail="Invalid model name")
    img_path = os.path.join(config.REPORTS_DIR, f"confusion_matrix_{model_name}.png")
    if not os.path.exists(img_path):
        raise HTTPException(status_code=404, detail="Confusion matrix image not found")
    return FileResponse(img_path, media_type="image/png")


@app.get("/api/samples")
def get_sample_documents():
    samples_by_class: Dict[str, list] = {}
    for c in config.CLASS_NAMES:
        class_dir = os.path.join(config.DATA_RAW_DIR, c)
        if os.path.exists(class_dir):
            files = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            # Take up to 3 samples per class
            samples_by_class[c] = files[:3]
        else:
            samples_by_class[c] = []
    return {"samples": samples_by_class}


@app.get("/api/samples/{class_name}/{filename}")
def get_sample_image(class_name: str, filename: str):
    if class_name not in config.CLASS_NAMES:
        raise HTTPException(status_code=400, detail="Invalid class name")
    safe_name = os.path.basename(filename)
    path = os.path.join(config.DATA_RAW_DIR, class_name, safe_name)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Sample image not found")
    return FileResponse(path, media_type="image/png")


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL, description="Which trained model to use"),
):
    if model not in config.MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model must be one of {config.MODEL_NAMES}")

    net = load_model(model)
    tmp_path = save_upload_to_temp(file)
    t0 = time.time()
    try:
        img = Image.open(tmp_path).convert("RGB")
        tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = net(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            class_idx = int(probs.argmax().item())
            confidence = float(probs[class_idx].item())

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        all_probs = {
            config.CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(config.CLASS_NAMES))
        }

        return PredictionResponse(
            document_type=config.CLASS_NAMES[class_idx],
            confidence=round(confidence, 4),
            model_used=model,
            probabilities=all_probs,
            inference_time_ms=elapsed_ms,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.post("/predict/gradcam", response_model=GradCamResponse)
async def predict_gradcam(
    request: Request,
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
):
    if model not in config.MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model must be one of {config.MODEL_NAMES}")

    net = load_model(model)
    tmp_path = save_upload_to_temp(file)
    try:
        # Calculate full probs
        img = Image.open(tmp_path).convert("RGB")
        tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = net(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        all_probs = {
            config.CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(config.CLASS_NAMES))
        }

        save_path, predicted_class, confidence = run_gradcam(tmp_path, model)
        gradcam_filename = os.path.basename(save_path)
        base_url = str(request.base_url).rstrip("/")
        image_url = f"{base_url}/predict/gradcam/image?path={save_path}"

        return GradCamResponse(
            document_type=predicted_class,
            confidence=round(confidence, 4),
            model_used=model,
            gradcam_image_path=save_path,
            gradcam_image_url=image_url,
            probabilities=all_probs,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/predict/gradcam/image")
def get_gradcam_image(path: str):
    normalized_path = os.path.normpath(path)
    normalized_dir = os.path.normpath(config.GRADCAM_DIR)
    if not os.path.isfile(normalized_path) or not normalized_path.startswith(normalized_dir):
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(normalized_path, media_type="image/png")


@app.post("/predict/ocr", response_model=OCRResponse)
async def predict_ocr(
    file: UploadFile = File(...),
    model: str = Query(DEFAULT_MODEL),
):
    """Phase 2: classify the document, then (if it's a cheque) run OCR field extraction."""
    if model not in config.MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model must be one of {config.MODEL_NAMES}")

    net = load_model(model)
    tmp_path = save_upload_to_temp(file)
    try:
        img = Image.open(tmp_path).convert("RGB")
        tensor = EVAL_TRANSFORMS(img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            outputs = net(tensor)
            probs = torch.softmax(outputs, dim=1)[0]
            class_idx = int(probs.argmax().item())
            confidence = float(probs[class_idx].item())
        doc_type = config.CLASS_NAMES[class_idx]

        all_probs = {
            config.CLASS_NAMES[i]: round(float(probs[i].item()), 4)
            for i in range(len(config.CLASS_NAMES))
        }

        if doc_type == "cheque":
            fields = extract_cheque_fields(tmp_path)
        else:
            fields = {
                "note": f"OCR field extraction is currently calibrated for 'cheque' documents "
                        f"(document was classified as '{doc_type}')."
            }

        return OCRResponse(
            document_type=doc_type,
            confidence=round(confidence, 4),
            model_used=model,
            extracted_fields=fields,
            probabilities=all_probs,
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
