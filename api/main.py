"""
main.py (FastAPI)
------------------
Serves the trained document classifier as a REST API.

Endpoints:
    GET  /health                  -> liveness check
    POST /predict                 -> {"document_type": ..., "confidence": ...}
    POST /predict/gradcam         -> classification + Grad-CAM heatmap image
    POST /predict/ocr             -> classification + OCR-extracted fields (cheques)

Run:
    uvicorn api.main:app --reload --port 8000
"""

import os
import sys
import shutil
import tempfile

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import FileResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config  # noqa: E402
from src.models.transfer_models import build_model  # noqa: E402
from src.dataset import EVAL_TRANSFORMS  # noqa: E402
from src.gradcam import run_gradcam  # noqa: E402
from src.ocr_extraction import extract_cheque_fields  # noqa: E402
from api.schemas import PredictionResponse, GradCamResponse, OCRResponse  # noqa: E402

from PIL import Image

app = FastAPI(
    title="Bank Document Classification & Information Extraction API",
    description="CNN + Transfer Learning document classifier with Grad-CAM explainability and OCR extraction.",
    version="1.0.0",
)

DEVICE = torch.device(config.DEVICE)
_LOADED_MODELS = {}   # cache: model_name -> nn.Module
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "custom_cnn")


def load_model(model_name):
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


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "classes": config.CLASS_NAMES}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model: str = Query(DEFAULT_MODEL, description="Which trained model to use")):
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
        return PredictionResponse(
            document_type=config.CLASS_NAMES[class_idx],
            confidence=round(confidence, 4),
            model_used=model,
        )
    finally:
        os.remove(tmp_path)


@app.post("/predict/gradcam", response_model=GradCamResponse)
async def predict_gradcam(file: UploadFile = File(...), model: str = Query(DEFAULT_MODEL)):
    if model not in config.MODEL_NAMES:
        raise HTTPException(status_code=400, detail=f"model must be one of {config.MODEL_NAMES}")

    load_model(model)  # ensures checkpoint exists / raises 503 otherwise
    tmp_path = save_upload_to_temp(file)
    try:
        save_path, predicted_class, confidence = run_gradcam(tmp_path, model)
        return GradCamResponse(
            document_type=predicted_class,
            confidence=round(confidence, 4),
            gradcam_image_path=save_path,
        )
    finally:
        os.remove(tmp_path)


@app.get("/predict/gradcam/image")
def get_gradcam_image(path: str):
    if not os.path.isfile(path) or not path.startswith(config.GRADCAM_DIR):
        raise HTTPException(status_code=404, detail="Grad-CAM image not found.")
    return FileResponse(path, media_type="image/png")


@app.post("/predict/ocr", response_model=OCRResponse)
async def predict_ocr(file: UploadFile = File(...), model: str = Query(DEFAULT_MODEL)):
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

        if doc_type == "cheque":
            fields = extract_cheque_fields(tmp_path)
        else:
            fields = {"note": f"OCR field extraction is currently implemented for 'cheque' only "
                               f"(document was classified as '{doc_type}')."}

        return OCRResponse(
            document_type=doc_type,
            confidence=round(confidence, 4),
            extracted_fields=fields,
        )
    finally:
        os.remove(tmp_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
