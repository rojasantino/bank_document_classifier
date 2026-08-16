from typing import Optional, Dict, Any
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    document_type: str
    confidence: float
    model_used: str
    probabilities: Optional[Dict[str, float]] = None
    inference_time_ms: Optional[float] = None


class GradCamResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    document_type: str
    confidence: float
    model_used: Optional[str] = None
    gradcam_image_path: str
    gradcam_image_url: Optional[str] = None
    probabilities: Optional[Dict[str, float]] = None


class OCRResponse(BaseModel):
    model_config = {"protected_namespaces": ()}
    document_type: str
    confidence: float
    model_used: Optional[str] = None
    extracted_fields: Dict[str, Any]
    probabilities: Optional[Dict[str, float]] = None
