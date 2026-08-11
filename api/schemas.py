from typing import Optional, Dict
from pydantic import BaseModel


class PredictionResponse(BaseModel):
    document_type: str
    confidence: float
    model_used: str


class GradCamResponse(BaseModel):
    document_type: str
    confidence: float
    gradcam_image_path: str


class OCRResponse(BaseModel):
    document_type: str
    confidence: float
    extracted_fields: Dict[str, Optional[str]]
