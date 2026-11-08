"""Small FastAPI service for safe image quality checks and model status.

This service deliberately does not load the unsuccessful DermNet V2 checkpoint.
Prediction and Grad-CAM stay disabled until a suitable, independently validated
model is available.
"""

from __future__ import annotations

import os
from io import BytesIO

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000
MIN_IMAGE_SIDE = 64
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP"}
DERMNET_CATEGORIES = [
    "Acne and Rosacea Photos", "Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions",
    "Atopic Dermatitis Photos", "Bullous Disease Photos", "Cellulitis Impetigo and other Bacterial Infections",
    "Eczema Photos", "Exanthems and Drug Eruptions", "Hair Loss Photos Alopecia and other Hair Diseases",
    "Herpes HPV and other STDs Photos", "Light Diseases and Disorders of Pigmentation",
    "Lupus and other Connective Tissue diseases", "Melanoma Skin Cancer Nevi and Moles",
    "Nail Fungus and other Nail Disease", "Poison Ivy Photos and other Contact Dermatitis",
    "Psoriasis pictures Lichen Planus and related diseases", "Scabies Lyme Disease and other Infestations and Bites",
    "Seborrheic Keratoses and other Benign Tumors", "Systemic Disease", "Tinea Ringworm Candidiasis and other Fungal Infections",
    "Urticaria Hives", "Vascular Tumors", "Vasculitis Photos", "Warts Molluscum and other Viral Infections",
]

app = FastAPI(
    title="DermaVision Research API",
    version="0.1.0",
    description="Research interface with image-quality checks. This is not a diagnostic service.",
)
origins = [item.strip() for item in os.getenv("DERMAVISION_CORS_ORIGINS", "http://localhost:5173").split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "DermaVision Research API"}


@app.get("/api/model/status")
def model_status() -> dict[str, object]:
    return {
        "available": False,
        "prediction_enabled": False,
        "reason": "The existing DermNet V2 experiment collapsed to one validation class and is not suitable for screening predictions. No HAM10000 model is available.",
        "supported_research_categories": {"DermNet": len(DERMNET_CATEGORIES), "HAM10000": 7},
        "clinical_validation": False,
    }


@app.get("/api/categories/dermnet")
def dermnet_categories() -> dict[str, object]:
    return {"categories": DERMNET_CATEGORIES, "prediction_enabled": False, "meaning": "Dataset taxonomy only; these are not predictions."}


def inspect_image_quality(data: bytes) -> dict[str, object]:
    """Decode an upload in memory and return basic, non-diagnostic quality cues."""
    if not data:
        raise ValueError("The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("The image exceeds the 10 MB upload limit.")
    try:
        with Image.open(BytesIO(data)) as source:
            image_format = (source.format or "").upper()
            if image_format not in ALLOWED_FORMATS:
                raise ValueError("Use a JPEG, PNG, WebP, or BMP image.")
            source.verify()
        with Image.open(BytesIO(data)) as source:
            source = source.convert("RGB")
            width, height = source.size
            if width * height > MAX_IMAGE_PIXELS:
                raise ValueError("The image dimensions exceed the 20-megapixel safety limit.")
            # Bound analysis memory for high-resolution phone images while
            # reporting their original dimensions above.
            analysis_image = source.copy()
            analysis_image.thumbnail((512, 512), Image.Resampling.LANCZOS)
            rgb = np.asarray(analysis_image, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise ValueError("The file could not be decoded as a valid image.") from exc

    gray = np.dot(rgb[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    # Laplacian variance is a rough blur cue; it is not a medical image-quality standard.
    padded = np.pad(gray, 1, mode="edge")
    laplacian = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
        - 4 * gray
    )
    blur_score = float(np.var(laplacian))
    brightness = float(np.mean(gray) / 255.0)
    warnings: list[str] = []
    if min(width, height) < MIN_IMAGE_SIDE:
        warnings.append(f"Small image: each side should be at least {MIN_IMAGE_SIDE} pixels for this check.")
    if blur_score < 20:
        warnings.append("The image may be blurry; retake it in focus if appropriate.")
    if brightness < 0.08 or brightness > 0.92:
        warnings.append("The image may be unusually dark or bright.")
    return {
        "format": image_format,
        "width": width,
        "height": height,
        "channels": "RGB (3 channels)",
        "blur_score_laplacian_variance": round(blur_score, 3),
        "mean_brightness_0_to_1": round(brightness, 4),
        "warnings": warnings,
        "quality_check_only": True,
        "disclaimer": "These simple image checks do not assess skin health and do not guarantee that an image is suitable for screening.",
    }


@app.post("/api/image/quality")
async def image_quality(image: UploadFile = File(...)) -> dict[str, object]:
    data = await image.read(MAX_UPLOAD_BYTES + 1)
    try:
        result = inspect_image_quality(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result


@app.post("/api/predict")
async def predict() -> None:
    raise HTTPException(
        status_code=503,
        detail="Screening predictions are unavailable: no suitable validated model is currently enabled.",
    )


@app.post("/api/gradcam")
async def gradcam() -> None:
    raise HTTPException(status_code=503, detail="Grad-CAM is unavailable until a suitable model is available.")
