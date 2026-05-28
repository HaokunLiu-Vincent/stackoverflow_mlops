import os
import torch
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Initialize FastAPI app
app = FastAPI(
    title="Financial News Sentiment API",
    description="Predict sentiment (Positive/Negative) of financial news text",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and tokenizer
MODEL_DIR = os.getenv("MODEL_DIR", "model")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
model.eval()

# Label mapping
LABELS = {0: "Negative", 1: "Positive"}


class PredictionRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "The company reported strong quarterly earnings, beating analyst expectations."
            }
        }


class PredictionResponse(BaseModel):
    text: str
    sentiment: str
    confidence: float
    probabilities: dict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="healthy",
        model_loaded=model is not None
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Tokenize input
    inputs = tokenizer(
        request.text,
        truncation=True,
        padding="max_length",
        max_length=256,
        return_tensors="pt"
    )

    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).numpy()[0]

    predicted_label = int(np.argmax(probs))
    confidence = float(probs[predicted_label])

    return PredictionResponse(
        text=request.text,
        sentiment=LABELS[predicted_label],
        confidence=round(confidence, 4),
        probabilities={
            "Negative": round(float(probs[0]), 4),
            "Positive": round(float(probs[1]), 4)
        }
    )