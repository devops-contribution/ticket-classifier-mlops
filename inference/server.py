import os
import joblib

from fastapi import FastAPI
from pydantic import BaseModel

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "model",
        "model.joblib",
    ),
)

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Customer Ticket Classifier")


class TicketRequest(BaseModel):
    text: str


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(request: TicketRequest):
    category = model.predict([request.text])[0]

    return {
        "text": request.text,
        "category": category,
    }
