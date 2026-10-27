# MLOps Ticket Classifier

A small MLOps project that classifies customer support tickets
as Billing, Technical, or Refund.

## Components

- Python and scikit-learn: model training
- MLflow: experiment tracking and model logging
- FastAPI: local prediction API
- Docker: model-serving container
- Minikube and KServe: Kubernetes-based model serving

## Train the model

```bash
pip install -r requirements.txt
python training/train.py
```

## Run the prediction API

```bash
pip install -r inference/requirements.txt
uvicorn inference.server:app --reload
```

The trained model must exist at model/model.joblib first.

## Test the API

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice"}'
```
