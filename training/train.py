import os
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib

from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Resolve paths relative to the project root.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(ROOT, "data", "tickets.csv")
MODEL_DIR = os.path.join(ROOT, "model")

# Keep MLflow's database and artifacts in the project directory.
mlflow.set_tracking_uri(f"sqlite:///{os.path.join(ROOT, 'mlflow.db')}")
mlflow.set_experiment("customer-ticket-classifier")

# Load the labelled dataset.
df = pd.read_csv(DATA_PATH)

X = df["text"]
y = df["category"]

# Keep examples from every category in both sets.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
    stratify=y,
)

# TF-IDF converts text to numeric features.
# Logistic Regression learns to predict the category.
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", LogisticRegression(max_iter=1000)),
])

with mlflow.start_run() as run:
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    macro_f1 = f1_score(y_test, predictions, average="macro")

    mlflow.log_param("algorithm", "LogisticRegression")
    mlflow.log_param("vectorizer", "TfidfVectorizer")
    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("macro_f1", macro_f1)

    # Log the trained model to MLflow.
    mlflow.sklearn.log_model(
        sk_model=model,
        name="ticket_classifier",
        input_example=["I was charged twice for my subscription"],
    )

    # Save a local copy for our inference server.
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, "model.joblib"))

    print(f"MLflow run ID: {run.info.run_id}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Macro F1: {macro_f1:.3f}")
    print(f"Model saved to: {MODEL_DIR}/model.joblib")
