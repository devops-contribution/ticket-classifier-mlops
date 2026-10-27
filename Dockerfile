FROM python:3.12-slim

WORKDIR /app

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY inference/ ./inference/
COPY model/model.joblib ./model/model.joblib

EXPOSE 8000

CMD ["uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8000"]
