# Ticket Classifier MLOps

A support-ticket text classifier (scikit-learn) served on Kubernetes with [KServe](https://kserve.github.io/website/). This guide takes you from zero to a working prediction on your own machine using **minikube**.

```
 your laptop                         minikube cluster
┌────────────┐   kubectl cp    ┌──────────────────────────────┐
│ model.joblib│ ─────────────▶ │ PVC (my-model-pvc)           │
└────────────┘                 │   └─ my-model/model.joblib   │
                               │            │ mounted          │
 curl :8080  ◀── port-forward ──│  KServe predictor pod        │
                               │  (sklearn server, "my-model") │
                               └──────────────────────────────┘
```

## What you will end up with

- A local Kubernetes cluster (minikube)
- KServe installed in **RawDeployment** mode (no Knative or Istio needed)
- Your model stored on a PersistentVolumeClaim
- An `InferenceService` called `my-model` that answers prediction requests

---

## 1. Prerequisites

| Tool | Why | Check |
|------|-----|-------|
| Docker | Runs the minikube node | `docker --version` |
| minikube | Local Kubernetes cluster | `minikube version` |
| kubectl | Talks to the cluster | `kubectl version --client` |
| Python 3.9+ | Train/export the model | `python3 --version` |

Install steps (Linux x86_64; macOS: `brew install minikube kubectl`; Windows: `winget install Kubernetes.minikube Kubernetes.kubectl`):

```bash
# minikube
curl -LO https://github.com/kubernetes/minikube/releases/latest/download/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube && rm minikube-linux-amd64

# kubectl
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl && rm kubectl

# let your user run Docker without sudo (log out/in or run newgrp)
sudo usermod -aG docker $USER && newgrp docker
```

---

## 2. Python environment

```bash
cd ~/ticket-classifier-mlops
python3 -m venv venv
source venv/bin/activate          # Windows PowerShell: .\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

### Important: how the model must be saved

KServe's built-in sklearn server loads the file with `joblib`, and it will be given **raw text**. So save a full **Pipeline** (vectorizer + classifier), not just the classifier:

```python
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression(max_iter=1000)),
])
pipeline.fit(X_train, y_train)

joblib.dump(pipeline, "model.joblib")   # filename must be model.joblib / model.pkl / model.pickle
```

Tip: train with a scikit-learn version close to the one in the KServe sklearn image, otherwise loading can fail.

---

## 3. Start the cluster

```bash
minikube start --driver=docker --cpus=4 --memory=8192
kubectl get nodes                  # should show STATUS Ready
```

---

## 4. Install KServe

KServe needs cert-manager first.

```bash
# 1. cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.16.2/cert-manager.yaml
kubectl wait --for=condition=Available deployment --all -n cert-manager --timeout=180s

# 2. KServe (server-side apply is required: the CRDs are large)
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve.yaml
kubectl wait --for=condition=Available deployment --all -n kserve --timeout=180s
kubectl apply --server-side -f https://github.com/kserve/kserve/releases/download/v0.15.0/kserve-cluster-resources.yaml

# 3. Use plain Kubernetes Deployments instead of Knative/Istio
kubectl patch configmap/inferenceservice-config -n kserve --type=strategic \
  -p '{"data": {"deploy": "{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
```

Verify:

```bash
kubectl get crd | grep serving.kserve.io     # InferenceService CRD must be listed
kubectl get pods -n kserve                   # controller pod Running
```

> If you ever see `no matches for kind "InferenceService"`, this step did not finish.

---

## 5. Create storage and upload the model

### 5.1 Create the PVC

Save as `k8s/pvc.yaml`:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: my-model-pvc
spec:
  accessModes: ["ReadWriteOnce"]
  resources:
    requests:
      storage: 1Gi
```

```bash
kubectl apply -f k8s/pvc.yaml
kubectl get pvc
```

### 5.2 Start a temporary helper pod that mounts the PVC

Save as `k8s/helper-pod.yaml`:

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: helper-pod
spec:
  restartPolicy: Never
  containers:
    - name: loader
      image: busybox:1.36
      command: ["sleep", "3600"]
      volumeMounts:
        - name: model-store
          mountPath: /mnt/models
  volumes:
    - name: model-store
      persistentVolumeClaim:
        claimName: my-model-pvc
```

```bash
kubectl apply -f k8s/helper-pod.yaml
kubectl wait --for=condition=Ready pod/helper-pod --timeout=120s
```

### 5.3 Copy the model in

The folder layout on the PVC must match the `storageUri` in `k8s/isvc.yaml` (`pvc://my-model-pvc/my-model`):

```
my-model-pvc/
└── my-model/
    └── model.joblib
```

```bash
kubectl exec helper-pod -- mkdir -p /mnt/models/my-model
kubectl cp ./model.joblib helper-pod:/mnt/models/my-model/model.joblib
kubectl exec helper-pod -- find /mnt/models
```

Expected output:

```
/mnt/models
/mnt/models/my-model
/mnt/models/my-model/model.joblib
```

### 5.4 Remove the helper pod

```bash
kubectl delete pod helper-pod
```

Deleting the pod does **not** delete the PVC or your model.

---

## 6. Deploy the model

`k8s/isvc.yaml`:

```yaml
apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: my-model
  annotations:
    serving.kserve.io/deploymentMode: RawDeployment
spec:
  predictor:
    model:
      modelFormat:
        name: sklearn
      storageUri: pvc://my-model-pvc/my-model
```

```bash
kubectl apply -f k8s/isvc.yaml
kubectl get isvc                       # wait for READY = True
kubectl get pods
```

---

## 7. Send a prediction

In **terminal 1**, forward the service (leave it running):

```bash
kubectl port-forward svc/my-model-predictor 8080:80
```

In **terminal 2**:

```bash
# Is the model loaded?
curl http://127.0.0.1:8080/v1/models/my-model
# -> {"name":"my-model","ready":true}

# Predict
curl -X POST http://127.0.0.1:8080/v1/models/my-model:predict \
  -H "Content-Type: application/json" \
  -d '{"instances": ["I was charged twice"]}'
# -> {"predictions":["billing"]}   (your labels will differ)
```

Note: the built-in sklearn server uses the KServe path `/v1/models/<name>:predict` and an `instances` list. It does **not** serve a custom `/predict` route with `{"text": ...}`. For that, build a custom container (see "Custom API" below).

---

## 8. Everyday commands

```bash
kubectl get pods                                        # what is running
kubectl logs -f deploy/my-model-predictor               # follow logs
kubectl describe pod <pod-name>                         # why is it stuck?
kubectl get events --sort-by=.lastTimestamp            # recent cluster events
kubectl rollout restart deployment my-model-predictor   # reload the model after changing files

minikube stop       # pause the cluster, keep everything
minikube start      # resume
minikube delete     # destroy everything and start over
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|------|-----|
| `no matches for kind "InferenceService"` | KServe CRDs not installed | Redo section 4 |
| `Model with name my-model does not exist` (404) | Server started but found no model file | Check `kubectl exec deploy/my-model-predictor -- ls -la /mnt/models`; `model.joblib` must be there. Then restart the predictor |
| Logs say `failed to locate model file ... under dir /mnt/models` | Wrong folder, wrong filename, or `storageUri` points at an empty subfolder | Match the file path to `storageUri` (section 5.3) |
| Predictor lists the file but fails to load it | scikit-learn version mismatch or a bad pickle | Re-export with `joblib.dump` using a compatible version |
| Predict returns a shape/dtype error | You saved only the classifier, not the vectorizer | Save the whole `Pipeline` |
| `GET /v1/models/my-model/health` returns 404 | Wrong path | Use `GET /v1/models/my-model` |
| Port-forward stops working after a restart | Forward is attached to the old pod | Stop it and run `kubectl port-forward` again |
| `Unable to listen on port 8080` | Port in use | Use another local port, e.g. `9090:80` |
| `PROVIDER_DOCKER_NOT_RUNNING` | Docker is not started | `sudo systemctl start docker` |
| Pod `Pending` forever | PVC not bound or cluster too small | `kubectl describe pvc my-model-pvc`; give minikube more CPU/memory |
| `ReadWriteOnce` volume conflict | Two pods on different nodes mounting the PVC | Delete the helper pod before deploying |

Golden rule: **the model is read only once, at pod startup.** If you change the files on the PVC, restart the predictor.

---

## 10. Custom API (optional)

If you want your own FastAPI service with the original `/predict` endpoint:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"I was charged twice"}'
```

build an image, load it into minikube, and use a custom container instead of `modelFormat`:

```bash
docker build -t ticket-classifier:latest .
minikube image load ticket-classifier:latest
```

```yaml
spec:
  predictor:
    containers:
      - name: kserve-container
        image: ticket-classifier:latest
        imagePullPolicy: IfNotPresent
        ports:
          - containerPort: 8000
```

Then port-forward and call `/predict` exactly as above.

---

## 11. Clean up

```bash
kubectl delete -f k8s/isvc.yaml
kubectl delete -f k8s/pvc.yaml     # this deletes the stored model
minikube delete
```

---

## Quick start (cheat sheet)

```bash
minikube start --driver=docker --cpus=4 --memory=8192
# install cert-manager + KServe (section 4)
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/helper-pod.yaml
kubectl exec helper-pod -- mkdir -p /mnt/models/my-model
kubectl cp ./model.joblib helper-pod:/mnt/models/my-model/model.joblib
kubectl delete pod helper-pod
kubectl apply -f k8s/isvc.yaml
kubectl port-forward svc/my-model-predictor 8080:80
curl -X POST http://127.0.0.1:8080/v1/models/my-model:predict \
  -H "Content-Type: application/json" -d '{"instances": ["I was charged twice"]}'
```
