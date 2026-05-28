# 🧬 MLOPs: Financial News Sentiment

This repository is dedicated to the MLOps work of our [Stackoverflow_Project](https://github.com/Rongfeng-zhao/Stackoverflow_Project). It is built independently to implement **MLOps CI/CD workflows**, covering the entire workflow from HPO pipeline construction in ClearML, continuous integration with Github Action Workflow, and deployment with Docker&FastAPI.

---

## 📋 Table of Contents

- [Project Structure](#-project-structure)
- [Environment Setup](#-environment-setup)
- [Register HPO Base Tasks](#-step-1-register-the-hpo-base-tasks)
- [ClearML Queue Setup](#-step-2-clearml-queue-setup)
- [Run the HPO Pipeline](#-step-3-run-the-hpo-pipeline)
- [CI/CD](#-cicd)
- [Deployment](#-deployment)
- [Single Prediction (CLI)](#-single-prediction-cli)


## 📁 Project Structure

```
stackoverflow_mlops/
├── hpo_s1_dataset_artifact.py    # Step 1: Dataset upload
├── hpo_s2_process_dataset.py     # Step 2: Data preprocessing
├── hpo_s3_train_model.py         # Step 3: Model training
├── hpo_s4_task_hpo.py            # Step 4: Hyperparameter optimization
├── hpo_s5_final_model.py         # Step 5: Final model training
├── pipeline_hpo.py               # Pipeline orchestrator
├── requirements.txt              # Python dependencies
├── .github/
│   └── workflows/
│       └── pipeline.yaml         # CI/CD workflow
└── deployment/
    ├── app.py                    # FastAPI application
    ├── Dockerfile                # Docker configuration
    ├── requirements.txt          # Deployment dependencies
    └── model/                    # Model files (git-ignored)
```

---

## 🛠 Environment Setup

Create a conda environment with Python 3.12.3 and install all dependencies:

```bash
conda create --name clearml-mlops python=3.12.3
conda activate clearml-mlops
pip install -r requirements.txt
```

Initialize ClearML credentials (you will need your API key from https://app.clear.ml/settings):

```bash
clearml-init
```

Install the ClearML agent for running pipeline tasks:

```bash
pip install clearml-agent
```

---

## 🚀 Step 1: Register the HPO Base Tasks

Before running the pipeline, each task must be registered in ClearML.  Run them once:

```bash
python hpo_s1_dataset_artifact.py    # Upload raw dataset (ClearML Dataset API)
python hpo_s2_process_dataset.py     # Filter neutral + stratified split
python hpo_s3_train_model.py         # Fine-tune BERT (parameterized for HPO)
python hpo_s4_task_hpo.py            # Hyperparameter optimization
python hpo_s5_final_model.py         # Final model with best parameters
```

---

## ⚙ Step 2: ClearML Queue Setup

Create a queue named `hpo_finance` in the ClearML web UI (Workers & Queues), and ensure it matches the value in `pipeline_hpo.py`:

```python
EXECUTION_QUEUE = "hpo_finance"
```

Start two workers on the queue (two are required — one for the HPO controller, one for the trial tasks):

```bash
CLEARML_WORKER_ID=worker_1 clearml-agent daemon --queue hpo_finance --detached
CLEARML_WORKER_ID=worker_2 clearml-agent daemon --queue hpo_finance --detached
```

Verify both workers are running:

```bash
ps aux | grep clearml-agent | grep -v grep
```

---

## 🔁 Step 3: Run the HPO Pipeline

```bash
python pipeline_hpo.py
```

The pipeline runs 5 stages in sequence:

1. **stage_data** — Downloads and uploads the Financial PhraseBank dataset
2. **stage_process** — Filters neutral samples, splits into train/val/test
3. **stage_train** — Fine-tunes BERT as a baseline and HPO template
4. **stage_hpo** — Runs hyperparameter optimization (6 trials, optimizing validation F1)
5. **stage_final_model** — Retrains with best parameters on train+val, evaluates on test

Monitor progress in the ClearML web UI under the **Financial_News_Sentiment_HPO** project.

---

## 🔄 CI/CD

The repository includes a GitHub Actions workflow (`.github/workflows/pipeline.yaml`) that triggers on push and pull requests to `main`.

### GitHub Secrets

Add the following secrets in your GitHub repo under Settings → Secrets and variables → Actions:

- `CLEARML_API_ACCESS_KEY` — your ClearML access key
- `CLEARML_API_SECRET_KEY` — your ClearML secret key
- `CLEARML_API_HOST` — `https://api.clear.ml`

### Trigger the CI/CD Pipeline

Create a feature branch, make a change, and open a pull request:

```bash
git checkout -b my-feature
echo "# update" >> README.md
git add .
git commit -m "trigger CI/CD"
git push origin my-feature
```

Then go to GitHub → Pull requests → New pull request → base: `main`, compare: `my-feature` → Create pull request.

The workflow will automatically run a smoke test (steps 1–3) to verify code correctness and ClearML connectivity.

---

## 🐳 Deployment

After the pipeline completes, deploy the final model as a FastAPI service using Docker.

### 1. Download the Model

Download `final_model.tar.gz` from the ClearML web UI:
- Go to the **Final Model Training** task → **Artifacts** tab → download `model`

### 2. Unpack the Model

```bash
cd deployment
mkdir -p model
tar -xzf final_model.tar.gz -C model/
```

Verify the model files:

```bash
ls deployment/model/
# Expected: config.json  model.safetensors  tokenizer.json  tokenizer_config.json  training_args.bin
```

### 3. Build the Docker Image

```bash
cd deployment
docker build -t financial-sentiment-api .
```

### 4. Launch the Service

```bash
docker run -p 8000:8000 financial-sentiment-api
```

### 5. Test the API

Health check:

```bash
curl http://localhost:8000/health
```

Predict sentiment:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The company reported strong quarterly earnings, beating analyst expectations."}'
```

Interactive API docs are available at: http://localhost:8000/docs

---

## 🔮 Single Prediction (CLI)

After a model has been trained and saved to `artifacts/final_model/`:

```bash
python predict.py "The company reported strong quarterly earnings and positive market sentiment."
```

---

