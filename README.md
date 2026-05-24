### 🧬 Financial News Sentiment: HPO Pipeline (5 Steps)

Binary sentiment classification of financial news (Financial PhraseBank) using
`bert-base-uncased`, organized as a ClearML hyperparameter-optimization pipeline
with final-model retraining. Uses the ClearML Dataset API for data management.

#### Step 0: Install dependencies

```bash
pip install -r requirements.txt
```

#### Step 1: Register the HPO Base Tasks

> **Note:** When running for the first time, comment out `task.execute_remotely()` in each .py file to successfully create a task template.

```bash
# Step 1: Upload raw dataset (ClearML Dataset API)
python hpo_s1_dataset_artifact.py

# Step 2: Filter neutral + stratified split, upload processed dataset
python hpo_s2_process_dataset.py

# Step 3: Fine-tune BERT (parameterized for HPO)
python hpo_s3_train_model.py

# Step 4: Hyperparameter optimization (selected by validation F1)
python task_hpo.py

# Step 5: Final model with best parameters + full test evaluation
python final_model.py
```

#### Step 1.5: Initial ClearML Queue
Create a queue named `hpo_finance` (or your customized one), and ensure it is consistent in `pipeline_hpo.py`:
```python
EXECUTION_QUEUE = "hpo_finance"
```

Run the agent for the queue worker:
```bash
clearml-agent daemon --queue "hpo_finance" --detached
```

#### Step 2: Run the HPO Pipeline

```bash
python pipeline_hpo.py
```

#### Single prediction (CLI)

After a model has been trained and saved to `artifacts/final_model/`:

```bash
python predict.py "The company reported strong quarterly earnings and positive market sentiment."
```
# CI/CD test
