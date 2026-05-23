import os
os.environ.pop('MPLBACKEND', None)
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from clearml import Task, Dataset
import torch
import numpy as np
import pandas as pd
from datasets import Dataset as HFDataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, matthews_corrcoef, roc_auc_score,
    average_precision_score, log_loss, brier_score_loss,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve,
)
import logging
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)
os.makedirs('artifacts/evaluation', exist_ok=True)

# Initialize the task
task = Task.init(
    project_name='Financial_News_Sentiment_HPO',
    task_name='Final Model Training',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'processed_dataset_id': '',  # Will be set from pipeline
    'hpo_task_id': None,  # Will be set from pipeline
    'test_queue': 'hpo_finance',
    'model_name': 'bert-base-uncased',
    'num_epochs': 3,
    'batch_size': 8,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'warmup_ratio': 0.1,
    'max_seq_len': 256
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# Execute the task remotely
task.execute_remotely()

# Get the dataset ID from pipeline parameters
dataset_id = task.get_parameter('General/processed_dataset_id')
if not dataset_id:
    dataset_id = args.get('processed_dataset_id')
logger.info(f"Using dataset ID: {dataset_id}")

if not dataset_id:
    raise ValueError("Processed dataset ID not found in parameters.")

# Get the HPO task ID
hpo_task_id = args.get('hpo_task_id')
if not hpo_task_id:
    raise ValueError("HPO task ID not found in parameters")

# Get the HPO task and best parameters
hpo_task = Task.get_task(task_id=hpo_task_id)
logger.info(f"Retrieved HPO task: {hpo_task.name}")

try:
    best_params = hpo_task.get_parameter('best_parameters')
    best_f1 = hpo_task.get_parameter('best_f1')

    if best_params is None:
        logger.info("Best parameters not found in task parameters, trying artifact...")
        if 'best_parameters' not in hpo_task.artifacts:
            raise ValueError("No best_parameters artifact found in HPO task")

        artifact_path = hpo_task.artifacts['best_parameters'].get_local_copy()
        if artifact_path is None:
            raise ValueError("Failed to get local copy of best_parameters artifact")

        with open(artifact_path, 'r') as f:
            best_results = json.load(f)

        best_params = best_results['parameters']
        best_f1 = best_results.get('f1')

    # Update training parameters with best values
    args['num_epochs'] = int(best_params.get('num_epochs', args['num_epochs']))
    args['batch_size'] = int(best_params.get('batch_size', args['batch_size']))
    args['learning_rate'] = float(best_params.get('learning_rate', args['learning_rate']))
    args['weight_decay'] = float(best_params.get('weight_decay', args['weight_decay']))
    args['warmup_ratio'] = float(best_params.get('warmup_ratio', args['warmup_ratio']))
    args['max_seq_len'] = int(best_params.get('max_seq_len', args['max_seq_len']))

    logger.info(f"Using best parameters from HPO: {best_params}")
    logger.info(f"Best validation F1 from HPO: {best_f1}")
except Exception as e:
    logger.error(f"Failed to get best parameters from HPO task: {e}")
    raise

# Load data from ClearML Dataset
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy()
logger.info(f"Dataset downloaded to: {dataset_path}")

train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
val_df = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
test_df = pd.read_csv(os.path.join(dataset_path, 'test.csv'))
# Train on the combined train+val portion now that hyperparameters are fixed
train_full_df = pd.concat([train_df, val_df], ignore_index=True)
logger.info(f"Training samples: {len(train_full_df)}, Test samples: {len(test_df)}")

# Tokenize
tokenizer = AutoTokenizer.from_pretrained(args['model_name'])


def tokenize(batch):
    return tokenizer(
        batch['sentence'],
        truncation=True,
        padding='max_length',
        max_length=int(args['max_seq_len'])
    )


train_ds = HFDataset.from_pandas(train_full_df[['sentence', 'label']]).map(tokenize, batched=True)
test_ds = HFDataset.from_pandas(test_df[['sentence', 'label']]).map(tokenize, batched=True)
train_ds = train_ds.rename_column('label', 'labels')
test_ds = test_ds.rename_column('label', 'labels')
columns = ['input_ids', 'attention_mask', 'labels']
train_ds.set_format('torch', columns=columns)
test_ds.set_format('torch', columns=columns)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(args['model_name'], num_labels=2)

training_args = TrainingArguments(
    output_dir='assets/final_checkpoints',
    num_train_epochs=int(args['num_epochs']),
    per_device_train_batch_size=int(args['batch_size']),
    per_device_eval_batch_size=int(args['batch_size']),
    learning_rate=float(args['learning_rate']),
    weight_decay=float(args['weight_decay']),
    warmup_ratio=float(args['warmup_ratio']),
    eval_strategy='no',
    save_strategy='no',
    logging_strategy='epoch',
    report_to=[],
)

trainer = Trainer(model=model, args=training_args, train_dataset=train_ds)

logger.info("Starting final training...")
trainer.train()

# Evaluate on the held-out test set
predictions = trainer.predict(test_ds)
logits = predictions.predictions
y_true = predictions.label_ids
y_pred = np.argmax(logits, axis=-1)
probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
y_score = probs[:, 1]

metrics = {
    'accuracy': accuracy_score(y_true, y_pred),
    'f1': f1_score(y_true, y_pred, average='binary'),
    'precision': precision_score(y_true, y_pred, average='binary'),
    'recall': recall_score(y_true, y_pred, average='binary'),
    'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
    'mcc': matthews_corrcoef(y_true, y_pred),
    'roc_auc': roc_auc_score(y_true, y_score),
    'pr_auc': average_precision_score(y_true, y_score),
    'log_loss': log_loss(y_true, probs),
    'brier_score': brier_score_loss(y_true, y_score),
}
logger.info(f"Test metrics: {metrics}")

# Report final metrics as single-value scalars
clearml_logger = task.get_logger()
for name, value in metrics.items():
    clearml_logger.report_scalar(title='test', series=name, value=value, iteration=0)

# Persist metrics summary
with open('artifacts/evaluation/after_hyperparameter_tuning_metrics_summary.json', 'w') as f:
    json.dump(metrics, f, indent=2)
task.upload_artifact('test_metrics', 'artifacts/evaluation/after_hyperparameter_tuning_metrics_summary.json')

# Confusion matrix
label_mapping = {0: 'Negative', 1: 'Positive'}
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values()))
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.savefig('figs/confusion_matrix.png')
clearml_logger.report_matplotlib_figure('Confusion Matrix', 'confusion_matrix', plt.gcf(), 0)
plt.close()

# ROC curve
fpr, tpr, _ = roc_curve(y_true, y_score)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"ROC (AUC={metrics['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.savefig('figs/roc_curve.png')
clearml_logger.report_matplotlib_figure('ROC Curve', 'roc_curve', plt.gcf(), 0)
plt.close()

# Precision-Recall curve
precision, recall, _ = precision_recall_curve(y_true, y_score)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label=f"PR (AP={metrics['pr_auc']:.3f})")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.savefig('figs/pr_curve.png')
clearml_logger.report_matplotlib_figure('PR Curve', 'pr_curve', plt.gcf(), 0)
plt.close()

# Per-sample predictions
pred_df = test_df.copy()
pred_df['true_label'] = y_true
pred_df['predicted_label'] = y_pred
pred_df['confidence'] = y_score
pred_df.to_csv('artifacts/evaluation/after_hyperparameter_tuning_full_predictions.csv', index=False)
task.upload_artifact('full_predictions', 'artifacts/evaluation/after_hyperparameter_tuning_full_predictions.csv')

# Save the final model
model_dir = os.path.join(os.getcwd(), 'artifacts', 'final_model')
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
task.upload_artifact('model', model_dir)
logger.info("Model saved and uploaded as artifact")

print('Training completed successfully!')