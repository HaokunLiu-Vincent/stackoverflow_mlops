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
    TrainerCallback,
)
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('assets', exist_ok=True)
os.makedirs('figs', exist_ok=True)

# Initialize the task
task = Task.init(
    project_name='Financial_News_Sentiment_HPO',
    task_name='HPO step 3 train model',
    task_type=Task.TaskTypes.training,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'processed_dataset_id': '',
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
dataset_id = task.get_parameter('processed_dataset_id')
if not dataset_id:
    dataset_id = task.get_parameter('General/processed_dataset_id')
logger.info(f"Using dataset ID: {dataset_id}")

if not dataset_id:
    raise ValueError("Processed dataset ID not found in parameters.")

# Load data from ClearML Dataset
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy()
logger.info(f"Dataset downloaded to: {dataset_path}")

train_df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))
val_df = pd.read_csv(os.path.join(dataset_path, 'val.csv'))
logger.info(f"Loaded {len(train_df)} train and {len(val_df)} val rows")

# Tokenize the text
tokenizer = AutoTokenizer.from_pretrained(args['model_name'])


def tokenize(batch):
    return tokenizer(
        batch['sentence'],
        truncation=True,
        padding='max_length',
        max_length=int(args['max_seq_len'])
    )


train_ds = HFDataset.from_pandas(train_df[['sentence', 'label']]).map(tokenize, batched=True)
val_ds = HFDataset.from_pandas(val_df[['sentence', 'label']]).map(tokenize, batched=True)
train_ds = train_ds.rename_column('label', 'labels')
val_ds = val_ds.rename_column('label', 'labels')
columns = ['input_ids', 'attention_mask', 'labels']
train_ds.set_format('torch', columns=columns)
val_ds.set_format('torch', columns=columns)

# Initialize the model
model = AutoModelForSequenceClassification.from_pretrained(args['model_name'], num_labels=2)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1_score(labels, preds, average='binary'),
    }


# Report validation metrics to ClearML under the exact title/series the HPO step
# optimizes against (validation/f1, validation/accuracy).
clearml_logger = task.get_logger()


class ClearMLMetricsCallback(TrainerCallback):
    def on_evaluate(self, train_args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        epoch = int(state.epoch) if state.epoch is not None else 0
        if 'eval_accuracy' in metrics:
            clearml_logger.report_scalar('validation', 'accuracy', value=metrics['eval_accuracy'], iteration=epoch)
        if 'eval_f1' in metrics:
            clearml_logger.report_scalar('validation', 'f1', value=metrics['eval_f1'], iteration=epoch)
        logger.info(f"Epoch {epoch}: {metrics}")


training_args = TrainingArguments(
    output_dir='assets/checkpoints',
    num_train_epochs=int(args['num_epochs']),
    per_device_train_batch_size=int(args['batch_size']),
    per_device_eval_batch_size=int(args['batch_size']),
    learning_rate=float(args['learning_rate']),
    weight_decay=float(args['weight_decay']),
    warmup_ratio=float(args['warmup_ratio']),
    eval_strategy='epoch',
    save_strategy='no',
    logging_strategy='epoch',
    report_to=[],
    disable_tqdm=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[ClearMLMetricsCallback()],
)

# Train once across all epochs; the callback streams per-epoch validation scalars.
trainer.train()

# Save the model
model_dir = os.path.join(os.getcwd(), 'assets', 'final_model')
trainer.save_model(model_dir)
tokenizer.save_pretrained(model_dir)
task.upload_artifact('model', model_dir)

print('Training completed successfully')

# Plotting confusion matrix on the validation set
predictions = trainer.predict(val_ds)
y_pred = np.argmax(predictions.predictions, axis=-1)
y_true = predictions.label_ids

label_mapping = {0: 'Negative', 1: 'Positive'}
y_true_names = [label_mapping[label] for label in y_true]
y_pred_names = [label_mapping[label] for label in y_pred]

cm = confusion_matrix(y_true_names, y_pred_names, labels=list(label_mapping.values()))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(label_mapping.values()))
disp.plot(cmap=plt.cm.Blues)

plt.title('Confusion Matrix')
plt.savefig('figs/confusion_matrix.png')

print('Confusion matrix plotted and saved as confusion_matrix.png')