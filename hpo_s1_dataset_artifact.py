from clearml import Task, Dataset
import pandas as pd
from datasets import load_dataset, concatenate_datasets
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the task
task = Task.init(project_name="Financial_News_Sentiment_HPO", task_name="HPO step 1 dataset artifact")

# Connect parameters
args = {
    'hf_dataset_name': 'lmassaron/FinancialPhraseBank',
}
args = task.connect(args)
logger.info(f"Connected parameters: {args}")

# only create the task, we will actually execute it later
task.execute_remotely()

# Load the dataset from Hugging Face
ds = load_dataset(args['hf_dataset_name'])

# Merge all splits if they exist
parts = []
for split_name in ["train", "validation", "test"]:
    if split_name in ds:
        parts.append(ds[split_name])

if not parts:
    raise ValueError("No dataset splits found.")

dataset_raw = concatenate_datasets(parts)
df = pd.DataFrame(dataset_raw)
logger.info(f"Loaded {len(df)} total rows")
logger.info(f"Original columns: {df.columns.tolist()}")

# Find text column
if "sentence" in df.columns:
    text_col = "sentence"
elif "text" in df.columns:
    text_col = "text"
else:
    raise ValueError(f"Cannot find text column. Columns: {df.columns.tolist()}")

# Find label column
if "sentiment" in df.columns:
    label_col = "sentiment"
elif "label" in df.columns:
    label_col = "label"
else:
    raise ValueError(f"Cannot find label column. Columns: {df.columns.tolist()}")

# Keep only needed columns
df = df[[text_col, label_col]].copy()
df.columns = ["sentence", "label"]

# Normalize label values
df["label"] = df["label"].astype(str).str.strip().str.lower()
logger.info(f"Unique labels before processing: {sorted(df['label'].unique().tolist())}")

# Map numeric labels if needed
numeric_map = {"0": "negative", "1": "neutral", "2": "positive"}
if set(df["label"].unique()).issubset(set(numeric_map.keys())):
    df["label"] = df["label"].map(numeric_map)

# Add sentiment column (keep text labels) and convert label to int
df["sentiment"] = df["label"]
label_to_int = {"negative": 0, "neutral": 1, "positive": 2}
df["label"] = df["sentiment"].map(label_to_int)

# Clean
df = df.dropna(subset=["sentence", "label"])
df = df.drop_duplicates(subset=["sentence"]).reset_index(drop=True)
df = df[["sentiment", "sentence", "label"]]
logger.info(f"Final dataset: {len(df)} rows")

# Pick only top 200 rows for quick testing
df = df.head(200)
logger.info(f"Trimmed dataset to {len(df)} rows for testing")

# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name="Financial PhraseBank Raw Dataset",
    dataset_project="Financial_News_Sentiment_HPO"
)

# Save the data to a temporary file
temp_file = "financial_phrasebank_raw.csv"
df.to_csv(temp_file, index=False)
logger.info(f"Saved data to temporary file: {temp_file}")

# Add the data to the dataset
dataset.add_files(temp_file)
logger.info("Added data file to dataset")

# Upload the dataset
dataset.upload()
logger.info("Uploaded dataset files")

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/dataset_id", dataset.id)
logger.info(f"Stored dataset ID: {dataset.id}")

# Clean up temporary file
os.remove(temp_file)
logger.info("Cleaned up temporary file")

print("Dataset created and uploaded to ClearML")