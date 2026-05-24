from clearml import Task, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import logging
import os
import shutil

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the task
task = Task.init(project_name="Financial_News_Sentiment_HPO", task_name="HPO step 2 process dataset")

# Connect parameters
args = {
    'dataset_task_id': '',  # Will be set from pipeline
    'val_size': 0.15,
    'test_size': 0.15,
    'random_state': 42
}
task.connect(args)

# Execute the task remotely
task.execute_remotely()

# Get the dataset task ID from pipeline parameters
dataset_task_id = task.get_parameter('General/dataset_task_id')
logger.info(f"Using dataset task ID: {dataset_task_id}")

# Load the raw dataset from ClearML
dataset_task = Task.get_task(task_id=dataset_task_id)
raw_dataset = Dataset.get(dataset_id=dataset_task.get_parameter('General/dataset_id'))
logger.info(f"Loaded raw dataset: {raw_dataset.name}")

# Get the raw data
# dataset_path = raw_dataset.get_mutable_local_copy("financial_phrasebank_raw")
# To this:
dataset_path = raw_dataset.get_local_copy()

raw_data = pd.read_csv(os.path.join(dataset_path, "financial_phrasebank_raw.csv"))
logger.info(f"Successfully loaded raw data: {len(raw_data)} rows")

# Drop neutral samples and keep only positive / negative (binary task)
data = raw_data[raw_data['sentiment'].isin(['positive', 'negative'])].copy()
logger.info(f"Total rows after removing neutral: {len(data)}")

# Remap to a contiguous binary label space: negative -> 0, positive -> 1
binary_map = {'negative': 0, 'positive': 1}
data['label'] = data['sentiment'].map(binary_map)

X = data[['sentence']]
y = data['label']

# Stratified split into train / val / test, preserving the positive/negative ratio
val_test_size = args['val_size'] + args['test_size']
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=val_test_size, random_state=args['random_state'], stratify=y
)
# Split the held-out portion into validation and test
relative_test_size = args['test_size'] / val_test_size
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=relative_test_size, random_state=args['random_state'], stratify=y_temp
)

logger.info(f"Train size: {len(X_train)}")
logger.info(f"Val size:   {len(X_val)}")
logger.info(f"Test size:  {len(X_test)}")

# Recombine features and labels so each split is a self-contained CSV
train_df = X_train.assign(label=y_train)
val_df = X_val.assign(label=y_val)
test_df = X_test.assign(label=y_test)

# Create a new dataset in ClearML
dataset = Dataset.create(
    dataset_name="Financial PhraseBank Processed Dataset",
    dataset_project="Financial_News_Sentiment_HPO"
)

# Save processed data to temporary files
train_df.to_csv("train.csv", index=False)
val_df.to_csv("val.csv", index=False)
test_df.to_csv("test.csv", index=False)

# Add the processed data to the dataset
for file in ["train.csv", "val.csv", "test.csv"]:
    dataset.add_files(file)
    logger.info(f"Added {file} to dataset")

# Upload the dataset
dataset.upload()
logger.info("Uploaded dataset files")

# Finalize the dataset
dataset.finalize()
logger.info(f"Dataset created with ID: {dataset.id}")

# Store the dataset ID as a task parameter for other steps to use
task.set_parameter("General/processed_dataset_id", str(dataset.id))
logger.info(f"Stored processed dataset ID: {dataset.id}")

# Also store as a task artifact
task.upload_artifact(name='processed_dataset_id', artifact_object=str(dataset.id))
logger.info(f"Stored dataset ID as task artifact: {dataset.id}")

# Clean up temporary files
for file in ["train.csv", "val.csv", "test.csv"]:
    if os.path.exists(file):
        if os.path.isdir(file):
            shutil.rmtree(file)
        else:
            os.remove(file)
        logger.info(f"Cleaned up: {file}")

print("Dataset processing completed and uploaded to ClearML")