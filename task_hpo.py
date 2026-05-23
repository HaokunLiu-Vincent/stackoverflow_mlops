from clearml import Task, Dataset
from clearml.automation import HyperParameterOptimizer
from clearml.automation import (
    UniformIntegerParameterRange,
    UniformParameterRange,
    DiscreteParameterRange,
)
import logging
import time
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the HPO task
task = Task.init(
    project_name='Financial_News_Sentiment_HPO',
    task_name='HPO: Train Model',
    task_type=Task.TaskTypes.optimizer,
    reuse_last_task_id=False
)

# Connect parameters
args = {
    'base_train_task_id': '',  # Will be set from pipeline
    'num_trials': 6,
    'time_limit_minutes': 60,
    'run_as_service': False,
    'test_queue': 'hpo_finance',  # Queue for test tasks
    'processed_dataset_id': '',  # Will be set from pipeline
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
dataset_id = task.get_parameter('General/processed_dataset_id')  # Get from General namespace
if not dataset_id:
    # Try getting from args as fallback
    dataset_id = args.get('processed_dataset_id')
    logger.info(f"No dataset_id in General namespace, using from args: {dataset_id}")

if not dataset_id:
    logger.error("Processed dataset ID not found in parameters.")
    raise ValueError("Processed dataset ID not found in parameters.")

logger.info(f"Using dataset ID: {dataset_id}")

# Get the actual training model task
try:
    BASE_TRAIN_TASK_ID = args['base_train_task_id']
    logger.info(f"Using base training task ID: {BASE_TRAIN_TASK_ID}")
except Exception as e:
    logger.error(f"Failed to get base training task ID: {e}")
    raise

# Verify dataset exists
try:
    dataset = Dataset.get(dataset_id=dataset_id)
    logger.info(f"Successfully verified dataset: {dataset.name}")
except Exception as e:
    logger.error(f"Failed to verify dataset: {e}")
    raise

# Create the HPO task. Best run is selected by validation F1 (max).
hpo_task = HyperParameterOptimizer(
    base_task_id=BASE_TRAIN_TASK_ID,
    hyper_parameters=[
        UniformIntegerParameterRange('num_epochs', min_value=2, max_value=args['num_epochs']),
        DiscreteParameterRange('batch_size', values=[8, 16, 32]),
        UniformParameterRange('learning_rate', min_value=1e-5, max_value=5e-5),
        UniformParameterRange('weight_decay', min_value=0.0, max_value=0.1),
        UniformParameterRange('warmup_ratio', min_value=0.0, max_value=0.2),
        DiscreteParameterRange('max_seq_len', values=[128, 256]),
    ],
    objective_metric_title='validation',
    objective_metric_series='f1',
    objective_metric_sign='max',
    max_number_of_concurrent_tasks=2,
    optimization_time_limit=args['time_limit_minutes'] * 60,
    compute_time_limit=None,
    total_max_jobs=args['num_trials'],
    min_iteration_per_job=1,
    max_iteration_per_job=args['num_epochs'],
    pool_period_min=1.0,
    execution_queue=args['test_queue'],
    save_top_k_tasks_only=2,
    parameter_override={
        'processed_dataset_id': dataset_id,
        'General/processed_dataset_id': dataset_id,
        'test_queue': args['test_queue'],
        'General/test_queue': args['test_queue'],
        'num_epochs': args['num_epochs'],
        'General/num_epochs': args['num_epochs'],
        'batch_size': args['batch_size'],
        'General/batch_size': args['batch_size'],
        'learning_rate': args['learning_rate'],
        'General/learning_rate': args['learning_rate'],
        'weight_decay': args['weight_decay'],
        'General/weight_decay': args['weight_decay'],
        'warmup_ratio': args['warmup_ratio'],
        'General/warmup_ratio': args['warmup_ratio'],
        'max_seq_len': args['max_seq_len'],
        'General/max_seq_len': args['max_seq_len'],
    }
)

# Start the HPO task
logger.info("Starting HPO task...")
hpo_task.start()

# Wait for optimization to complete
logger.info(f"Waiting for optimization to complete (time limit: {args['time_limit_minutes']} minutes)...")
time.sleep(args['time_limit_minutes'] * 60)  # Wait for the full time limit

# Get the top performing experiments
try:
    top_exp = hpo_task.get_top_experiments(top_k=1)  # Get only the best experiment
    if top_exp:
        best_exp = top_exp[0]
        logger.info(f"Best experiment: {best_exp.id}")

        # Get the best parameters and accuracy
        best_params = best_exp.get_parameters()
        metrics = best_exp.get_last_scalar_metrics()
        best_f1 = metrics['validation']['f1'] if metrics and 'validation' in metrics and 'f1' in metrics['validation'] else None

        # Log detailed information about the best experiment
        logger.info("Best experiment parameters:")
        logger.info(f"  - num_epochs: {best_params.get('num_epochs')}")
        logger.info(f"  - batch_size: {best_params.get('batch_size')}")
        logger.info(f"  - learning_rate: {best_params.get('learning_rate')}")
        logger.info(f"  - weight_decay: {best_params.get('weight_decay')}")
        logger.info(f"  - warmup_ratio: {best_params.get('warmup_ratio')}")
        logger.info(f"  - max_seq_len: {best_params.get('max_seq_len')}")
        logger.info(f"Best validation F1: {best_f1}")

        # Save best parameters and F1
        best_results = {
            'parameters': best_params,
            'f1': best_f1
        }

        # Save to a temporary file
        temp_file = 'best_parameters.json'
        with open(temp_file, 'w') as f:
            json.dump(best_results, f, indent=4)

        # Upload as artifact
        task.upload_artifact('best_parameters', temp_file)
        logger.info(f"Saved best parameters with F1: {best_f1}")

        # Also save as task parameters for easier access
        task.set_parameter('best_parameters', best_params)
        task.set_parameter('best_f1', best_f1)

        logger.info("Best parameters saved as both artifact and task parameters")
    else:
        logger.warning("No experiments completed yet. This might be normal if the optimization just started.")
except Exception as e:
    logger.error(f"Failed to get top experiments: {e}")
    raise

# Make sure background optimization stopped
hpo_task.stop()
logger.info("Optimizer stopped")

print('We are done, good bye')
