from clearml.automation import PipelineController
import logging
# import os
# os.environ["CLEARML_API_ACCESS_KEY"] = os.getenv("CLEARML_API_ACCESS_KEY")
# os.environ["CLEARML_API_SECRET_KEY"] = os.getenv("CLEARML_API_SECRET_KEY")
# os.environ["CLEARML_API_HOST"] = os.getenv("CLEARML_API_HOST")

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Queue configuration - using same queue for everything
EXECUTION_QUEUE = "hpo_finance"


def run_pipeline():
    # Connecting ClearML with the current pipeline
    pipe = PipelineController(
        name="Financial_News_Sentiment_Pipeline",
        project="Financial_News_Sentiment_HPO",
        version="0.0.1",
        add_pipeline_tags=False
    )

    # Set default queue for pipeline control
    pipe.set_default_execution_queue(EXECUTION_QUEUE)
    logger.info(f"Set default execution queue to: {EXECUTION_QUEUE}")

    # Add dataset creation step
    pipe.add_step(
        name="stage_data",
        base_task_project="Financial_News_Sentiment_HPO",
        base_task_name="HPO step 1 dataset artifact",
        execution_queue=EXECUTION_QUEUE
    )

    # Add dataset processing step
    pipe.add_step(
        name="stage_process",
        parents=["stage_data"],
        base_task_project="Financial_News_Sentiment_HPO",
        base_task_name="HPO step 2 process dataset",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/dataset_task_id": "${stage_data.id}",
            "General/val_size": 0.15,
            "General/test_size": 0.15,
            "General/random_state": 42
        }
    )

    # Add initial training step
    pipe.add_step(
        name="stage_train",
        parents=["stage_process"],
        base_task_project="Financial_News_Sentiment_HPO",
        base_task_name="HPO step 3 train model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_epochs": 3,
            "General/batch_size": 8,
            "General/learning_rate": 2e-5,
            "General/weight_decay": 0.01,
            "General/warmup_ratio": 0.1,
            "General/max_seq_len": 256
        }
    )

    # Add HPO step
    pipe.add_step(
        name="stage_hpo",
        parents=["stage_train", "stage_process", "stage_data"],
        base_task_project="Financial_News_Sentiment_HPO",
        base_task_name="HPO: Train Model",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/test_queue": EXECUTION_QUEUE,
            "General/num_trials": 6,
            "General/time_limit_minutes": 60,
            "General/run_as_service": False,
            "General/dataset_task_id": "${stage_data.id}",
            "General/base_train_task_id": "${stage_train.id}"
        }
    )

    # Add final model training step
    pipe.add_step(
        name="stage_final_model",
        parents=["stage_hpo", "stage_process"],
        base_task_project="Financial_News_Sentiment_HPO",
        base_task_name="Final Model Training",
        execution_queue=EXECUTION_QUEUE,
        parameter_override={
            "General/processed_dataset_id": "${stage_process.parameters.General/processed_dataset_id}",
            "General/hpo_task_id": "${stage_hpo.id}",
            "General/test_queue": EXECUTION_QUEUE
        }
    )

    # Start the pipeline locally but tasks will run on queue
    logger.info("Starting pipeline locally with tasks on queue: %s", EXECUTION_QUEUE)
    pipe.start_locally()
    logger.info("Pipeline started successfully")


if __name__ == "__main__":
    run_pipeline()
