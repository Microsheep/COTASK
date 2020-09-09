import os
import pwd
import logging
import pprint

import wandb

import torch

from ROTASK.utils import setup_logging, safe_dir
from ROTASK.experiment import ECGExperimentConfig, run_ecg_experiment
# from ROTASK.datasets import ImbalancedDatasetSampler
from ROTASK.datasets import TaskAwareImbalancedDatasetSampler

from ROTASK.models import MTStanfordModel

# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Setup Experiment Config
    config = ECGExperimentConfig()

    # Setup Log Location
    # Get current username to prevent collision
    username = pwd.getpwuid(os.getuid()).pw_name
    config.tensorboard_log_root = f"/tmp/ROTASK_tb_{username}/"
    config.wandb_dir = f"/tmp/ROTASK_wandb_{username}/"
    # WandB need the directory to be present
    safe_dir(config.wandb_dir)

    # Setup Logging Group
    config.wandb_group = "test_ecg"

    # Setup GPU ID
    config.gpu_device_id = "0"

    # Setup Random Seed
    config.random_seed = 666

    # Setup Data Source (MD5 Checksum: )
    config.db_location = ""

    # Set Selected Tasks
    config.selected_tasks = [
        "===ECHO_LVH===",
        "LVH#LEFT VENTRICULAR HYPERTROPHY",
        "AFIB#ATRIAL FIBRILLATION",
        "RBBB#RIGHT BUNDLE BRANCH BLOCK",
        "1AVB#FIRST DEGREE AV BLOCK",
    ]

    # Set Number of Auxiliary Task to Generate
    # This is the total aux task count including mi count
    config.aux_task_count = 8
    config.mi_aux_task_count = 4
    # config.aux_task_count = 0
    # config.mi_aux_task_count = 0

    # Set Transformations
    config.train_transform = None
    config.valid_transform = None
    config.test_transform = None

    # Set Model
    config.model = MTStanfordModel
    kernel_sz = 25
    config.model_args = {
        "model_structure": {
            'feature_layer': [(64, kernel_sz)],
            'residual': [
                [(64, kernel_sz), (64, kernel_sz)],
                [(64, kernel_sz), (64, kernel_sz)],
                [(128, kernel_sz), (128, kernel_sz)],
                [(128, kernel_sz), (128, kernel_sz)],
                [(256, kernel_sz), (256, kernel_sz)],
                [(256, kernel_sz), (256, kernel_sz)],
                [(512, kernel_sz), (512, kernel_sz)],
                [(512, kernel_sz), (512, kernel_sz)],
            ]
        },
        "dropout_ratio": 0.2,
    }

    # Set Batch Size
    config.batch_size = 64

    # Use ImbalancedDatasetSampler
    # config.dataset_sampler = ImbalancedDatasetSampler
    # config.dataset_sampler_args = {
    #     "num_samples": 100 * config.batch_size,
    #     "weight_function": ImbalancedDatasetSampler.wf_logxdivx
    # }

    config.dataset_sampler = TaskAwareImbalancedDatasetSampler
    # MultiTask-5
    config.num_batch_scaler = 3
    # Single Major Task
    # config.num_batch_scaler = 1

    # Set major task weight scaler
    config.major_task_scaler = 5

    # task_weights: scaled by major_task_scaler
    # num_batches: scaled by num_batch_scaler and train_sampler_ratio
    config.dataset_sampler_args = {
        "batch_size": config.batch_size,
        "weight_function": TaskAwareImbalancedDatasetSampler.wf_logxdivx,
        # "weight_function": TaskAwareImbalancedDatasetSampler.wf_onedivx,
        # "weight_function": TaskAwareImbalancedDatasetSampler.wf_onedivlogx,
    }

    # Set Number of Epochs
    # Single Major Task
    # config.num_epochs = 20
    # MultiTask-5
    config.num_epochs = 50

    # Set Loss function
    config.loss_function = None

    # Set Dynamic Task Weighting
    # config.task_weight_type = "DWA"
    # config.task_weight_args = {
    #     'T': 2.0,
    # }

    # Downsample training data
    config.train_sampler_ratio = None
    # config.train_sampler_ratio = 0.9

    # Set Optimizer
    config.optimizer = torch.optim.Adam
    config.optimizer_args = {
        # Single Major Task
        # "lr": 2e-4,
        # MultiTask-5
        "lr": 1e-4,
    }

    # Set lr_scheduler
    config.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
    config.lr_scheduler_args = {
        # Single Major Task
        # "milestones": [4, 8, 12, 16],
        # MultiTask-5
        "milestones": [7, 14, 21, 28, 35, 42],
        "gamma": 0.5
    }

    # Setup result and model storage
    config.result_root = f"./eval_result/{config.cur_time}"
    config.checkpoint_root = f"./models_saved/{config.cur_time}"

    # Init logging
    log_file_path = f'./logs/{config.cur_time}.log'
    setup_logging(log_file_path, "DEBUG")

    # Init wandb
    wandb.init(
        entity=config.wandb_repo, project=config.wandb_project,
        name=f'{config.cur_time}', group=config.wandb_group,
        dir=config.wandb_dir, config=config.to_dict()
    )
    wandb.tensorboard.patch(pytorch=True)

    # Set config for sweeps
    config.set_wandb_sweep(wandb.config.as_dict())

    # Log Final Config
    logger.info("Experiment Config:\n%s", pprint.pformat(config.to_dict()))

    # Run Experiment
    training_history, test_report = run_ecg_experiment(config)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)
