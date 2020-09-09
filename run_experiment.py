import os
import pwd
import logging
import pprint

import wandb

import torch
import torchvision.transforms as transforms

from ROTASK.utils import setup_logging, safe_dir
from ROTASK.experiment import ExperimentConfig, run_experiment

from ROTASK.datasets import MTCIFAR100Dataset
from ROTASK.models import MTResNet, BasicBlock
# from ROTASK.models import TRMTResNet, TRBasicBlock
# from ROTASK.models import MTSimpleConvNet
# from ROTASK.models import CSMTSimpleConvNet
# from ROTASK.models import TRSimpleConvNet

# Initiate Logger
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Setup Experiment Config
    config = ExperimentConfig()

    # Setup Log Location
    # Get current username to prevent collision
    username = pwd.getpwuid(os.getuid()).pw_name
    config.tensorboard_log_root = f"/tmp/ROTASK_tb_{username}/"
    config.wandb_dir = f"/tmp/ROTASK_wandb_{username}/"
    # WandB need the directory to be present
    safe_dir(config.wandb_dir)

    # Setup Logging Group
    config.wandb_group = "test"

    # Setup GPU ID
    config.gpu_device_id = "0"

    # Setup Random Seed
    config.random_seed = 666

    # Set Transformations
    config.train_transform = (
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(MTCIFAR100Dataset.transform_mean_std["mean"], MTCIFAR100Dataset.transform_mean_std["std"]),
    )
    config.valid_transform = (
        transforms.ToTensor(),
        transforms.Normalize(MTCIFAR100Dataset.transform_mean_std["mean"], MTCIFAR100Dataset.transform_mean_std["std"]),
    )
    config.test_transform = (
        transforms.ToTensor(),
        transforms.Normalize(MTCIFAR100Dataset.transform_mean_std["mean"], MTCIFAR100Dataset.transform_mean_std["std"]),
    )

    # Set Model
    config.model = MTResNet
    config.model_args = {
        "block": BasicBlock,
        "num_block": [3, 4, 6, 3]
    }
    # config.model = TRMTResNet
    # config.model_args = {
    #     "block": TRBasicBlock,
    #     "num_block": [3, 4, 6, 3],
    #     "sigma": 0.6,
    # }
    # config.model = MTSimpleConvNet
    # config.model_args = {}
    # config.model = CSMTSimpleConvNet
    # config.model_args = {
    #     "alpha_self": 0.9,
    # }
    # config.model = TRSimpleConvNet
    # config.model_args = {
    #     "sigma": 0.2,
    # }

    # Set Number of Epochs
    # Multitask, AUX 40
    config.num_epochs = 400
    # Single Task
    # config.num_epochs = 900
    # SimpleConvNet
    # config.num_epochs = 100

    # Set Batch Size
    config.batch_size = 128

    # Set Number of Auxiliary Task to Generate
    config.aux_task_count = 0
    # config.aux_task_count = 40
    # config.aux_task_count = 50

    # Amount of training data noise to add
    config.training_noise_ratio = 0.0

    # Downsample training data
    config.train_sampler_ratio = None
    # config.train_sampler_ratio = 0.9

    # Set Loss function
    config.loss_function = None

    # Set Optimizer
    config.optimizer = torch.optim.SGD
    # Single Task, Multitask, AUX 40
    config.optimizer_args = {
        "lr": 0.1,
        "momentum": 0.9,
        "nesterov": True,
        "weight_decay": 5e-4,
    }
    # SimpleConvNet
    # config.optimizer_args = {
    #     "lr": 0.01,
    #     "momentum": 0.9,
    #     "nesterov": True,
    #     "weight_decay": 5e-4,
    # }

    # Set lr_scheduler
    config.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR
    config.lr_scheduler_args = {
        # Multitask, AUX 40
        "milestones": [60, 120, 180, 240, 300, 360],
        "gamma": 0.5,
        # Single Task
        # "milestones": [250, 500, 750],
        # "gamma": 0.5,
        # SimpleConvNet
        # "milestones": [20, 40, 60, 80],
        # "gamma": 0.1,
    }

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
    training_history, test_report = run_experiment(config)
    logger.info("Test Result:\n%s", pprint.pformat(test_report))

    # Save log file to wandb
    wandb.save(log_file_path)
