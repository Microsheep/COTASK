import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type, List

import torch
import torch.nn as nn

from ROTASK.datasets.dataset import MTCIFAR100Dataset

# Initiate Logger
logger = logging.getLogger(__name__)


def transform_dict(config_dict: Dict, expand: bool = True):
    """
    General function to transform any dictionary into wandb config acceptable format
    (This is mostly due to datatypes that are not able to fit into YAML format which makes wandb angry)
    The expand argument is used to expand iterables into dictionaries
    So the configs can be used when comparing results across runs
    """
    ret: Dict[str, Any] = {}
    for k, v in config_dict.items():
        if v is None or isinstance(v, (int, float, str)):
            ret[k] = v
        elif isinstance(v, (list, tuple, set)):
            # Need to check if item in iterable is YAML-friendly
            t = transform_dict(dict(enumerate(v)), expand)
            # Transform back to iterable if expand is False
            ret[k] = t if expand else [t[i] for i in range(len(v))]
        elif isinstance(v, dict):
            ret[k] = transform_dict(v, expand)
        else:
            # Transform to YAML-friendly (str) format
            # Need to handle both Classes, Callables, Object Instances
            # Custom Classes might not have great __repr__ so __name__ might be better in these cases
            vname = v.__name__ if hasattr(v, '__name__') else v.__class__.__name__
            ret[k] = f"{v.__module__}:{vname}"
    return ret


def dfac_cur_time():
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def dfac_optimizer_args():
    return {
        "lr": 1e-2,
    }


@dataclass
class ExperimentConfig:  # pylint: disable=too-many-instance-attributes
    # GPU Device Setting
    gpu_device_id: str = "1"

    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)
    tensorboard_log_root: str = "/tmp/ROTASK_tb/"
    wandb_dir: str = "/tmp/ROTASK_wandb/"

    # WandB setting
    wandb_repo: str = "REMOVED"
    wandb_project: str = "REMOVED"
    wandb_group: str = "test"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = None

    # Transform Function
    global_transform: Optional[Tuple[Callable, ...]] = None
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None

    # Increase dataloader worker to increase throughput
    dataloader_num_worker_per_task: int = 2

    # Training Related
    batch_size: int = 128

    # Default No Dataset Sampler
    # Eg. ROTASK.dataset.dataset: ImbalancedDatasetSampler
    dataset_sampler: Optional[Type[torch.utils.data.sampler.Sampler]] = None
    dataset_sampler_args: Dict[str, Any] = field(default_factory=dict)

    # Default No Random Data Sampling
    train_sampler_ratio: Optional[float] = None
    valid_sampler_ratio: Optional[float] = None
    test_sampler_ratio: Optional[float] = None

    # Default Don't Select Model
    model: Optional[Type[torch.nn.Module]] = None
    model_args: Dict[str, Any] = field(default_factory=dict)

    # Default to train on all tasks
    selected_tasks: Optional[List[str]] = None

    # Default Cross Entropy loss
    loss_function: Optional[nn.Module] = None

    # Default Select Adam as Optimizer
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam  # type: ignore
    optimizer_args: Dict[str, Any] = field(default_factory=dfac_optimizer_args)

    # Default adjust learning rate
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    lr_scheduler_args: Dict[str, Any] = field(default_factory=dict)

    # Set number of epochs to train
    num_epochs: int = 200

    # Number of aux jobs
    aux_task_count: int = 0

    # Drop AUX Epoch Start Point
    num_aux_epochs: Optional[int] = None

    # Amount of label noise
    training_noise_ratio: float = 0.0

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)

    def set_wandb_sweep(self, wandb_config: Dict[str, Any]):
        SWEEP_ARG_PREFIX = "WS_"
        for k, v in wandb_config.items():
            if k.startswith(SWEEP_ARG_PREFIX):
                sp_name = k[len(SWEEP_ARG_PREFIX):]
                # wandb.config.as_dict() returns Dict[k, Dict[str, v]]
                # https://github.com/wandb/client/blob/master/wandb/wandb_config.py#L321
                sp_value = v['value']
                if sp_name == "lr":
                    assert isinstance(sp_value, float)
                    assert isinstance(self.optimizer_args, dict)
                    self.optimizer_args["lr"] = sp_value
                    logger.info("Setting %s to %s according to sweep!", sp_name, sp_value)
                elif sp_name == "lr_step":
                    assert isinstance(sp_value, int)
                    assert isinstance(self.lr_scheduler_args, dict)
                    # Small Hack: num_epochs might also be searched which makes milestone calculations wrong
                    if f"{SWEEP_ARG_PREFIX}num_epochs" in wandb_config:
                        real_epoch = wandb_config[f"{SWEEP_ARG_PREFIX}num_epochs"]['value']
                        assert isinstance(real_epoch, int)
                        self.lr_scheduler_args["milestones"] = list(range(sp_value, real_epoch, sp_value))
                    else:
                        self.lr_scheduler_args["milestones"] = list(range(sp_value, self.num_epochs, sp_value))
                    logger.info("Setting %s to %s according to sweep!", sp_name, self.lr_scheduler_args["milestones"])
                elif sp_name == "selected_tasks":
                    assert isinstance(sp_value, str)
                    assert sp_value in MTCIFAR100Dataset.coarseclasses
                    self.selected_tasks = [sp_value]
                    logger.info("Setting %s to %s according to sweep!", sp_name, self.selected_tasks)
                elif hasattr(self, sp_name):
                    assert isinstance(getattr(self, sp_name, None), type(sp_value))
                    setattr(self, sp_name, sp_value)
                    logger.info("Setting %s to %s according to sweep!", sp_name, sp_value)
                else:
                    raise NameError(f"{k} can not be matched to set sweep function")
