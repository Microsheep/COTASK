import logging

from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Optional, Dict, Tuple, Any, Type, List

import torch
import torch.nn as nn

from ROTASK.datasets.ecg_data_model import Base, ECGtoLVH
from ROTASK.datasets.ecg_dataset import ECGDataset
from ROTASK.preprocess.signal_preprocess import preprocess_leads

from ROTASK.datasets import TaskAwareImbalancedDatasetSampler


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
class ECGExperimentConfig:  # pylint: disable=too-many-instance-attributes
    # GPU Device Setting
    gpu_device_id: str = "0"

    # Logging Related
    cur_time: str = field(default_factory=dfac_cur_time)
    tensorboard_log_root: str = "/tmp/ROTASK_tb/"
    wandb_dir: str = "/tmp/ROTASK_wandb/"

    # Setup result and model storage
    result_root: Optional[str] = None
    checkpoint_root: Optional[str] = None

    # WandB setting
    wandb_repo: str = "REMOVED"
    wandb_project: str = "REMOVED"
    wandb_group: str = "test"

    # Set random seed. Set to None to create new Seed
    random_seed: Optional[int] = None

    # Database location
    db_location: str = ""

    # Dataset Settings
    target_table: Base = ECGtoLVH
    target_attr: str = "LVH_LVmass_level"
    stratify_attr: Tuple[str, ...] = ('gender', 'EKG_age',
                                      'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD')
    train_test_patient_possible_overlap: bool = False
    target_attr_transform: Optional[Callable] = ECGDataset.transform_lvh_level
    preprocess_lead: Optional[Callable] = preprocess_leads

    # Transform Function
    global_transform: Optional[Tuple[Callable, ...]] = None
    train_transform: Optional[Tuple[Callable, ...]] = None
    valid_transform: Optional[Tuple[Callable, ...]] = None
    test_transform: Optional[Tuple[Callable, ...]] = None

    # Increase dataloader worker to increase throughput
    dataloader_num_worker: int = 8

    # Training Related
    batch_size: int = 64

    # Scale the number of batches for the sampler
    num_batch_scaler: int = 1

    # Scale the sample probability for major task
    major_task_scaler: int = 1

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

    # The first one have to be `===ECHO_LVH===`
    selected_tasks: Optional[List[str]] = None

    # Whether to use Dynamic Task Weighting or not
    task_weight_type: str = ""
    task_weight_args: Dict[str, Any] = field(default_factory=dict)

    # Default Cross Entropy loss
    loss_function: Optional[nn.Module] = None

    # Default Select Adam as Optimizer
    optimizer: Type[torch.optim.Optimizer] = torch.optim.Adam  # type: ignore
    optimizer_args: Dict[str, Any] = field(default_factory=dfac_optimizer_args)

    # Default adjust learning rate
    lr_scheduler: Optional[Type[torch.optim.lr_scheduler._LRScheduler]] = None  # pylint: disable=protected-access
    lr_scheduler_args: Dict[str, Any] = field(default_factory=dict)

    # Set number of epochs to traisn
    num_epochs: int = 50

    # Number of aux jobs
    # This is the total aux task count including major_included aux task count
    aux_task_count: int = 0
    # This is major_included aux task count
    mi_aux_task_count: int = 0

    # Drop AUX Epoch Start Point
    num_aux_epochs: Optional[int] = None

    # Amount of label noise
    training_noise_ratio: float = 0.0

    def to_dict(self, expand: bool = True):
        return transform_dict(asdict(self), expand)

    def set_wandb_sweep(self, wandb_config: Dict[str, Any]):  # pylint: disable=too-many-branches
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
                elif sp_name == "weight_function":
                    assert sp_value in ["wf_logxdivx", "wf_onedivx", "wf_onedivlogx"]
                    if sp_value == "wf_logxdivx":
                        self.dataset_sampler_args["weight_function"] = TaskAwareImbalancedDatasetSampler.wf_logxdivx
                    elif sp_value == "wf_onedivx":
                        self.dataset_sampler_args["weight_function"] = TaskAwareImbalancedDatasetSampler.wf_onedivx
                    elif sp_value == "wf_onedivlogx":
                        self.dataset_sampler_args["weight_function"] = TaskAwareImbalancedDatasetSampler.wf_onedivlogx
                    logger.info("Setting %s to %s according to sweep!", sp_name, sp_value)
                elif sp_name == "selected_tasks":
                    assert isinstance(sp_value, str)
                    self.selected_tasks = [sp_value]
                    logger.info("Setting %s to %s according to sweep!", sp_name, self.selected_tasks)
                elif hasattr(self, sp_name):
                    assert isinstance(getattr(self, sp_name, None), type(sp_value))
                    setattr(self, sp_name, sp_value)
                    logger.info("Setting %s to %s according to sweep!", sp_name, sp_value)
                else:
                    raise NameError(f"{k} can not be matched to set sweep function")
