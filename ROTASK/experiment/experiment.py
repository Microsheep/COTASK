import random
import logging

from typing import Dict
from itertools import combinations

import wandb

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ROTASK.datasets.dataset import MTCIFAR100Dataset, MTCIFAR100TaskDataset, MTCIFAR100TaskDatasetSubset
from ROTASK.datasets.dataset import MergedMTCIFAR100TaskDataset, MTCIFAR100TaskDatasetDownsampleSubset
from ROTASK.training.training import train_model
from ROTASK.evaluate.evaluate import evaluate_model
from ROTASK.experiment.config import ExperimentConfig


# Initiate Logger
logger = logging.getLogger(__name__)


def run_experiment(config: ExperimentConfig):  # pylint: disable=too-many-statements, too-many-branches, too-many-locals
    # Check Pytorch Version Before Running
    logger.info('Torch Version: %s', torch.__version__)  # type: ignore
    logger.info('Cuda Version: %s', torch.version.cuda)  # type: ignore

    # Initialize Writer
    writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
    writer = SummaryWriter(log_dir=writer_dir)

    # Initialize Device
    device = torch.device(f"cuda:{config.gpu_device_id}")

    # Initialize Dataset and Split into train/valid/test DataSets
    logger.info('Global Transform:\n%s', config.global_transform)
    mtcifar_dataset = MTCIFAR100Dataset(training_noise_ratio=config.training_noise_ratio,
                                        transform=config.global_transform,
                                        random_seed=config.random_seed)

    logger.info('Train Transform:\n%s', config.train_transform)
    if config.train_sampler_ratio is not None:
        logger.info("Sub-sampling training dataset to %s", config.train_sampler_ratio)
    logger.info('Valid Transform:\n%s', config.valid_transform)
    if config.valid_sampler_ratio is not None:
        logger.info("Sub-sampling validation dataset to %s", config.valid_sampler_ratio)
    logger.info('Test Transform:\n%s', config.test_transform)
    if config.test_sampler_ratio is not None:
        logger.info("Sub-sampling testing dataset to %s", config.test_sampler_ratio)

    if config.selected_tasks is None:
        selected_tasks = mtcifar_dataset.coarseclasses
    else:
        selected_tasks = config.selected_tasks
    logger.info("Select Tasks: %s", config.selected_tasks)

    mtcifar_taskdatasets = {}
    mtcifar_taskdataloaders = {}
    for task_name in selected_tasks:
        mtcifar_taskdataset = MTCIFAR100TaskDataset(dataset=mtcifar_dataset, task_name=task_name)

        data_sampler_ratio = (config.train_sampler_ratio, config.valid_sampler_ratio, config.test_sampler_ratio)
        if data_sampler_ratio != (None, None, None):
            mtcifar_taskdataset = MTCIFAR100TaskDatasetDownsampleSubset(
                mtcifar_taskdataset, data_sampler_ratio, random_seed=config.random_seed)

        train_dataset = MTCIFAR100TaskDatasetSubset(mtcifar_taskdataset, "train", transform=config.train_transform)
        valid_dataset = MTCIFAR100TaskDatasetSubset(mtcifar_taskdataset, "valid", transform=config.valid_transform)
        test_dataset = MTCIFAR100TaskDatasetSubset(mtcifar_taskdataset, "test", transform=config.test_transform)

        mtcifar_taskdatasets[task_name] = {
            'full': mtcifar_taskdataset,
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        }
        mtcifar_taskdataloaders[task_name] = {
            'train': DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True,
                                num_workers=config.dataloader_num_worker_per_task),
            'valid': DataLoader(valid_dataset, batch_size=config.batch_size,
                                num_workers=config.dataloader_num_worker_per_task),
            'test': DataLoader(test_dataset, batch_size=config.batch_size,
                               num_workers=config.dataloader_num_worker_per_task)
        }

    # Add permutated aux_tasks
    # TODO: Make it possible to combine more than 2 tasks
    aux_tasks = random.sample(list(combinations(selected_tasks, 2)), config.aux_task_count)
    for aux_task_id, aux_task in enumerate(aux_tasks):
        logger.info("Merging Tasks %s into ID.%s:", aux_task, aux_task_id)

        aux_task_name = f"AUX_JOB_{aux_task_id:02d}_{'.'.join(aux_task)}"
        selected_tasks.append(aux_task_name)

        # Merge the tasks
        aux_datasets = [mtcifar_taskdatasets[task_name]["full"] for task_name in aux_task]
        merged_dataset = MergedMTCIFAR100TaskDataset(aux_datasets)

        train_dataset = MTCIFAR100TaskDatasetSubset(merged_dataset, "train", transform=config.train_transform)
        valid_dataset = MTCIFAR100TaskDatasetSubset(merged_dataset, "valid", transform=config.valid_transform)
        test_dataset = MTCIFAR100TaskDatasetSubset(merged_dataset, "test", transform=config.test_transform)

        mtcifar_taskdatasets[aux_task_name] = {
            'train': train_dataset,
            'valid': valid_dataset,
            'test': test_dataset
        }
        mtcifar_taskdataloaders[aux_task_name] = {
            'train': DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True, shuffle=True,
                                num_workers=config.dataloader_num_worker_per_task),
            'valid': DataLoader(valid_dataset, batch_size=config.batch_size,
                                num_workers=config.dataloader_num_worker_per_task),
            'test': DataLoader(test_dataset, batch_size=config.batch_size,
                               num_workers=config.dataloader_num_worker_per_task)
        }

    if config.model is not None:
        # TODO: Handle tasks with different class count
        class_cnt = mtcifar_taskdatasets[selected_tasks[0]]['train'].class_cnt
        model = config.model(num_class=class_cnt, num_task=len(selected_tasks),
                             **config.model_args).to(device)
        # Make wandb Track the model
        wandb.watch(model, "parameters")
        logger.info('Model: %s', model.__class__.__name__)
        # Log total parameters in the model
        pytorch_total_params = sum(p.numel() for p in model.parameters())
        logger.info('Model params: %s', pytorch_total_params)
        pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info('Model params trainable: %s', pytorch_total_params_trainable)
        # Maybe use https://github.com/sksq96/pytorch-summary in the future
        model_structure_str = "Model Structue:\n"
        for name, param in model.named_parameters():
            model_structure_str += f"\t{name}: {param.requires_grad}, {param.numel()}\n"
        logger.info(model_structure_str)
    else:
        logger.critical("Model not chosen in config!")
        return None

    optimizer = config.optimizer(params=model.parameters(), **config.optimizer_args)
    logger.info("Optimizer: %s\n%s", config.optimizer.__name__, config.optimizer_args)

    if config.lr_scheduler is not None:
        lr_scheduler = config.lr_scheduler(optimizer, **config.lr_scheduler_args)
        logger.info("LR Scheduler: %s\n%s", config.lr_scheduler.__name__, config.lr_scheduler_args)
    else:
        lr_scheduler = None
        logger.info("No LR Scheduler")

    logger.info("Training Started!")
    major_task_count = len(selected_tasks) - config.aux_task_count
    training_history, total_steps = train_model(
        model=model,
        optimizer=optimizer,
        mtcifar_taskdataloaders=mtcifar_taskdataloaders,
        selected_tasks=selected_tasks,
        major_task_count=major_task_count,
        device=device,
        writer=writer,
        num_epochs=config.num_epochs,
        num_aux_epochs=config.num_aux_epochs,
        loss_function=config.loss_function,
        lr_scheduler=lr_scheduler,
    )
    logger.info("Training Complete!")

    logger.info("Testing Started!")
    test_report: Dict[str, Dict] = {}
    for task_id, task_name in enumerate(selected_tasks):
        test_report[task_name] = evaluate_model(
            model, mtcifar_taskdataloaders[task_name]['test'], task_id, task_name,
            device, f"Testing/{task_id}", total_steps, writer, config.loss_function)

    # Calculate average metrics
    metric_mapping = {
        'all': [0, len(selected_tasks)],
        'major': [0, major_task_count],
        'aux': [major_task_count, len(selected_tasks)],
    }
    logger.info("Metric Mapping: %s", metric_mapping)

    for mname, (ltb, utb) in metric_mapping.items():
        avg_loss_test = np.mean(
            [m['Loss'] for task_name, m in test_report.items()
             if ltb <= selected_tasks.index(task_name) < utb])
        logger.info("Average %s Testing Loss: %s", mname, avg_loss_test)
        writer.add_scalar(f'Testing/AVG_{mname}/Loss', avg_loss_test, total_steps)
        avg_accuracy_test = np.mean(
            [m['Accuracy'] for task_name, m in test_report.items()
             if ltb <= selected_tasks.index(task_name) < utb])
        logger.info("Average %s Testing Accuracy: %s", mname, avg_accuracy_test)
        writer.add_scalar(f'Testing/AVG_{mname}/Accuracy', avg_accuracy_test, total_steps)

    logger.info("Testing Complete!")

    return training_history, test_report
