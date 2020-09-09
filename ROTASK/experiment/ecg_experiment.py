import logging
import random
import copy

from typing import Dict
from itertools import combinations

import wandb

import numpy as np

import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ROTASK.experiment.ecg_config import ECGExperimentConfig
from ROTASK.datasets.ecg_dataset import ECGDataset, ECGDatasetSubset, RandomECGDatasetSubset
from ROTASK.training.ecg_training import train_ecg_model
from ROTASK.evaluate.ecg_evaluate import evaluate_ecg_model
from ROTASK.datasets.ecg_data_model import ECGtoLVH
from ROTASK.utils.utils import safe_dir


# Initiate Logger
logger = logging.getLogger(__name__)


def run_ecg_experiment(config: ECGExperimentConfig):  # pylint: disable=too-many-statements, too-many-locals, too-many-branches
    # Check Pytorch Version Before Running
    logger.info('Torch Version: %s', torch.__version__)  # type: ignore
    logger.info('Cuda Version: %s', torch.version.cuda)  # type: ignore

    # Initialize Writer
    writer_dir = f"{config.tensorboard_log_root}/{config.cur_time}/"
    writer = SummaryWriter(log_dir=writer_dir)

    # Initialize Device
    device = torch.device(f"cuda:{config.gpu_device_id}")

    # Generate the targeted AUX tasks
    selected_tasks = copy.copy(config.selected_tasks)
    logger.info("Original Selected Tasks: %s", config.selected_tasks)

    if config.aux_task_count != 0:
        # Generate major task included aux task
        if config.target_table != ECGtoLVH:
            assert config.mi_aux_task_count == 0
        assert config.mi_aux_task_count <= config.aux_task_count
        sig = random.sample(config.selected_tasks[1:], config.mi_aux_task_count)
        for j in sig:
            i = config.selected_tasks[0]
            neg_map = random.randint(0, 2)
            if neg_map == 0:
                nt = f"{i}*{j}"
            elif neg_map == 1:
                nt = f"^{i}*{j}"
            elif neg_map == 2:
                nt = f"{i}*^{j}"
            selected_tasks.append(nt)
            logger.info("Add Major Included AUX Tasks: %s", nt)

        # Generate combinations without major task
        # TODO: Make it possbile for more item combinations
        wm_aux_count = config.aux_task_count - config.mi_aux_task_count
        mul = random.sample(list(combinations(config.selected_tasks[1:], 2)), wm_aux_count)
        for i, j in mul:
            neg_map = random.randint(0, 2)
            if neg_map == 0:
                nt = f"{i}*{j}"
            elif neg_map == 1:
                nt = f"^{i}*{j}"
            elif neg_map == 2:
                nt = f"{i}*^{j}"
            selected_tasks.append(nt)
            logger.info("Add Normal AUX Tasks: %s", nt)

    logger.info("Final Selected Tasks: %s", selected_tasks)

    # Initialize Dataset and Split into train/valid/test DataSets
    logger.info('Global Transform:\n%s', config.global_transform)

    ecg_dataset = ECGDataset(db_location=config.db_location,
                             target_table=config.target_table,
                             target_attr=config.target_attr,
                             target_attr_transform=config.target_attr_transform,
                             stratify_attr=config.stratify_attr,
                             train_test_patient_possible_overlap=config.train_test_patient_possible_overlap,
                             preprocess_lead=config.preprocess_lead,
                             transform=config.global_transform,
                             random_seed=config.random_seed)

    # Set the targeted AUX tasks
    ecg_dataset.set_selected_tasks(selected_tasks)

    logger.info('Train Transform:\n%s', config.train_transform)
    train_dataset = ECGDatasetSubset(ecg_dataset, "train", transform=config.train_transform)
    if config.train_sampler_ratio is not None:
        logger.info("Sub-sampling training dataset to %s", config.train_sampler_ratio)
        train_dataset = RandomECGDatasetSubset(train_dataset, config.train_sampler_ratio)
    logger.info('Valid Transform:\n%s', config.valid_transform)
    valid_dataset = ECGDatasetSubset(ecg_dataset, "valid", transform=config.valid_transform)
    if config.valid_sampler_ratio is not None:
        logger.info("Sub-sampling validation dataset to %s", config.valid_sampler_ratio)
        valid_dataset = RandomECGDatasetSubset(valid_dataset, config.valid_sampler_ratio)
    logger.info('Test Transform:\n%s', config.test_transform)
    test_dataset = ECGDatasetSubset(ecg_dataset, "test", transform=config.test_transform)
    if config.test_sampler_ratio is not None:
        logger.info("Sub-sampling testing dataset to %s", config.test_sampler_ratio)
        test_dataset = RandomECGDatasetSubset(test_dataset, config.test_sampler_ratio)

    # Init Imbalance Sampler if Needed
    if config.dataset_sampler is not None:
        if config.train_sampler_ratio is None:
            num_batches = int(100 * config.num_batch_scaler)
        else:
            num_batches = int(100 * config.num_batch_scaler * config.train_sampler_ratio)
        logger.info("Setting num batches to %s", num_batches)
        if config.major_task_scaler != 1:
            assert config.target_table == ECGtoLVH
            task_weights = [config.major_task_scaler] + [1] * (len(selected_tasks) - 1)
            logger.info("Setting task weights to %s", task_weights)
            dataset_sampler = config.dataset_sampler(train_dataset, task_weights=task_weights, num_batches=num_batches,
                                                     **config.dataset_sampler_args)
        else:
            dataset_sampler = config.dataset_sampler(train_dataset, num_batches=num_batches,
                                                     **config.dataset_sampler_args)
        logger.info('Sampler: %s', config.dataset_sampler.__name__)
    else:
        dataset_sampler = None
        logger.info('Sampler: None')

    # Shuffle must be False for dataset_sampler to work
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, drop_last=True,
                              sampler=dataset_sampler, shuffle=(dataset_sampler is None),
                              num_workers=config.dataloader_num_worker)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.dataloader_num_worker)

    # TODO: Add permutated aux_tasks

    if config.model is not None:
        model = config.model(num_class=ecg_dataset.class_cnt,
                             n_variate=ecg_dataset.variate_cnt,
                             num_task=len(selected_tasks),
                             signal_len=ecg_dataset.data_len,
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
    # TODO: Make this flexible
    major_task_count = 1 if config.target_table == ECGtoLVH else 0
    minor_task_count = len(selected_tasks) - major_task_count - config.aux_task_count
    training_history, total_steps = train_ecg_model(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        valid_loader=valid_loader,
        selected_tasks=selected_tasks,
        major_task_count=major_task_count,
        minor_task_count=minor_task_count,
        task_weight_type=config.task_weight_type,
        task_weight_args=config.task_weight_args,
        device=device,
        writer=writer,
        num_epochs=config.num_epochs,
        loss_function=config.loss_function,
        lr_scheduler=lr_scheduler,
    )
    logger.info("Training Complete!")

    # Saving Model
    if config.checkpoint_root is not None:
        logger.info("Saving model at %s", config.checkpoint_root)
        checkpoint_root = safe_dir(config.checkpoint_root)
        logger.info('Saved model at %s!', checkpoint_root)
        save_dict = {
            'state_dict': model.state_dict(),
        }
        torch.save(save_dict, f"{checkpoint_root}/final_model.ckpt")

    logger.info("Testing Started!")
    # Saving prediction probability
    if config.result_root is not None:
        logger.info("Saving prediction probability at %s", config.result_root)
        result_root = safe_dir(config.result_root)
        test_report: Dict[str, Dict] = {}
        for task_id, task_name in enumerate(selected_tasks):
            logger.info("Task %s: %s", task_id, task_name)
            test_report[task_name] = evaluate_ecg_model(
                model, test_loader, task_id, task_name,
                device, f"Testing/{task_id}", total_steps, writer, config.loss_function,
                save_prob_path=f"{result_root}/{task_id}_{task_name}.csv")
    else:
        logger.info("Not saving prediction probability!")
        test_report: Dict[str, Dict] = {}
        for task_id, task_name in enumerate(selected_tasks):
            logger.info("Task %s: %s", task_id, task_name)
            test_report[task_name] = evaluate_ecg_model(
                model, test_loader, task_id, task_name,
                device, f"Testing/{task_id}", total_steps, writer, config.loss_function)

    # Calculate average metrics
    metric_mapping = {
        'all': [0, len(selected_tasks)],
        'major': [0, major_task_count],
        'minor': [major_task_count, major_task_count + minor_task_count],
        'full': [0, major_task_count + minor_task_count],
        'aux': [major_task_count + minor_task_count, len(selected_tasks)],
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
        avg_auroc_test = np.mean(
            [m['1']['AUROC'] for task_name, m in test_report.items()
             if ltb <= selected_tasks.index(task_name) < utb])
        logger.info("Average %s Testing AUROC: %s", mname, avg_auroc_test)
        writer.add_scalar(f'Testing/AVG_{mname}/AUROC', avg_auroc_test, total_steps)
        # TODO: Make it a config
        target_spc = [70, 80, 90, 95]
        for spc in target_spc:
            avg_auroc_test = np.mean(
                [m['1'][f'AUROC_{spc}'] for task_name, m in test_report.items()
                 if ltb <= selected_tasks.index(task_name) < utb])
            logger.info("Average %s Testing AUROC at %s: %s", mname, spc / 100, avg_auroc_test)
            writer.add_scalar(f'Testing/AVG_{mname}/AUROC_{spc}', avg_auroc_test, total_steps)

    logger.info("Testing Complete!")

    return training_history, test_report
