import random
import logging

from typing import List, Dict, Tuple, Optional

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm, trange

from ROTASK.evaluate.ecg_evaluate import evaluate_ecg_model

# Initiate Logger
logger = logging.getLogger(__name__)


def train_ecg_model(model: nn.Module,  # pylint: disable=too-many-locals, too-many-statements, too-many-branches
                    optimizer: torch.optim.Optimizer,  # type: ignore
                    train_loader: DataLoader,
                    valid_loader: DataLoader,
                    selected_tasks: List[str],
                    major_task_count: int,
                    minor_task_count: int,
                    task_weight_type: str,
                    task_weight_args: Dict,
                    device: torch.device,
                    writer: SummaryWriter,
                    num_epochs: int = 100,
                    loss_function: Optional[nn.Module] = None,
                    lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None) -> Tuple[List[Dict], int]:  # pylint: disable=protected-access

    # Remember total instances trained for plotting
    total_steps = 0

    # Save Per Epoch Progress
    result: List[Dict] = []

    epochs = trange(1, num_epochs + 1, dynamic_ncols=True)
    for epoch in epochs:
        epochs.set_description(f'Training Epoch: {epoch}')

        # Set model to Training Mode
        model.train()

        train_loss: Dict[int, float] = {}
        correct_cnt: Dict[int, int] = {}
        total_batches: Dict[int, int] = {}
        total_cnt: Dict[int, int] = {}

        targeted_tasks = selected_tasks

        # Whether to use Dynamic Task Weighting or not
        use_task_weight = (task_weight_type != "")

        if use_task_weight:
            assert task_weight_type in ["DWA"]

            # Gerenate Task Weights
            task_weight: Dict[int, float] = {}

            if task_weight_type == "DWA":
                # Dynamic Weight Average (End-to-End Multi-Task Learning with Attention)
                his_result = result[-6:]
                if len(his_result) < 6:
                    for task_id, task_name in enumerate(selected_tasks):
                        task_weight[task_id] = 1.0
                        logger.info("Task %s | Loss Slope: %s Weight: %s", task_id, None, task_weight[task_id])
                        writer.add_scalar(f'Task_Weight/{task_id}', task_weight[task_id], total_steps)
                else:
                    T = task_weight_args['T']
                    t_ls = {}
                    t_ls_sum = 0.0
                    for task_id, task_name in enumerate(selected_tasks):
                        his_l = list(filter(  # type: ignore
                            None, [r[task_name]['valid'].get('Loss', None) for r in his_result]))[-2:]
                        t_ls[task_id] = his_l[1] / his_l[0]
                        t_ls_sum += np.exp(t_ls[task_id] / T)
                    for task_id, task_name in enumerate(selected_tasks):
                        task_weight[task_id] = len(selected_tasks) * np.exp(t_ls[task_id] / T) / t_ls_sum
                        logger.info("Task %s | Loss Slope: %s Weight: %s", task_id, t_ls[task_id], task_weight[task_id])
                        writer.add_scalar(f'Task_Weight/{task_id}', task_weight[task_id], total_steps)

        training_data = tqdm(train_loader, dynamic_ncols=True, leave=False)
        for data, (_target_mhash, target_task, task_labels) in training_data:
            if target_task[0] == -1:
                assert set(target_task.tolist()) == set([-1]), target_task
                # Pick a task to update
                task_id = random.randrange(len(targeted_tasks))
            else:
                assert len(set(target_task.tolist())) == 1, target_task
                task_id = int(target_task[0])

            # Set the model to use a specific head
            model.set_task(task_id)  # type: ignore

            # Move data to device, model shall already be at device
            target = task_labels[task_id] if target_task[0] == -1 else task_labels
            data = data.to(device)
            target = target.to(device)

            # Run batch data through model
            output_prob, output_logits = model(data)
            prediction = output_prob.max(1, keepdim=True)[1]

            # Get and Sum up Batch Loss
            if loss_function is None:
                batch_loss = F.cross_entropy(output_logits, target)
            else:
                batch_loss = loss_function(output_logits, target)
            if use_task_weight:
                # Multiply by current task weight
                batch_loss = batch_loss * task_weight[task_id]
            train_loss[task_id] = train_loss.get(task_id, 0.0) + batch_loss.item()

            # Increment Correct Count and Total Count
            batch_size = len(data)
            correct_cnt[task_id] = correct_cnt.get(task_id, 0) + prediction.eq(target.view_as(prediction)).sum().item()
            total_batches[task_id] = total_batches.get(task_id, 0) + 1
            total_cnt[task_id] = total_cnt.get(task_id, 0) + batch_size

            # Back Propagation the Loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            training_data.set_description(
                f'Task {task_id:02d} | '
                f'Train loss: {train_loss[task_id] / total_batches[task_id]:.4f} '
                f'Accuracy: {correct_cnt[task_id] / total_cnt[task_id]:.4f}')

            # Write Progress to Tensorboard
            total_steps += batch_size
            writer.add_scalar(f'BATCH/{task_id}/Training Loss',
                              train_loss[task_id] / total_batches[task_id], total_steps)
            writer.add_scalar(f'BATCH/{task_id}/Training Accuracy',
                              correct_cnt[task_id] / total_cnt[task_id], total_steps)

        # Log per epoch metric
        writer.add_scalar('Epoch', epoch, total_steps)

        # TODO: Make it a variable
        # do_validation = (epoch % 2 == 0)
        do_validation = (epoch % 5 == 0)
        # do_validation = True

        per_epoch_metric: Dict[str, Dict] = {}
        for task_id, task_name in enumerate(targeted_tasks):
            logger.info("Task %s: %s", task_id, task_name)
            per_epoch_metric[task_name] = {"train": {}, "valid": {}}

            # Prevent error when the worse sampling probability happen
            if task_id in train_loss:
                per_epoch_metric[task_name]['train']['Loss'] = train_loss[task_id] / total_batches[task_id]
                logger.info("Training Loss: %s", per_epoch_metric[task_name]['train']['Loss'])
                writer.add_scalar(f'Training/{task_id}/Loss',
                                  per_epoch_metric[task_name]['train']['Loss'], total_steps)

                per_epoch_metric[task_name]['train']['Accuracy'] = correct_cnt[task_id] / total_cnt[task_id]
                logger.info("Training Accuracy: %s", per_epoch_metric[task_name]['train']['Accuracy'])
                writer.add_scalar(f'Training/{task_id}/Accuracy',
                                  per_epoch_metric[task_name]['train']['Accuracy'], total_steps)

            if do_validation:
                # Start Validation
                epochs.set_description(f'Validating Epoch: {epoch} / Task: {task_id}')
                per_epoch_metric[task_name]['valid'] = evaluate_ecg_model(
                    model, valid_loader, task_id, task_name,
                    device, f"Validation/{task_id}", total_steps, writer, loss_function)

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
            avg_loss_train = np.mean(
                [m['train']['Loss'] for task_name, m in per_epoch_metric.items()
                 if 'Loss' in m['train'] and ltb <= selected_tasks.index(task_name) < utb])
            logger.info("Average %s Training Loss: %s", mname, avg_loss_train)
            writer.add_scalar(f'Training/AVG_{mname}/Loss', avg_loss_train, total_steps)
            avg_accuracy_train = np.mean(
                [m['train']['Accuracy'] for task_name, m in per_epoch_metric.items()
                 if 'Accuracy' in m['train'] and ltb <= selected_tasks.index(task_name) < utb])
            logger.info("Average %s Training Accuracy: %s", mname, avg_accuracy_train)
            writer.add_scalar(f'Training/AVG_{mname}/Accuracy', avg_accuracy_train, total_steps)

        if do_validation:
            # Calculate average metrics
            for mname, (ltb, utb) in metric_mapping.items():
                avg_loss_valid = np.mean(
                    [m['valid']['Loss'] for task_name, m in per_epoch_metric.items()
                     if ltb <= selected_tasks.index(task_name) < utb])
                logger.info("Average %s Validation Loss: %s", mname, avg_loss_valid)
                writer.add_scalar(f'Validation/AVG_{mname}/Loss', avg_loss_valid, total_steps)
                avg_accuracy_valid = np.mean(
                    [m['valid']['Accuracy'] for task_name, m in per_epoch_metric.items()
                     if ltb <= selected_tasks.index(task_name) < utb])
                logger.info("Average %s Validation Accuracy: %s", mname, avg_accuracy_valid)
                writer.add_scalar(f'Validation/AVG_{mname}/Accuracy', avg_accuracy_valid, total_steps)
                avg_auroc_valid = np.mean(
                    [m['valid']['1']['AUROC'] for task_name, m in per_epoch_metric.items()
                     if ltb <= selected_tasks.index(task_name) < utb])
                logger.info("Average %s Validation AUROC: %s", mname, avg_auroc_valid)
                writer.add_scalar(f'Validation/AVG_{mname}/AUROC', avg_auroc_valid, total_steps)
                # TODO: Make it a config
                target_spc = [70, 80, 90, 95]
                for spc in target_spc:
                    avg_auroc_valid = np.mean(
                        [m['valid']['1'][f'AUROC_{spc}'] for task_name, m in per_epoch_metric.items()
                         if ltb <= selected_tasks.index(task_name) < utb])
                    logger.info("Average %s Validation AUROC at %s: %s", mname, spc / 100, avg_auroc_valid)
                    writer.add_scalar(f'Validation/AVG_{mname}/AUROC_{spc}', avg_auroc_valid, total_steps)

        if lr_scheduler is not None:
            lr_scheduler.step()  # type: ignore
            last_lr = lr_scheduler.get_last_lr()  # type: ignore
            logger.info("lr_scheduler Last lr: %s", last_lr)
            writer.add_scalar('lr_scheduler', last_lr[0], total_steps)

        result.append(per_epoch_metric)

    return result, total_steps
