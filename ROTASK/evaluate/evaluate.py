import logging

from typing import Dict, Tuple
from typing import Optional

import numpy as np

from sklearn.metrics import classification_report, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

from ROTASK.evaluate.metrics import multiclass_roc_auc_score, sensitivity_specificity_support_with_avg
from ROTASK.utils.wandb import log_classification_report

# Initiate Logger
logger = logging.getLogger(__name__)


def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   task_id: int,
                   task_name: str,  # pylint: disable=unused-argument
                   device: torch.device,
                   wandb_name: str,
                   wandb_step: int,
                   writer: SummaryWriter,
                   loss_function: Optional[nn.Module] = None) -> Tuple[Dict, Tuple]:

    # Set model to Eval Mode (For Correct Dropout and BatchNorm Behavior)
    model.eval()
    # Set the model to use a specific head
    model.set_task(task_id)  # type: ignore

    test_loss = 0.0
    correct_cnt = 0

    # Save Predictions, Predicted probability and Truth Data for Evaluation Report
    y_pred, y_pred_prob, y_truth = [], [], []

    with torch.no_grad():
        testing_data = tqdm(test_loader, dynamic_ncols=True, leave=False)
        for data, metadata in testing_data:
            # Move data to device, model shall already be at device
            target = metadata['target']

            data = data.to(device)
            target = target.to(device)

            # Run batch data through model
            output_prob, output_logits = model(data)
            prediction = output_prob.max(1, keepdim=True)[1]

            # Get and Sum up Batch Loss
            if loss_function is None:
                batch_loss = F.cross_entropy(output_logits, target, reduction='sum')
            else:
                batch_loss = loss_function(output_logits, target, reduction='sum')
            test_loss += batch_loss.item()

            # Increment Correct Count and Total Count
            correct_cnt += prediction.eq(target.view_as(prediction)).sum().item()

            # Append Prediction Results
            y_truth.append(target.cpu())
            y_pred_prob.append(output_prob.cpu())
            y_pred.append(prediction.reshape(-1).cpu())

    # Calculate average evaluation loss
    test_loss = test_loss / len(test_loader.dataset)

    # Merge results from each batch
    y_truth = np.concatenate(y_truth)
    y_pred = np.concatenate(y_pred)
    y_pred_prob = np.concatenate(y_pred_prob)

    # Get unique y values
    unique_y = np.unique(np.concatenate([y_truth, y_pred])).tolist()

    # Print Evaluation Metrics and log to wandb
    # TODO: Add name mapping for categories
    report = log_classification_report(
        wandb_name, wandb_step, writer, test_loss,
        classification_report(y_true=y_truth, y_pred=y_pred, labels=unique_y, output_dict=True),
        sensitivity_specificity_support_with_avg(y_truth, y_pred, unique_y),
        confusion_matrix(y_truth, y_pred, labels=unique_y),
        multiclass_roc_auc_score(y_truth, y_pred_prob, unique_y),
        cid2name=None
    )

    return report
