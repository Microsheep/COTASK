"""
Task Routing Layer Implementation
Source: https://github.com/gstrezoski/TaskRouting
"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskRouter(nn.Module):

    def __init__(self, unit_count: int, task_count: int, sigma: float = 0.5):
        super().__init__()

        self.unit_count = unit_count
        self.task_count = task_count

        assert 0 < sigma < 1
        self.sigma = sigma

        # We save a different mapping for each task
        self.cur_task = -1

        # Setup the masks for each task
        self.task_mask = torch.ones((task_count, unit_count))
        zero_mask_index = np.random.rand(task_count, unit_count).argsort(1)[:, :int(self.unit_count * sigma)]
        self.task_mask[np.arange(task_count)[:, None], zero_mask_index] = 0

        # Add it to parameter list but disable gradient
        self.task_mask = nn.Parameter(self.task_mask, requires_grad=False)
        # self.task_mask = nn.Parameter(self.task_mask, requires_grad=True)

    def set_task(self, task_id: int):
        assert task_id < self.task_count
        self.cur_task = task_id

    def forward(self, x):  # pylint: disable=arguments-differ
        orig_shape = x.shape
        sel_mask = self.task_mask[self.cur_task].reshape(self.unit_count, 1)
        # orig_shape[0] is batch size
        # assert orig_shape[1] == self.unit_count
        return (x.reshape(orig_shape[0], self.unit_count, -1) * sel_mask).reshape(orig_shape)


class TRBasicBlock(nn.Module):
    # Basic Block for resnet 18 and resnet 34 with Task Routing
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, task_count=20, sigma=0.5):
        super().__init__()

        # We save a different mapping for each task
        self.cur_task = -1

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            # TaskRouter 1
            # TaskRouter(out_channels, task_count, sigma),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * TRBasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * TRBasicBlock.expansion),
            # TaskRouter 2
            TaskRouter(out_channels * TRBasicBlock.expansion, task_count, sigma)
        )
        self.shortcut = nn.Sequential()

        # The shortcut output dimension is not the same with residual function
        # Use 1*1 convolution to match the dimensions
        if stride != 1 or in_channels != TRBasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * TRBasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * TRBasicBlock.expansion),
                # TaskRouter 3
                # TaskRouter(out_channels * BasicBlock.expansion, task_count, sigma)
            )

    def set_task(self, task_id: int):
        self.cur_task = task_id
        # TaskRouter 1
        # self.residual_function[2].set_task(task_id)
        # TaskRouter 2
        # self.residual_function[6].set_task(task_id)
        self.residual_function[5].set_task(task_id)
        # if len(self.shortcut) != 0:
        #     # TaskRouter 3
        #     self.shortcut[2].set_task(task_id)

    def forward(self, x):  # pylint: disable=arguments-differ
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))


class TRMTResNet(nn.Module):

    def __init__(self, block, num_block, num_class=5, num_task=20, sigma=0.5):
        super().__init__()

        self.num_class = num_class
        self.num_task = num_task
        self.sigma = sigma

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        # We use a different inputsize than the original paper
        # (conv2_x's stride is 1)
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # We save a different head for each task
        self.cur_task = -1
        self.fcs = nn.ModuleList([nn.Linear(512 * block.expansion, self.num_class) for _ in range(self.num_task)])

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # We have num_block blocks per layer, the first block could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s, self.num_task, self.sigma))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def set_task(self, task_id: int):
        self.cur_task = task_id
        # Setup Task Mask for Task Routing Layers
        for block in self.conv2_x:
            block.set_task(task_id)
        for block in self.conv3_x:
            block.set_task(task_id)
        for block in self.conv4_x:
            block.set_task(task_id)
        for block in self.conv5_x:
            block.set_task(task_id)

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fcs[self.cur_task](output)
        output_logits = output
        output = F.softmax(output_logits, dim=1)
        output_prob = output
        return output_prob, output_logits


class TRSimpleConvNet(nn.Module):
    # SimpleConvNet https://openreview.net/pdf?id=rJY0-Kcll, https://arxiv.org/pdf/1711.01239.pdf
    # https://github.com/gitabcworld/FewShotLearning/blob/master/model/matching-net-classifier.py
    # https://github.com/markdtw/meta-learning-lstm-pytorch/blob/master/learner.py

    def __init__(self, num_class=5, num_task=20, sigma=0.5):
        super().__init__()

        self.num_class = num_class
        self.num_task = num_task
        self.sigma = sigma

        self.num_filter = 32
        self.num_hidden_fc = 128

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            TaskRouter(self.num_filter, self.num_task, self.sigma),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            TaskRouter(self.num_filter, self.num_task, self.sigma),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            TaskRouter(self.num_filter, self.num_task, self.sigma),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            TaskRouter(self.num_filter, self.num_task, self.sigma),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(self.num_filter * 2 * 2, self.num_hidden_fc)
        self.fc2 = nn.Linear(self.num_hidden_fc, self.num_hidden_fc)

        # We save a different head for each task
        self.cur_task = -1
        self.fc3s = nn.ModuleList([nn.Linear(self.num_hidden_fc, self.num_class) for _ in range(self.num_task)])

    def set_task(self, task_id: int):
        self.cur_task = task_id
        # Setup Task Mask for Task Routing Layers
        self.layer1[2].set_task(task_id)
        self.layer2[2].set_task(task_id)
        self.layer3[2].set_task(task_id)
        self.layer4[2].set_task(task_id)

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = self.layer4(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3s[self.cur_task](output)
        output_logits = output
        output = F.softmax(output_logits, dim=1)
        output_prob = output
        return output_prob, output_logits
