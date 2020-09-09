import math
from typing import Dict, List, Any

import torch.nn as nn
import torch.nn.functional as F


class BasicConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_ratio: float = 0.0, **kwargs):
        super(BasicConv1d, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm1d(out_channels, eps=0.001)
        self.dropout_ratio = dropout_ratio
        self.dropout = nn.Dropout(self.dropout_ratio)

    def forward(self, x):  # pylint: disable=arguments-differ
        x = self.dropout(x)
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class StanfordResidualBlock(nn.Module):

    def __init__(self,
                 stride_flag: bool,
                 last_layer_filter: int,
                 layer_attrs: List[Any],
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(StanfordResidualBlock, self).__init__()

        self.last_layer_filter = last_layer_filter
        stride_size = 2 if stride_flag else 1
        if self.last_layer_filter != layer_attrs[0][0]:
            self.short: nn.Module = nn.Sequential(
                nn.Conv1d(self.last_layer_filter, layer_attrs[0][0], kernel_size=1),
                nn.BatchNorm1d(layer_attrs[0][0]),
                nn.MaxPool1d(stride_size)
            )
        else:
            self.short = nn.MaxPool1d(stride_size)

        convs: List[nn.Module] = []
        for idx, layer_attr in enumerate(layer_attrs):
            layer_filter, kernel_size = layer_attr
            padding_size = math.ceil((kernel_size - 1) / 2)
            dropout_ratio = 0 if idx == 0 else dropout_ratio
            convs += [
                nn.BatchNorm1d(self.last_layer_filter),
                nn.ReLU(),
                nn.Dropout(dropout_ratio),
                nn.Conv1d(
                    self.last_layer_filter, layer_filter,
                    kernel_size=kernel_size, stride=1 if idx == 0 else stride_size, padding=padding_size),
            ]

            self.last_layer_filter = layer_filter

        self.convs = nn.Sequential(*convs)

    def forward(self, x):  # pylint: disable=arguments-differ
        short = self.short(x)
        convs = self.convs(x)
        if short.size()[-1] != convs.size()[-1]:
            short = F.pad(short, (0, 1))
        out = convs + short

        return out


class StanfordFirstResidualBlock(nn.Module):

    def __init__(self,
                 last_layer_filter: int,
                 layer_attrs: List[Any],
                 dropout_ratio: float = 0.0,
                 **kwargs):  # pylint: disable=unused-argument
        super(StanfordFirstResidualBlock, self).__init__()

        self.last_layer_filter = last_layer_filter
        first_residual = []
        for idx, layer_attr in enumerate(layer_attrs):
            layer_filter, kernel_size = layer_attr
            padding_size = kernel_size // 2
            if idx == 0:
                layer: nn.Module = BasicConv1d(
                    self.last_layer_filter, layer_filter, dropout_ratio=dropout_ratio,
                    kernel_size=kernel_size, padding=padding_size)
            else:
                layer = nn.Conv1d(
                    self.last_layer_filter, layer_filter, stride=2, kernel_size=kernel_size, padding=padding_size)
            first_residual.append(layer)
            self.last_layer_filter = layer_filter
        self.convs = nn.Sequential(*first_residual)
        self.short = nn.MaxPool1d(2)

    def forward(self, x):  # pylint: disable=arguments-differ
        short = self.short(x)
        convs = self.convs(x)
        if short.size()[-1] != convs.size()[-1]:
            short = F.pad(short, (0, 1))
        out = convs + short

        return out


class StanfordModel(nn.Module):
    # Residual Layers should be even number, otherwise length might be wrong

    def __init__(self,
                 n_class: int,
                 n_variate: int,
                 signal_len: int,
                 model_structure: Dict[str, List[Any]],
                 dropout_ratio: float = 0.2,
                 **kwargs):  # pylint: disable=unused-argument
        super(StanfordModel, self).__init__()

        self.last_layer_filter = n_variate

        convs = []
        for layer_attr in model_structure['feature_layer']:
            layer_filter, kernel_size = layer_attr
            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, kernel_size=kernel_size)
            convs.append(layer)
            self.last_layer_filter = layer_filter
        self.convs = nn.Sequential(*convs)

        residual_blocks: List[nn.Module] = []
        for idx, layer_attrs in enumerate(model_structure['residual']):
            assert len(layer_attrs) == 2, "Only two layers can be included in one residual block!"

            if idx == 0:
                residual_blocks.append(
                    StanfordFirstResidualBlock(self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio))
            else:
                residual_blocks.append(
                    StanfordResidualBlock(
                        idx % 2 == 0, self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio))
            self.last_layer_filter = layer_attrs[-1][0]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        stride2_times = len(model_structure['residual']) // 2
        length = signal_len // (2 ** stride2_times)
        length = length + 1 if length % 2 != 0 else length
        self.bn = nn.Sequential(
            nn.BatchNorm1d(self.last_layer_filter),
            nn.ReLU()
        )
        self.fc = nn.Linear(self.last_layer_filter * length, n_class)

    def forward(self, x):  # pylint: disable=arguments-differ
        out = self.convs(x)
        out = self.residual_blocks(out)
        out = self.bn(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        output_logits = out
        out = F.softmax(output_logits, dim=1)
        output_prob = out
        return output_prob, output_logits


class MTStanfordModel(nn.Module):
    # Residual Layers should be even number, otherwise length might be wrong

    def __init__(self,
                 num_class: int,
                 n_variate: int,
                 num_task: int,
                 signal_len: int,
                 model_structure: Dict[str, List[Any]],
                 dropout_ratio: float = 0.2,
                 **kwargs):  # pylint: disable=unused-argument
        super(MTStanfordModel, self).__init__()

        self.last_layer_filter = n_variate

        convs = []
        for layer_attr in model_structure['feature_layer']:
            layer_filter, kernel_size = layer_attr
            layer = BasicConv1d(
                self.last_layer_filter, layer_filter, kernel_size=kernel_size)
            convs.append(layer)
            self.last_layer_filter = layer_filter
        self.convs = nn.Sequential(*convs)

        residual_blocks: List[nn.Module] = []
        for idx, layer_attrs in enumerate(model_structure['residual']):
            assert len(layer_attrs) == 2, "Only two layers can be included in one residual block!"

            if idx == 0:
                residual_blocks.append(
                    StanfordFirstResidualBlock(self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio))
            else:
                residual_blocks.append(
                    StanfordResidualBlock(
                        idx % 2 == 0, self.last_layer_filter, layer_attrs, dropout_ratio=dropout_ratio))
            self.last_layer_filter = layer_attrs[-1][0]
        self.residual_blocks = nn.Sequential(*residual_blocks)

        # TODO: Fix length problem
        # stride2_times = len(model_structure['residual']) // 2
        length = 311
        self.bn = nn.Sequential(
            nn.BatchNorm1d(self.last_layer_filter),
            nn.ReLU()
        )

        # We save a different head for each task
        self.cur_task = -1
        self.fcs = nn.ModuleList([nn.Linear(self.last_layer_filter * length, num_class) for _ in range(num_task)])

    def set_task(self, task_id: int):
        self.cur_task = task_id

    def forward(self, x):  # pylint: disable=arguments-differ
        output = self.convs(x)
        output = self.residual_blocks(output)
        output = self.bn(output)
        # print(output.shape)
        output = output.view(output.size(0), -1)
        output = self.fcs[self.cur_task](output)
        output_logits = output
        output = F.softmax(output_logits, dim=1)
        output_prob = output
        return output_prob, output_logits
