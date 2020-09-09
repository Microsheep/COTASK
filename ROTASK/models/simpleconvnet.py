import torch
import torch.nn as nn
import torch.nn.functional as F


class MTSimpleConvNet(nn.Module):
    # SimpleConvNet https://openreview.net/pdf?id=rJY0-Kcll, https://arxiv.org/pdf/1711.01239.pdf
    # https://github.com/gitabcworld/FewShotLearning/blob/master/model/matching-net-classifier.py
    # https://github.com/markdtw/meta-learning-lstm-pytorch/blob/master/learner.py

    def __init__(self, num_class=5, num_task=20):
        super().__init__()

        self.num_class = num_class
        self.num_task = num_task

        self.num_filter = 32
        self.num_hidden_fc = 128

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(self.num_filter * 2 * 2, self.num_hidden_fc)
        self.fc2 = nn.Linear(self.num_hidden_fc, self.num_hidden_fc)

        # We save a different head for each task
        self.cur_task = -1
        self.fc3s = nn.ModuleList([nn.Linear(self.num_hidden_fc, self.num_class) for _ in range(self.num_task)])

        # TODO: figure out if this is needed
        # Initialize layers
        # self.weights_init(self.layer1)
        # self.weights_init(self.layer2)
        # self.weights_init(self.layer3)
        # self.weights_init(self.layer4)

    # def weights_init(self, module):
    #     for m in module.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
    #             nn.init.constant(m.bias, 0)
    #         elif isinstance(m, nn.BatchNorm2d):
    #             m.weight.data.fill_(1)
    #             m.bias.data.zero_()

    def set_task(self, task_id: int):
        self.cur_task = task_id

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


class CSMTSimpleConvNet(nn.Module):

    def __init__(self, num_class=5, num_task=20, alpha_self=0.9):
        super().__init__()

        self.num_class = num_class
        self.num_task = num_task

        self.alpha_self = alpha_self
        self.alpha_others = (1 - self.alpha_self) / (self.num_task - 1)

        self.num_filter = 32
        self.num_hidden_fc = 128

        # We save a different head for each task
        self.cur_task = -1

        self.layer1s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(3, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) for _ in range(self.num_task)])
        self.layer2s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) for _ in range(self.num_task)])
        self.layer3s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) for _ in range(self.num_task)])
        self.layer4s = nn.ModuleList([nn.Sequential(
            nn.Conv2d(self.num_filter, self.num_filter, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(self.num_filter),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        ) for _ in range(self.num_task)])

        self.fc1s = nn.ModuleList([nn.Linear(self.num_filter * 4, self.num_hidden_fc) for _ in range(self.num_task)])
        self.fc2s = nn.ModuleList([nn.Linear(self.num_hidden_fc, self.num_hidden_fc) for _ in range(self.num_task)])
        self.fc3s = nn.ModuleList([nn.Linear(self.num_hidden_fc, self.num_class) for _ in range(self.num_task)])

        # Cross-stitch units
        self.cs_unit = nn.ParameterList([nn.Parameter(
            torch.eye(self.num_task) * (self.alpha_self - self.alpha_others)
            + torch.ones(self.num_task, self.num_task) * self.alpha_others
        ) for _ in range(6)])

    def set_task(self, task_id: int):
        self.cur_task = task_id

    def forward(self, x):  # pylint: disable=arguments-differ, too-many-locals, too-many-branches
        current_outputs_l1 = []
        for ts_layer1 in self.layer1s:
            current_outputs_l1.append(ts_layer1(x))

        current_outputs_l2 = []
        for to_task_id, ts_layer2 in enumerate(self.layer2s):
            cum_l2 = None
            for from_task_id, c_o in enumerate(current_outputs_l1):
                tmp_l2 = self.cs_unit[0][from_task_id][to_task_id] * c_o
                cum_l2 = tmp_l2 if cum_l2 is None else cum_l2 + tmp_l2
            current_outputs_l2.append(ts_layer2(cum_l2))

        current_outputs_l3 = []
        for to_task_id, ts_layer3 in enumerate(self.layer3s):
            cum_l3 = None
            for from_task_id, c_o in enumerate(current_outputs_l2):
                tmp_l3 = self.cs_unit[1][from_task_id][to_task_id] * c_o
                cum_l3 = tmp_l3 if cum_l3 is None else cum_l3 + tmp_l3
            current_outputs_l3.append(ts_layer3(cum_l3))

        current_outputs_l4 = []
        for to_task_id, ts_layer4 in enumerate(self.layer4s):
            cum_l4 = None
            for from_task_id, c_o in enumerate(current_outputs_l3):
                tmp_l4 = self.cs_unit[2][from_task_id][to_task_id] * c_o
                cum_l4 = tmp_l4 if cum_l4 is None else cum_l4 + tmp_l4
            current_outputs_l4.append(ts_layer4(cum_l4))

        current_outputs_fc1 = []
        for to_task_id, ts_fc1 in enumerate(self.fc1s):
            cum_fc1 = None
            for from_task_id, c_o in enumerate(current_outputs_l4):
                tmp_fc1 = self.cs_unit[3][from_task_id][to_task_id] * c_o.view(c_o.size(0), -1)
                cum_fc1 = tmp_fc1 if cum_fc1 is None else cum_fc1 + tmp_fc1
            current_outputs_fc1.append(ts_fc1(cum_fc1))

        current_outputs_fc2 = []
        for to_task_id, ts_fc2 in enumerate(self.fc2s):
            cum_fc2 = None
            for from_task_id, c_o in enumerate(current_outputs_fc1):
                tmp_fc2 = self.cs_unit[4][from_task_id][to_task_id] * c_o
                cum_fc2 = tmp_fc2 if cum_fc2 is None else cum_fc2 + tmp_fc2
            current_outputs_fc2.append(ts_fc2(cum_fc2))

        current_outputs_fc3 = []
        for to_task_id, ts_fc3 in enumerate(self.fc3s):
            cum_fc3 = None
            for from_task_id, c_o in enumerate(current_outputs_fc2):
                tmp_fc3 = self.cs_unit[5][from_task_id][to_task_id] * c_o
                cum_fc3 = tmp_fc3 if cum_fc3 is None else cum_fc3 + tmp_fc3
            current_outputs_fc3.append(ts_fc3(cum_fc3))

        output = current_outputs_fc3[self.cur_task]
        output_logits = output
        output = F.softmax(output_logits, dim=1)
        output_prob = output
        return output_prob, output_logits
