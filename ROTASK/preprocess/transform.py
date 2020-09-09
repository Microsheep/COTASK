import random

from typing import Iterable, Callable, Optional

import numpy as np


class Compose():
    def __init__(self, transforms: Iterable[Callable]):
        self.transforms = transforms

    def __call__(self, dat):
        for t in self.transforms:
            dat = t(dat)
        return dat

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += f'\n    {t}'
        format_string += '\n)'
        return format_string


class RandomShift():
    def __init__(self, shift_tolerance: int):
        self.shift_tolerance = shift_tolerance

    def __call__(self, lead_data: np.array) -> np.array:
        roll_shift = random.randint(0, self.shift_tolerance)
        lead_data = lead_data[:, roll_shift:]
        lead_data = np.pad(lead_data, ((0, 0), (0, roll_shift)), 'constant', constant_values=(0, 0))
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(shift_tolerance={self.shift_tolerance})'


class ZNormalize_1D():
    def __init__(self, mean: Optional[np.array] = None, std: Optional[np.array] = None, eps: float = 1e-5):
        self.mean = mean
        self.std = std
        self.eps = eps

    def __call__(self, lead_data: np.array) -> np.array:
        if self.mean is None:
            self.mean = np.mean(lead_data, axis=-1).reshape([-1, 1])
        if self.std is None:
            self.std = np.std(lead_data, axis=-1).reshape([-1, 1])
            self.std[self.std < self.eps] = 1.0
        lead_data = (lead_data - self.mean) / self.std
        return lead_data

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, eps={self.eps})'


class RandomCrop():
    def __init__(self, length: int, validate: bool = False):
        self.length = length
        self.validate = validate

    def __call__(self, lead_data: np.array) -> np.array:
        max_startpoint = len(lead_data[0]) - self.length
        start_point = random.randint(0, max_startpoint) if not self.validate else 0
        return lead_data[:, start_point:start_point + self.length]

    def __repr__(self):
        return f'{self.__class__.__name__}(length={self.length})'
