import random
import logging
import copy

from typing import Tuple, List, Dict, Optional, Callable, Union

from sklearn.model_selection import StratifiedShuffleSplit

from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.datasets import CIFAR100

from ROTASK.utils.utils import safe_dir


# Initiate Logger
logger = logging.getLogger(__name__)


class MTCIFAR100Dataset(Dataset):
    # Coarse Class and Fine Class
    coarse2fine = {
        "reptiles": ["crocodile", "dinosaur", "lizard", "snake", "turtle"],
        "fish": ["aquarium_fish", "flatfish", "ray", "shark", "trout"],
        "aquatic_mammal": ["beaver", "dolphin", "otter", "seal", "whale"],
        "small_mammal": ["hamster", "mouse", "rabbit", "shrew", "squirrel"],
        "medium_mammal": ["fox", "porcupine", "possum", "raccoon", "skunk"],
        "large_carnivore": ["bear", "leopard", "lion", "tiger", "wolf"],
        "large_omnivore_herbivore": ["camel", "cattle", "chimpanzee", "elephant", "kangaroo"],
        "insect": ["bee", "beetle", "butterfly", "caterpillar", "cockroach"],
        "non_insect_invertebrates": ["crab", "lobster", "snail", "spider", "worm"],
        "people": ["baby", "boy", "girl", "man", "woman"],
        "tree": ["maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree"],
        "flower": ["orchid", "poppy", "rose", "sunflower", "tulip"],
        "fruit_and_vegetable": ["apple", "mushroom", "orange", "pear", "sweet_pepper"],
        "food_container": ["bottle", "bowl", "can", "cup", "plate"],
        "electrical_device": ["clock", "keyboard", "lamp", "telephone", "television"],
        "furniture": ["bed", "chair", "couch", "table", "wardrobe"],
        "building": ["bridge", "castle", "house", "road", "skyscraper"],
        "natural_outdoor_scene": ["cloud", "forest", "mountain", "plain", "sea"],
        "vehicles_common": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
        "vehicles_uncommon": ["lawn_mower", "rocket", "streetcar", "tank", "tractor"],
    }
    # coarse2fine = {
    #     "all": [
    #         "crocodile", "dinosaur", "lizard", "snake", "turtle",
    #         "aquarium_fish", "flatfish", "ray", "shark", "trout",
    #         "beaver", "dolphin", "otter", "seal", "whale",
    #         "hamster", "mouse", "rabbit", "shrew", "squirrel",
    #         "fox", "porcupine", "possum", "raccoon", "skunk",
    #         "bear", "leopard", "lion", "tiger", "wolf",
    #         "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    #         "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    #         "crab", "lobster", "snail", "spider", "worm",
    #         "baby", "boy", "girl", "man", "woman",
    #         "maple_tree", "oak_tree", "palm_tree", "pine_tree", "willow_tree",
    #         "orchid", "poppy", "rose", "sunflower", "tulip",
    #         "apple", "mushroom", "orange", "pear", "sweet_pepper",
    #         "bottle", "bowl", "can", "cup", "plate",
    #         "clock", "keyboard", "lamp", "telephone", "television",
    #         "bed", "chair", "couch", "table", "wardrobe",
    #         "bridge", "castle", "house", "road", "skyscraper",
    #         "cloud", "forest", "mountain", "plain", "sea",
    #         "bicycle", "bus", "motorcycle", "pickup_truck", "train",
    #         "lawn_mower", "rocket", "streetcar", "tank", "tractor"
    #     ]
    # }
    fine2coarse = {}
    for cc, fcs in coarse2fine.items():
        for fc in fcs:
            fine2coarse[fc] = cc
    coarseclasses = list(coarse2fine.keys())

    # Mean STD for CIFAR_100
    transform_mean_std = {
        "mean": (0.5070751592371323, 0.48654887331495095, 0.4409178433670343),
        "std": (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
    }

    def __init__(self,
                 dataset_cache_root: str = '/tmp/cifar100/',
                 split_ratio: Tuple[float, float] = (0.8, 0.2),
                 training_noise_ratio: float = 0.0,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 random_seed: Optional[int] = None):

        # Make sure cache directory exists
        safe_dir(dataset_cache_root)
        self.cifar100_train = CIFAR100(dataset_cache_root, download=True, train=True)
        self.cifar100_test = CIFAR100(dataset_cache_root, download=True, train=False)
        assert self.cifar100_train.classes == self.cifar100_test.classes
        self.fineclasses = self.cifar100_train.classes
        assert self.cifar100_train.class_to_idx == self.cifar100_test.class_to_idx
        self.fineclass2fineid = self.cifar100_train.class_to_idx
        self.fineid2fineclass = {v: k for k, v in self.fineclass2fineid.items()}

        # Calculate Basic Statistics
        self.testid_shift = len(self.cifar100_train)
        self.data_cnt = len(self.cifar100_train) + len(self.cifar100_test)
        self.class_cnt = len(self.fineclasses)

        # Init Random Seed
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)

        # Save training_noise_ratio
        self.training_noise_ratio = training_noise_ratio
        logger.info("Setting training noise ratio to %s", self.training_noise_ratio)
        # TODO: Separate Random State for Noise
        # random.seed(self.random_seed)

        # Split data into tasks
        self.split_ratio = split_ratio
        self.task_split_indices_dict, self.noise_mapping = self._gen_task_split()

        # Save Global Transformations
        self.transform = transform

    def _gen_task_split(self):
        # Split Train and Validation
        train_ratio, valid_ratio = self.split_ratio
        sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_ratio, random_state=self.random_seed)
        train_id, valid_id = next(sss.split(list(range(0, len(self.cifar100_train))), self.cifar100_train.targets))
        test_id = list(range(self.testid_shift, self.testid_shift + len(self.cifar100_test)))

        # Double Check that there isn't overlapping indexes
        assert set(train_id) | set(valid_id) == set(range(len(self.cifar100_train)))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()

        # Generate Each Task
        task_split_indices_dict: Dict[str, Dict[str, List[int]]] = {
            cc: {"train": [], "valid": [], "test": []} for cc in self.coarseclasses}
        noise_mapping: Dict[int, int] = {}
        for idx in train_id:
            fc = self.fineid2fineclass[self.cifar100_train.targets[idx]]
            task_split_indices_dict[self.fine2coarse[fc]]["train"].append(idx)
            # Add training label noise
            if random.random() < self.training_noise_ratio:
                noise_mapping[idx] = random.choice(range(len(self.coarse2fine[self.fine2coarse[fc]])))
        for idx in valid_id:
            fc = self.fineid2fineclass[self.cifar100_train.targets[idx]]
            task_split_indices_dict[self.fine2coarse[fc]]["valid"].append(idx)
        for idx in test_id:
            fc = self.fineid2fineclass[self.cifar100_test.targets[idx - self.testid_shift]]
            task_split_indices_dict[self.fine2coarse[fc]]["test"].append(idx)

        # Double Check data count
        train_count = sum([len(tid["train"]) for cc, tid in task_split_indices_dict.items()])
        assert train_count == train_ratio * len(self.cifar100_train)
        valid_cnt = sum([len(tid["valid"]) for cc, tid in task_split_indices_dict.items()])
        assert valid_cnt == valid_ratio * len(self.cifar100_train)
        test_cnt = sum([len(tid["test"]) for cc, tid in task_split_indices_dict.items()])
        assert test_cnt == len(self.cifar100_test)

        return task_split_indices_dict, noise_mapping

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        # Get Train/Valid or Test Data
        if idx >= self.testid_shift:
            data, fineid = self.cifar100_test[idx - self.testid_shift]
        else:
            data, fineid = self.cifar100_train[idx]
        # Generate Metadata
        fineclass = self.fineid2fineclass[fineid]
        coarseclass = self.fine2coarse[fineclass]
        fineid_in_coarse = self.coarse2fine[coarseclass].index(fineclass)
        coarseid = self.coarseclasses.index(coarseclass)
        metadata = {
            "fineid": fineid,
            "fineclass": fineclass,
            "coarseid": coarseid,
            "coarseclass": coarseclass,
            "fineid_in_coarse": fineid_in_coarse,
            "target": self.noise_mapping.get(idx, fineid_in_coarse),
        }
        return data, metadata


class MTCIFAR100TaskDataset(Dataset):
    def __init__(self,
                 dataset: MTCIFAR100Dataset,
                 task_name: str,
                 transform: Optional[Tuple[Callable, ...]] = None):
        self.dataset = dataset
        self.task_name = task_name
        assert task_name in dataset.coarseclasses

        self.transform = transform
        self.indices: Dict[str, List[int]] = dataset.task_split_indices_dict[task_name]
        self.all_indices: List[int] = self.indices["train"] + self.indices["valid"] + self.indices["test"]

        self.class_cnt = len(dataset.coarse2fine[task_name])
        self.data_cnt = len(self.all_indices)

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        data, metadata = self.dataset[self.all_indices[idx]]
        if self.transform is not None:
            data = Compose(self.transform)(data)
        return data, metadata


class MergedMTCIFAR100TaskDataset(Dataset):
    def __init__(self,
                 datasets: List[MTCIFAR100TaskDataset],
                 transform: Optional[Tuple[Callable, ...]] = None,
                 random_seed: Optional[int] = None):

        self.datasets = datasets

        # Setup Random State
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        random.seed(self.random_seed)

        # Random the class_mappings
        # TODO: Extend to different class_cnt per dataset
        self.class_mappings = []
        for d in self.datasets:
            d_m = list(range(d.class_cnt))
            random.shuffle(d_m)
            self.class_mappings.append(d_m)
            logger.info("Class Mapping for %s: %s", d.task_name, d_m)

        # Merge one class from each dataset into one
        self.indices: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}
        self.dataset_id: Dict[str, List[int]] = {"train": [], "valid": [], "test": []}
        for did, d in enumerate(self.datasets):
            cur_offset = 0
            orig_train_cnt = len(d.indices["train"])
            sampled_train = random.sample(
                range(cur_offset, cur_offset + orig_train_cnt), orig_train_cnt // len(self.datasets))
            self.indices["train"] += sampled_train
            self.dataset_id["train"] += [did] * len(sampled_train)

            cur_offset += orig_train_cnt
            orig_valid_cnt = len(d.indices["valid"])
            sampled_valid = random.sample(
                range(cur_offset, cur_offset + orig_valid_cnt), orig_valid_cnt // len(self.datasets))
            self.indices["valid"] += sampled_valid
            self.dataset_id["valid"] += [did] * len(sampled_valid)

            cur_offset += orig_valid_cnt
            orig_test_cnt = len(d.indices["test"])
            sampled_test = random.sample(
                range(cur_offset, cur_offset + orig_test_cnt), orig_test_cnt // len(self.datasets))
            self.indices["test"] += sampled_test
            self.dataset_id["test"] += [did] * len(sampled_test)
        self.all_indices: List[int] = self.indices["train"] + self.indices["valid"] + self.indices["test"]
        self.all_dataset_id: List[int] = self.dataset_id["train"] + self.dataset_id["valid"] + self.dataset_id["test"]

        self.transform = transform

        # TODO: Make it possible for different class count
        self.class_cnt = self.datasets[0].class_cnt
        self.data_cnt = len(self.all_indices)

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        dataset_id = self.all_dataset_id[idx]
        data, metadata = self.datasets[dataset_id][self.all_indices[idx]]
        # Map class to merged class
        metadata["target"] = self.class_mappings[dataset_id][metadata["target"]]
        if self.transform is not None:
            data = Compose(self.transform)(data)
        return data, metadata


class MTCIFAR100TaskDatasetSubset(Dataset):
    def __init__(self,
                 dataset: Union[MTCIFAR100TaskDataset, MergedMTCIFAR100TaskDataset],
                 data_type: str,
                 transform: Optional[Tuple[Callable, ...]] = None):
        self.dataset = dataset
        self.data_type = data_type
        assert data_type in ["train", "valid", "test"]

        self.transform = transform
        if self.data_type == "train":
            self.all_indices = range(0,
                                     len(self.dataset.indices["train"]))
        elif self.data_type == "valid":
            self.all_indices = range(len(self.dataset.indices["train"]),
                                     len(self.dataset.indices["train"] + self.dataset.indices["valid"]))
        else:
            self.all_indices = range(len(self.dataset.indices["train"] + self.dataset.indices["valid"]),
                                     len(self.dataset.all_indices))

        self.class_cnt = dataset.class_cnt
        self.data_cnt = len(self.all_indices)

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        data, metadata = self.dataset[self.all_indices[idx]]
        if self.transform is not None:
            data = Compose(self.transform)(data)
        return data, metadata


class MTCIFAR100TaskDatasetDownsampleSubset(Dataset):
    def __init__(self,
                 dataset: MTCIFAR100TaskDataset,
                 random_subset_ratio: Tuple[Optional[float], Optional[float], Optional[float]] = (None, None, None),
                 random_seed: Optional[int] = None):
        self.dataset = dataset
        # Copy from Parent
        self.task_name = self.dataset.task_name
        self.transform = self.dataset.transform

        self.random_subset_ratio = random_subset_ratio

        # Setup Random State
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        random.seed(self.random_seed)

        # Get random indices
        self.indices: Dict[str, List[int]] = copy.deepcopy(self.dataset.indices)
        if random_subset_ratio[0] is not None:
            self.indices["train"] = random.sample(
                self.indices["train"], k=round(len(self.indices["train"]) * random_subset_ratio[0]))
        if random_subset_ratio[1] is not None:
            self.indices["valid"] = random.sample(
                self.indices["valid"], k=round(len(self.indices["valid"]) * random_subset_ratio[1]))
        if random_subset_ratio[2] is not None:
            self.indices["test"] = random.sample(
                self.indices["test"], k=round(len(self.indices["test"]) * random_subset_ratio[2]))
        self.all_indices: List[int] = self.indices["train"] + self.indices["valid"] + self.indices["test"]

        self.class_cnt = self.dataset.class_cnt
        self.data_cnt = len(self.all_indices)

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):
        # NOTE: Get data by MTCIFAR100TaskDataset's parent dataset due to mimic original MTCIFAR100TaskDataset
        data, metadata = self.dataset.dataset[self.all_indices[idx]]
        if self.transform is not None:
            data = Compose(self.transform)(data)
        return data, metadata

