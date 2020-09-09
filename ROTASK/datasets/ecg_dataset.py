import re
import os
import sys
import json
import math
import random
import logging

from typing import Tuple, List, Dict, Sequence, Optional, Callable, Union, ClassVar, Set
from datetime import datetime

import torch
from torch.utils.data import Dataset

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from prettytable import PrettyTable
from tqdm import tqdm, trange

from ROTASK.datasets.ecg_data_model import LEAD_NAMES, LEAD_LENGTH, LEAD_DROP_LEN
from ROTASK.datasets.ecg_data_model import Base, ECGtoLVH, ECGLeads, TaintCode
from ROTASK.datasets.ecg_data_model import unpack_leads, unpack_statements
from ROTASK.preprocess.transform import Compose


# Initiate Logger
logger = logging.getLogger(__name__)


class ECGDataset(Dataset):  # pylint: disable=too-many-instance-attributes
    MINOR_STRATIFY_CNT: ClassVar[Dict[str, int]] = {
        "ECGtoLVH": 20,
        "ECGLeads": 20,
    }
    # Stratify Tolerence: [train-test, train-valid], (rtol, atol)
    STRATIFY_TOL: ClassVar[Dict[str, List[Tuple[float, float]]]] = {
        "ECGtoLVH": [(0.0, 0.01), (0.0, 0.01)],
        "ECGLeads": [(0.0, 0.01), (0.0, 0.01)],
    }

    CLASS_NAME_MAPPING = {
        "transform_lvh_level": {
            0: "Not LVH",
            1: "LVH",
        },
        "transform_statements_cleansr": {
            0: "SR only",
            1: "Not SR only",
        },
    }

    # Use to identify major tasks
    MAJOR_TASK_NAME = "===ECHO_LVH==="

    # Base offset for task and item
    BASE_TASK_OFFSET = 1000000

    def __init__(self,
                 db_location: str,
                 target_table: Base = ECGtoLVH,
                 target_attr: str = 'LVH_LVmass_level',
                 target_attr_transform: Optional[Callable] = None,
                 stratify_attr: Sequence[str] = ('gender', 'EKG_age',
                                                 'his_HTN', 'his_DM', 'his_MI', 'his_HF', 'his_stroke', 'his_CKD'),
                 train_test_patient_possible_overlap: bool = False,
                 split_ratio: Tuple[float, float, float] = (0.7, 0.1, 0.2),
                 preprocess_lead: Optional[Callable] = None,
                 transform: Optional[Tuple[Callable, ...]] = None,
                 random_seed: Optional[int] = None):

        # Save DB Location
        self.db_location = db_location

        # Preprocess functions
        self.preprocess_lead = preprocess_lead
        self.transform = transform

        # Init pid as None to force `connect_to_db` function get new connection
        self.pid = None
        self.session, self.Session, self.engine = None, None, None

        # Set using metadata from `ECGDL.datasets.ecg_data_model`
        self.variate_cnt = len(LEAD_NAMES)
        self.data_len = LEAD_LENGTH - LEAD_DROP_LEN

        # Save Attributes
        self.target_table = target_table
        self.target_attr = target_attr
        self.stratify_attr = list(stratify_attr)
        self.train_test_patient_possible_overlap = train_test_patient_possible_overlap
        self.target_attr_transform = target_attr_transform

        if self.target_attr_transform is None and self.target_attr in self.CLASS_NAME_MAPPING:
            self.name_mapping = self.CLASS_NAME_MAPPING[self.target_attr]
        elif self.target_attr_transform is not None and self.target_attr_transform.__name__ in self.CLASS_NAME_MAPPING:
            self.name_mapping = self.CLASS_NAME_MAPPING[self.target_attr_transform.__name__]
        else:
            logger.warning("Name Mapping not recognized!")
            self.name_mapping = {}

        # Set limitations
        self.minor_stratify_cnt = self.MINOR_STRATIFY_CNT[self.target_table.__name__]
        self.stratify_tol = self.STRATIFY_TOL[self.target_table.__name__]

        # Get all mhash with lead data
        self.ecg_data, self.mhash_data, \
            self.data_cnt, self.class_cnt, self.cid2target, self.cid2name, self.stratify_cols = self._get_mhash()
        self.target2cid = {v: k for k, v in self.cid2target.items()}
        self.name2cid = {v: k for k, v in self.cid2name.items()}

        # Separate the mhash
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        self.split_ratio = split_ratio
        self.split_indices_dict = self._stratify_split()

        # Load statement extraction mapping
        with open("./ROTASK/datasets/ecg_mapping.json", "r") as f:
            self.statement_match_dict = json.load(f)

        # Save targeted tasks
        self.selected_tasks: List[str] = []
        if self.target_table == ECGtoLVH:
            self.selected_tasks = [self.MAJOR_TASK_NAME]

    def connect_to_db(self):
        if os.getpid() != self.pid:
            # Set pid for this instance
            self.pid = os.getpid()

            # Close Database Connections if there is already a connection
            if self.session is not None:
                # logger.info("PID %s: Closing Existing Session!", self.pid)
                self.session.close()
            if self.engine is not None:
                # logger.info("PID %s: Closing Existing Engine!", self.pid)
                self.engine.dispose()

            # Re-connect to Database
            try:
                self.engine = create_engine(f"sqlite:///{self.db_location}")
                self.Session = sessionmaker(bind=self.engine)
                self.session = self.Session()
            except Exception:
                logger.critical("PID %s: Can't connect to DB at `%s` !", self.pid, self.db_location)
                sys.exit(-1)

        return self.session

    def _get_mhash(self) -> Tuple[pd.DataFrame, pd.DataFrame, int, int, Dict[int, int], Dict[int, str], List[str]]:  # pylint: disable=too-many-statements
        # Get Session
        session = self.connect_to_db()

        if self.target_table == ECGtoLVH:
            # Get all possible mhashes that have LeadData with needed stratify_cols
            stratify_cols = self.stratify_attr + [self.target_attr]

            q = session.query(self.target_table.mhash, self.target_table.patient_id, self.target_table.req_no,
                              *[getattr(self.target_table, sc) for sc in stratify_cols])
            q = q.filter(self.target_table.leads.has(ECGLeads.taint_code < TaintCode.DEFAULT_SAFE))
            # Filter out one pair per patient and heartbeat count >= 8
            q = q.filter(self.target_table.LVH_Single == 1)
            q = q.filter(self.target_table.leads.has(ECGLeads.heartbeat_cnt >= 8))

            # Query the data from database
            ecg_data = pd.read_sql(q.statement, session.bind)

            target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
            logger.info("Target Count:\n%s", target_cnt)
        elif self.target_table == ECGLeads:
            # We do not have stratify columns when using large DB on statements
            assert len(self.stratify_attr) == 0
            assert self.target_attr == "statements"
            assert self.target_attr_transform == ECGDataset.transform_statements_cleansr  # pylint: disable=comparison-with-callable
            assert self.train_test_patient_possible_overlap

            q = session.query(self.target_table.mhash, self.target_table.patient_id, self.target_table.req_no,
                              self.target_table.statements)
            q = q.filter(self.target_table.taint_code < TaintCode.DEFAULT_SAFE)
            # Filter out only data after 2013
            q = q.filter(self.target_table.report_dt > datetime(year=2013, month=1, day=1))
            # Filter out one pair per patient and heartbeat count >= 8
            q = q.filter(self.target_table.heartbeat_cnt >= 8)

            # Query the data from database
            ecg_data = pd.read_sql(q.statement, session.bind)
        else:
            logger.error("Target table method not found for %s!", self.target_table)
            raise LookupError

        # Transform Target Column if target_attr_transform exist
        if self.target_attr_transform is not None:
            ecg_data[f"{self.target_attr}_transformed"] = ecg_data[self.target_attr].map(self.target_attr_transform)
            self.target_attr = f"{self.target_attr}_transformed"
            target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
            logger.info("Transformed Target Count:\n%s", target_cnt)

        # Drop Unwanted Data
        ecg_data = ecg_data[ecg_data[self.target_attr] != -1].reset_index()
        target_cnt = ecg_data.groupby([self.target_attr]).count()["mhash"].reset_index()
        logger.info("Drop Target Count:\n%s", target_cnt)

        # Need to map to 0 ~ n class id
        cid2target = target_cnt.to_dict()[self.target_attr]
        logger.info("cid2target: %s", cid2target)

        class_cnt = len(cid2target.keys())
        logger.info("Class Count: %s", class_cnt)
        if class_cnt < 2:
            logger.warning("Less than two class found!")

        # Map the name of the class
        cid2name = cid2target
        if self.name_mapping is not None:
            cid2name = {k: self.name_mapping[v] for k, v in cid2name.items()}
        logger.info("cid2name: %s", cid2name)

        # Clean and Bin the columns for stratify split
        if 'EKG_age' in self.stratify_attr:
            ecg_data['EKG_age'] = ecg_data['EKG_age'].fillna(value=-5)
            age_bins = [-10, -1, 20, 30, 40, 50, 60, 70, 80, 200]
            # Map back to 0 ~ k
            ecg_data['EKG_age'] = np.digitize(ecg_data['EKG_age'], age_bins) - 3

        # Create strat_id column
        stratify_cols = self.stratify_attr + [self.target_attr]
        ecg_data['strat_id'] = '#'
        for col in stratify_cols:
            ecg_data['strat_id'] = ecg_data['strat_id'] + ecg_data[col].astype(str).values + "#"

        # Group by stratify id
        count_groups = ecg_data.groupby('strat_id').count()

        # Take out minor group which number smaller than MINOR_STRATIFY_COUNT
        minor_index = count_groups['mhash'][count_groups['mhash'] < self.minor_stratify_cnt].index
        logger.info("Number of minor stratify groups (instance count < %s): %s with %s instances",
                    self.minor_stratify_cnt, len(minor_index), len(ecg_data[ecg_data['strat_id'].isin(minor_index)]))
        minor_tar, minor_tar_cnt = np.unique([i.split("#")[-2] for i in minor_index], return_counts=True)
        minor_tar_df = pd.DataFrame(data={self.target_attr: minor_tar, 'cnt': minor_tar_cnt})
        logger.info("Minor Target Distribution:\n%s", minor_tar_df)

        # Switch Minor strat_id back to target
        ecg_data.loc[ecg_data['strat_id'].isin(minor_index), 'strat_id'] = ecg_data[self.target_attr].astype(str)

        # Save only mhash and strat_id
        mhash_data = ecg_data[['mhash', 'patient_id', 'req_no', 'strat_id', *stratify_cols]]

        # Get Total Data Count
        data_cnt = len(mhash_data)
        logger.info("Data Count: %s", data_cnt)
        if data_cnt == 0:
            logger.warning("No data Found!")

        return ecg_data, mhash_data, data_cnt, class_cnt, cid2target, cid2name, stratify_cols

    def _stratify_split(self) -> Dict[str, List[int]]:  # pylint: disable=too-many-locals
        train_ratio, valid_ratio, test_ratio = self.split_ratio

        # Setup Random State
        random.seed(self.random_seed)

        if self.train_test_patient_possible_overlap:
            # Split the testing data without patient overlapping
            unique_patient_id = np.unique(self.mhash_data['patient_id'])
            testing_patient_id = random.sample(unique_patient_id.tolist(), k=round(test_ratio * len(unique_patient_id)))
            test_id = self.mhash_data[self.mhash_data['patient_id'].isin(testing_patient_id)].index
            train_valid_id = self.mhash_data[~self.mhash_data['patient_id'].isin(testing_patient_id)].index

            # Check that no patient_id overlaps
            patient_ids = self.mhash_data['patient_id']
            assert set(patient_ids.iloc[test_id]) & set(patient_ids.iloc[train_valid_id]) == set()
        else:
            # Stratify Split the testing data without considering patient overlapping
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=self.random_seed)
            train_valid_id, test_id = next(sss.split(self.mhash_data['mhash'], self.mhash_data['strat_id']))

            # Warns if patient_id overlaps
            patient_ids = self.mhash_data['patient_id']
            if set(patient_ids.iloc[test_id]) & set(patient_ids.iloc[train_valid_id]) != set():
                logger.warning("Setting set to not consider train/test patient possible overlap but patient overlaps!")

        validintrain_ratio = valid_ratio / (train_ratio + valid_ratio)
        sss = StratifiedShuffleSplit(n_splits=1, test_size=validintrain_ratio, random_state=self.random_seed)
        train_id, valid_id = next(sss.split(
            self.mhash_data['mhash'].iloc[train_valid_id], self.mhash_data['strat_id'].iloc[train_valid_id]))

        # Transform back to train_valid_id index
        train_id = train_valid_id[train_id]
        valid_id = train_valid_id[valid_id]

        # Double Check that there isn't overlapping indexes
        assert set(train_id) | set(valid_id) | set(test_id) == set(range(self.data_cnt))
        assert set(train_id) & set(valid_id) == set()
        assert set(train_id) & set(test_id) == set()
        assert set(valid_id) & set(test_id) == set()

        # Check on Stratify Quality
        stratify_cols = self.stratify_attr + [self.target_attr]
        for col in stratify_cols:
            logger.info("Stratifying Column: %s", col)

            # Get the distribution for certain stratify_attr
            u_train, cnt_train = np.unique(self.ecg_data[col].iloc[train_id], return_counts=True)
            u_valid, cnt_valid = np.unique(self.ecg_data[col].iloc[valid_id], return_counts=True)
            u_test, cnt_test = np.unique(self.ecg_data[col].iloc[test_id], return_counts=True)
            d_train = cnt_train / sum(cnt_train)
            d_valid = cnt_valid / sum(cnt_valid)
            d_test = cnt_test / sum(cnt_test)

            # Check that attribute list is the same
            assert set(u_train) == set(u_valid) and set(u_valid) == set(u_test), \
                (set(u_train), set(u_valid), set(u_test))

            tb = PrettyTable()
            tb.field_names = ["Type", "Training", "Validation", "Testing"]
            for name, dn, dv, dt, cn, cv, ct in zip(u_train, d_train, d_valid, d_test, cnt_train, cnt_valid, cnt_test):
                tb.add_row([name, f"{cn} ({dn:.5f})", f"{cv} ({dv:.5f})", f"{ct} ({dt:.5f})"])
            tb.align = "r"
            logger.info("Split Distribution: %s\n%s", col, tb.get_string())

            # Check that distribution is similar
            large_tol, small_tol = self.stratify_tol
            assert np.allclose(d_train, d_valid, *small_tol), np.isclose(d_train, d_valid, *small_tol)
            assert np.allclose(d_train, d_test, *large_tol), np.isclose(d_train, d_test, *large_tol)
            assert np.allclose(d_valid, d_test, *large_tol), np.isclose(d_valid, d_test, *large_tol)

        return {
            "train": train_id,
            "valid": valid_id,
            "test": test_id,
        }

    @staticmethod
    def transform_lvh_level(x: int) -> int:
        # Skip unknown/test data
        if x == -1:
            return -1
        return 1 if x >= 2 else 0

    @staticmethod
    def transform_statements_cleansr(x: str) -> int:
        statements = unpack_statements(x)
        all_safe = []
        for statement in statements:
            sid, sname, _exp = statement
            # Removed to preserve data related information
            all_safe.append(sid == "SR" or sname.upper() == "SINUS RHYTHM")
        return 0 if all(all_safe) else 1

    def extract_statments(self, x: str) -> Set[str]:
        extracted_statements = []
        for statement in unpack_statements(x):
            sid, sname, _exp = statement

            sname = sname.upper()
            # Removed to preserve data related information
            sname = re.sub(r"\s+", " ", sname).strip()

            for did, match_filter in self.statement_match_dict.items():
                if sid in match_filter["c_id"] or sname in match_filter["c_name"]:
                    extracted_statements += did.split("^")

        return set(extracted_statements)

    def set_selected_tasks(self, tasks: List[str]):
        if self.target_table == ECGtoLVH:
            # Dirty test to make sure we have major task as first only
            assert tasks[0] == self.MAJOR_TASK_NAME, tasks
            assert self.MAJOR_TASK_NAME not in tasks[1:], tasks
        self.selected_tasks = tasks

    def get_labels(self, idx):
        # Fast method for sampler to get label data

        # Get Session
        session = self.connect_to_db()

        # Select the targeted mhash
        item = session.query(self.target_table).filter(
            self.target_table.mhash == self.mhash_data["mhash"].iloc[int(idx)]).one()

        # Get target value and map to cid
        category = self.target2cid[self.mhash_data[self.target_attr].iloc[int(idx)]]

        # Unpack Statements for this item
        if self.target_table == ECGtoLVH:
            extracted_statements = self.extract_statments(item.leads.statements)
        else:
            extracted_statements = self.extract_statments(item.statements)
        targeted_statement_tasks = [t for t in self.selected_tasks if t != self.MAJOR_TASK_NAME]

        aux_tasks_labels = []
        for ts in targeted_statement_tasks:
            if "*" in ts:
                flag = False
                for its in ts.split("*"):
                    rts = its[1:] if its[0] == "^" else its
                    if rts == self.MAJOR_TASK_NAME:
                        corr = (category == 1)
                    else:
                        corr = (rts in extracted_statements)
                    if (corr and its[0] != "^") or (not corr and its[0] == "^"):
                        flag = True
                        break
                aux_tasks_labels.append(1 if flag else 0)
            else:
                aux_tasks_labels.append(1 if ts in extracted_statements else 0)

        if self.target_table == ECGtoLVH:
            return [category] + aux_tasks_labels
        else:
            # We do not need the category label for ECGLeads
            return aux_tasks_labels

    def __len__(self):
        return self.data_cnt

    def __getitem__(self, idx):  # pylint: disable=too-many-branches
        # Get Session
        session = self.connect_to_db()

        if idx >= self.BASE_TASK_OFFSET:
            # This means we are using a task based sampler to sample data per task
            real_idx = int(idx) % self.BASE_TASK_OFFSET
            target_task = int(idx) // self.BASE_TASK_OFFSET - 1
        else:
            # This means that we are not using specfic task sampling
            real_idx = idx
            # Use -1 because default_collate can not accept None
            target_task = -1

        # Select the targeted mhash
        target_mhash = self.mhash_data["mhash"].iloc[int(real_idx)]
        item = session.query(self.target_table).filter(
            self.target_table.mhash == target_mhash).one()

        # Unpack Lead data and remove last LEAD_DROP_LEN
        if self.target_table == ECGtoLVH:
            lead_data_dict = unpack_leads(item.leads.lead_data)
        else:
            lead_data_dict = unpack_leads(item.lead_data)

        if self.preprocess_lead is not None:
            lead_data_processed = [self.preprocess_lead(lead_data_dict[n][:-LEAD_DROP_LEN]) for n in LEAD_NAMES]
        else:
            lead_data_processed = [lead_data_dict[n][:-LEAD_DROP_LEN] for n in LEAD_NAMES]
        lead_data = np.array(lead_data_processed, dtype='float32')

        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)

        # Get target value and map to cid
        category = self.target2cid[self.mhash_data[self.target_attr].iloc[int(real_idx)]]

        # Unpack Statements for this item
        if self.target_table == ECGtoLVH:
            extracted_statements = self.extract_statments(item.leads.statements)
        else:
            extracted_statements = self.extract_statments(item.statements)
        targeted_statement_tasks = [t for t in self.selected_tasks if t != self.MAJOR_TASK_NAME]

        aux_tasks_labels = []
        for ts in targeted_statement_tasks:
            if "*" in ts:
                flag = False
                for its in ts.split("*"):
                    rts = its[1:] if its[0] == "^" else its
                    if rts == self.MAJOR_TASK_NAME:
                        corr = (category == 1)
                    else:
                        corr = (rts in extracted_statements)
                    if (corr and its[0] != "^") or (not corr and its[0] == "^"):
                        flag = True
                        break
                aux_tasks_labels.append(1 if flag else 0)
            else:
                aux_tasks_labels.append(1 if ts in extracted_statements else 0)

        if target_task == -1:
            # We return all aux task labels
            if self.target_table == ECGtoLVH:
                task_labels = [category] + aux_tasks_labels
            else:
                # We do not need the category label for ECGLeads
                task_labels = aux_tasks_labels
        else:
            # We return only the selected task's label
            if self.target_table == ECGtoLVH:
                task_labels = ([category] + aux_tasks_labels)[target_task]
            else:
                # We do not need the category label for ECGLeads
                task_labels = aux_tasks_labels[target_task]

        return lead_data, (target_mhash, target_task, task_labels)


class ECGDatasetSubset(Dataset):
    def __init__(self,
                 dataset: ECGDataset,
                 data_type: str,
                 transform: Optional[Tuple[Callable, ...]] = None):
        self.dataset = dataset
        self.data_type = data_type
        assert data_type in ["train", "valid", "test"]
        self.transform = transform
        self.indices = dataset.split_indices_dict[data_type]
        self.mhash_data: pd.DataFrame = self.dataset.mhash_data.iloc[self.indices]

        # Copy Parent Information
        self.variate_cnt: int = self.dataset.variate_cnt
        self.data_len: int = self.dataset.data_len
        self.target_table: Base = self.dataset.target_table
        self.target_attr: str = self.dataset.target_attr
        self.stratify_attr: Sequence[str] = self.dataset.stratify_attr
        self.random_seed: int = self.dataset.random_seed
        self.target2cid: Dict[int, int] = self.dataset.target2cid
        self.cid2target: Dict[int, int] = self.dataset.cid2target
        self.class_cnt: int = self.dataset.class_cnt
        self.cid2name: Dict[int, str] = self.dataset.cid2name
        self.selected_tasks: List[str] = self.dataset.selected_tasks
        self.BASE_TASK_OFFSET: int = self.dataset.BASE_TASK_OFFSET

    def get_labels(self, idx):
        assert idx < self.BASE_TASK_OFFSET
        return self.dataset.get_labels(self.indices[idx])

    def __getitem__(self, idx):
        if idx >= self.BASE_TASK_OFFSET:
            # This means we are using a task based sampler to sample data per task
            orig_dataset_idx = self.indices[int(idx) % self.BASE_TASK_OFFSET]
            target_task = int(idx) // self.BASE_TASK_OFFSET
            orig_dataset_idx = orig_dataset_idx + target_task * self.BASE_TASK_OFFSET
        else:
            # This means that we are not using specfic task sampling
            orig_dataset_idx = self.indices[idx]

        lead_data, (target_mhash, target_task, task_labels) = self.dataset[orig_dataset_idx]
        if self.transform is not None:
            lead_data = Compose(self.transform)(lead_data)
        return lead_data, (target_mhash, target_task, task_labels)

    def __len__(self):
        return len(self.indices)


class RandomECGDatasetSubset(Dataset):
    def __init__(self,
                 dataset: Union[ECGDataset, ECGDatasetSubset],
                 random_subset_ratio: float,
                 random_seed: Optional[int] = None):
        self.dataset = dataset
        self.random_subset_ratio = random_subset_ratio

        # Setup Random State
        self.random_seed = random_seed if random_seed is not None else random.randint(-999999, 999999)
        logger.info("Setting random seed to %s", self.random_seed)
        random.seed(self.random_seed)

        # Get random indices
        self.indices = random.sample(range(len(dataset)), k=round(len(dataset) * random_subset_ratio))
        self.mhash_data: pd.DataFrame = self.dataset.mhash_data.iloc[self.indices]

        # Copy Parent Information
        self.variate_cnt: int = self.dataset.variate_cnt
        self.data_len: int = self.dataset.data_len
        self.target_table: Base = self.dataset.target_table
        self.target_attr: str = self.dataset.target_attr
        self.stratify_attr: Sequence[str] = self.dataset.stratify_attr
        self.origdataset_random_seed: int = self.dataset.random_seed
        self.target2cid: Dict[int, int] = self.dataset.target2cid
        self.cid2target: Dict[int, int] = self.dataset.cid2target
        self.class_cnt: int = self.dataset.class_cnt
        self.cid2name: Dict[int, str] = self.dataset.cid2name
        self.selected_tasks: List[str] = self.dataset.selected_tasks
        self.BASE_TASK_OFFSET: int = self.dataset.BASE_TASK_OFFSET

    def get_labels(self, idx):
        assert idx < self.BASE_TASK_OFFSET
        return self.dataset.get_labels(self.indices[idx])

    def __getitem__(self, idx):
        if idx >= self.BASE_TASK_OFFSET:
            # This means we are using a task based sampler to sample data per task
            orig_dataset_idx = self.indices[int(idx) % self.BASE_TASK_OFFSET]
            target_task = int(idx) // self.BASE_TASK_OFFSET
            orig_dataset_idx = orig_dataset_idx + target_task * self.BASE_TASK_OFFSET
        else:
            # This means that we are not using specfic task sampling
            orig_dataset_idx = self.indices[idx]
        return self.dataset[orig_dataset_idx]

    def __len__(self):
        return len(self.indices)


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,  # pylint: disable=super-init-not-called
                 dataset: Union[ECGDataset, ECGDatasetSubset],
                 weight_function: Callable,
                 num_samples: Optional[int] = None):
        self.dataset = dataset
        # Not sure if other will work
        assert dataset.__class__.__name__ in ["ECGDataset", "ECGDatasetSubset"]
        self.weight_function = weight_function

        # Get distribution of target
        self.data_uni, self.data_cnt, self.weight_mapping = self.get_weight(self.dataset, self.weight_function)
        logger.info("Weight Mapping: %s", self.weight_mapping)

        # Set weight for each sample
        targets = self.dataset.mhash_data[self.dataset.target_attr]
        self.weights = torch.tensor([self.weight_mapping[tar] for tar in targets], dtype=torch.float64)  # pylint: disable=not-callable

        # Save number of samples per iteration
        self.num_samples = num_samples if num_samples is not None else len(self.dataset)
        logger.info("Weighted to total %s samples", self.num_samples)

    @staticmethod
    def get_weight(dataset: Union[ECGDataset, ECGDatasetSubset], weight_function: Callable):
        targets = dataset.mhash_data[dataset.target_attr]
        data_uni, data_cnt = np.unique(targets, return_counts=True)
        weight_mapping = {u: weight_function(c) for u, c in zip(data_uni, data_cnt)}
        return data_uni, data_cnt, weight_mapping

    @staticmethod
    def wf_one(x):  # pylint: disable=unused-argument
        return 1.0

    @staticmethod
    def wf_x(x):
        return x

    @staticmethod
    def wf_onedivx(x):
        return 1.0 / x

    @staticmethod
    def wf_logxdivx(x):
        return math.log(x) / x

    def __iter__(self):
        return (i for i in torch.multinomial(self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class TaskAwareImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):

    def __init__(self,  # pylint: disable=super-init-not-called
                 dataset: Union[ECGDataset, ECGDatasetSubset],
                 weight_function: Callable,
                 batch_size: int,
                 num_batches: Optional[int] = None,
                 task_weights: Optional[List[int]] = None):
        self.dataset = dataset
        self.weight_function = weight_function

        self.task_weights: Optional[List[int]] = task_weights
        if self.task_weights is not None:
            assert len(task_weights) == len(self.dataset.selected_tasks)  # type: ignore
            for task_name, task_weight in zip(self.dataset.selected_tasks, self.task_weights):
                logger.info("Task weight %s: %s", task_name, task_weight)

        # This is absolutely needed to make sure we are training the same tasks in the batch
        self.batch_size = batch_size

        # Get distribution of target
        self.weight_mappings, self.count_mappings, self.weights = self.get_weight(self.dataset, self.weight_function)
        for task_name in self.dataset.selected_tasks:
            logger.info("Task: %s", task_name)
            logger.info("Count Mapping: %s", self.count_mappings[task_name])
            logger.info("Weight Mapping: %s", self.weight_mappings[task_name])

        # Save number of samples per iteration
        self.num_batches = num_batches if num_batches is not None else len(self.dataset) // batch_size
        logger.info("Weighted sampled total %s batches with %s samples per batch", self.num_batches, self.batch_size)

    @staticmethod
    def get_weight(dataset: Union[ECGDataset, ECGDatasetSubset], weight_function: Callable):
        logger.info("Getting Tasks Distribution!")

        all_task_labels = []
        for idx in trange(len(dataset)):
            all_task_labels.append(dataset.get_labels(idx))
        all_task_labels = np.array(all_task_labels).transpose()

        weight_mappings, count_mappings, weights = {}, {}, {}
        for task_id, task_name in enumerate(tqdm(dataset.selected_tasks)):
            targets = all_task_labels[task_id]
            data_uni, data_cnt = np.unique(targets, return_counts=True)
            weight_mappings[task_name] = {u: weight_function(c) for u, c in zip(data_uni, data_cnt)}
            count_mappings[task_name] = dict(zip(data_uni, data_cnt))

            # Set weight for each sample for each task
            weights[task_id] = torch.tensor([weight_mappings[task_name][tar] for tar in targets], dtype=torch.float64)  # pylint: disable=not-callable

        return weight_mappings, count_mappings, weights

    @staticmethod
    def wf_one(x):  # pylint: disable=unused-argument
        return 1.0

    @staticmethod
    def wf_x(x):
        return x

    @staticmethod
    def wf_onedivlogx(x):
        return 1.0 / math.log(x)

    @staticmethod
    def wf_onedivx(x):
        return 1.0 / x

    @staticmethod
    def wf_logxdivx(x):
        return math.log(x) / x

    def __iter__(self):
        # Pick the tasks for all batches
        task_ids = random.choices(list(range(len(self.dataset.selected_tasks))),
                                  weights=self.task_weights, k=self.num_batches)

        ret = []
        for task_id in task_ids:
            # Skip the oringal mapping space for no task mapping
            task_offset = (task_id + 1) * self.dataset.BASE_TASK_OFFSET
            # Select the examples for this task for this batch
            ret += [i + task_offset
                    for i in torch.multinomial(self.weights[task_id], self.batch_size, replacement=True)]
        return iter(ret)

    def __len__(self):
        return self.num_batches * self.batch_size
