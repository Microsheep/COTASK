import enum
import zlib
import hashlib
import logging

from struct import pack, unpack
from typing import Optional, Dict, List, Tuple, Any

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Binary, Text, DateTime, Float, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy_utils.types.choice import ChoiceType

# Initiate Logger
logger = logging.getLogger(__name__)


LEAD_NAMES = ["I", "II", "III", "V1", "V2", "V3", "V4", "V5", "V6", "aVF", "aVL", "aVR"]
LEAD_LENGTH = 5500
LEAD_SAMPLING_RATE = 500
LEAD_DROP_LEN = 500


def tablerepr(self):
    repr_dat = []
    for k in sorted(self.__dict__.keys()):
        if k[0] != '_':
            if isinstance(self.__dict__[k], bytes):
                repr_dat.append(f"{k}=<binary>")
            else:
                repr_dat.append(f"{k}={repr(self.__dict__[k])}")
    return "<{}({})>".format(self.__class__.__name__, ', '.join(repr_dat))


Base = declarative_base()  # type: Any
Base.__repr__ = tablerepr


LeadData = Dict[str, List[float]]
StatementData = List[Tuple[str, ...]]


class TaintCode(enum.Enum):
    # Larger Value Indicate More Serious Problems
    # Removed to preserve data related information
    NORMAL = 0
    EXAMPLE_IGNORE = 200
    DEFAULT_SAFE = 300
    EXAMPLE_ERROR = 400


TaintData = List[Tuple[TaintCode, Dict[str, str]]]
TaintLabel = Tuple[TaintCode, TaintData]


# 0114 ECGtoLVH table definition
class ECGtoLVH(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGtoLVH"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), ForeignKey('ECGLeads.mhash'), unique=True)
    leads = relationship("ECGLeads", backref="lvhdata", uselist=False)
    req_no = Column(String(16))
    patient_id = Column(String(16))
    gender = Column(Integer)
    EKG_age = Column(Integer)
    echo_req_no = Column(String(16))
    # Patient Disease History
    his_HTN = Column(Integer)
    his_DM = Column(Integer)
    his_MI = Column(Integer)
    his_HF = Column(Integer)
    his_stroke = Column(Integer)
    his_CKD = Column(Integer)
    # Is this data testing data
    LVH_Testing = Column(Integer)
    # We choose one LVH label per patient
    LVH_Single = Column(Integer)
    # Raw value for LVmass
    LVH_LVmass_raw = Column(Float)
    LVH_LVmass_level = Column(Integer)
    # Doctor Annotation
    LVH_ECG = Column(Integer)
    # Date for Echo and ECG
    Echo_Date = Column(DateTime)
    ECG_Date = Column(DateTime)


class ECGLeads(Base):  # pylint: disable=too-few-public-methods
    __tablename__ = "ECGLeads"
    id = Column(Integer, primary_key=True)
    mhash = Column(String(16), unique=True)
    req_no = Column(String(16))
    patient_id = Column(String(16))  # No zeros in XML
    xmlhash = Column(String(32), unique=True)
    lead_data = Column(Binary)
    statements = Column(Text)
    report_dt = Column(DateTime)
    taint_code = Column(ChoiceType(TaintCode, impl=Integer()), default=TaintCode.NORMAL)
    taint_history = Column(Text, default="")
    heartbeat_cnt = Column(Integer)


def get_mhash(req_no: str, patient_id: str, salt: Optional[str] = None) -> str:
    s = hashlib.sha256()
    s.update(f"{req_no}{salt}{patient_id}".encode("utf-8"))
    return s.hexdigest()


def pack_leads(lead_data: LeadData) -> bytes:
    assert sorted(lead_data.keys()) == sorted(LEAD_NAMES)
    packed_data: List[float] = []
    for lead_name in LEAD_NAMES:
        assert len(lead_data[lead_name]) == LEAD_LENGTH
        packed_data += lead_data[lead_name]
    return zlib.compress(pack('d' * LEAD_LENGTH * len(LEAD_NAMES), *packed_data))


def unpack_leads(pack_lead_data: bytes) -> LeadData:
    f_lead_data = unpack('d' * LEAD_LENGTH * len(LEAD_NAMES), zlib.decompress(pack_lead_data))
    assert len(f_lead_data) == LEAD_LENGTH * len(LEAD_NAMES)
    ret = {}
    for lead_id, lead_name in enumerate(LEAD_NAMES):
        data_loc = lead_id * LEAD_LENGTH
        ret[lead_name] = list(f_lead_data[data_loc:data_loc + LEAD_LENGTH])
    return ret


def pack_statements(statement_data: StatementData) -> str:
    packed_statements = []
    for statement in statement_data:
        clean_statement = []
        for s in statement:
            assert s.find("^#^") == -1 and s.find("^$^") == -1
            clean_statement.append(s.strip().replace("\n", ""))
        packed_statements.append("^$^".join(clean_statement))
    return "^#^".join(packed_statements)


def unpack_statements(pack_statement_data: str) -> StatementData:
    statements = pack_statement_data.split("^#^")
    return [tuple(statement.split("^$^")) for statement in statements]


def pack_taint_history(taint_history: TaintData) -> str:
    packed_taint_history = []
    for taint in taint_history:
        taint_attr = [taint[0].name]
        for k, v in taint[1].items():
            kv = f"{k}:{v}"
            assert kv.find("#") == -1 and kv.find("$") == -1 and len(kv.split(":")) == 2
            taint_attr.append(kv)
        packed_taint_history.append("$".join(taint_attr))
    return "#".join(packed_taint_history)


def unpack_taint_history(pack_taint_data: str) -> TaintData:
    taint_history = pack_taint_data.split("#")
    ret = []
    for taint in taint_history:
        if taint != "":
            taint_dat = taint.split("$")
            taint_dict = {attr.split(":")[0]: attr.split(":")[1] for attr in taint_dat[1:]}
            ret.append((TaintCode[taint_dat[0]], taint_dict))
    return ret
