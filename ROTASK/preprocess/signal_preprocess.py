from typing import List

from biosppy.signals import ecg

from ROTASK.datasets.ecg_data_model import LEAD_SAMPLING_RATE


def preprocess_leads(lead: List[float]) -> List[float]:
    # Preprocess lead data to remove baseline wander and high freq noise
    corrected_signal, _, _ = ecg.st.filter_signal(lead, 'butter', 'highpass', 2, 1, LEAD_SAMPLING_RATE)
    corrected_signal, _, _ = ecg.st.filter_signal(corrected_signal, 'butter', 'lowpass', 12, 35, LEAD_SAMPLING_RATE)
    return corrected_signal
