
import numpy as np
import pandas as pd


def compute_per_engine_thresholds(reconstruction_error, engine_ids, percentile=97.5):
    """
    Compute per-engine anomaly thresholds using a percentile of reconstruction error.
    """
    engine_thresholds = {}

    for eng in np.unique(engine_ids):
        eng_errors = reconstruction_error[engine_ids == eng]
        engine_thresholds[eng] = np.percentile(eng_errors, percentile)

    return engine_thresholds


def apply_consecutive_logic(is_anomaly, engine_ids, consecutive_required=3):
    """
    Apply consecutive anomaly logic per engine.
    """
    consecutive_flags = np.zeros_like(is_anomaly, dtype=bool)

    for eng in np.unique(engine_ids):
        eng_mask = engine_ids == eng
        eng_anomalies = is_anomaly[eng_mask]

        count = 0
        indices = np.where(eng_mask)[0]

        for i, a in enumerate(eng_anomalies):
            if a:
                count += 1
                if count >= consecutive_required:
                    consecutive_flags[indices[i]] = True
            else:
                count = 0

    return consecutive_flags



def compute_lead_times(engine_ids, end_cycles, consecutive_flags):
    """
    Compute lead time per engine.
    """
    lead_times = {}

    for eng in np.unique(engine_ids):
        mask = engine_ids == eng

        eng_cycles = end_cycles[mask]
        eng_flags = consecutive_flags[mask]

        anomaly_cycles = eng_cycles[eng_flags]

        if len(anomaly_cycles) == 0:
            lead_times[eng] = 0
        else:
            lead_times[eng] = eng_cycles.max() - anomaly_cycles.min()

    return lead_times
