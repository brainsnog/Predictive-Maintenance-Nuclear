from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_fd001_data(relative_path):
    data_path = PROJECT_ROOT / relative_path
    df = pd.read_csv(
        data_path,
        sep=r"\s+",
        header=None
    )
    df.columns = [
        "engine_id", "cycle",
        "op_setting_1", "op_setting_2", "op_setting_3",
        "sensor_1", "sensor_2", "sensor_3", "sensor_4",
        "sensor_5", "sensor_6", "sensor_7", "sensor_8",
        "sensor_9", "sensor_10", "sensor_11", "sensor_12",
        "sensor_13", "sensor_14", "sensor_15", "sensor_16",
        "sensor_17", "sensor_18", "sensor_19", "sensor_20",
        "sensor_21"
    ]
    return df

def add_normal_operation_flag(df):
    max_cycles = df.groupby("engine_id")["cycle"].max()
    normal_op_limit = (max_cycles * 0.2).clip(upper=50).astype(int)
    df = df.copy()
    df["is_normal_operation"] = df["cycle"] <= df["engine_id"].map(normal_op_limit)
    return df

# new load_processed_data():
def load_processed_data():
    df = load_fd001_data("data/raw/train_FD001.txt")
    df = add_normal_operation_flag(df)
    return df
