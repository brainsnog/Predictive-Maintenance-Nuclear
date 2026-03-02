import numpy as np
import pandas as pd


def create_sequences(df, sensors, seq_length=40):
    sequences = []
    sequence_engine_ids = []
    sequence_end_cycles = []

    engine_ids = df['engine_id'].unique()

    for eng in engine_ids:
        df_eng = df[df['engine_id'] == eng]
        data = df_eng[sensors].values
        cycles = df_eng['cycle'].values

        for start in range(len(data) - seq_length + 1):
            seq = data[start:start + seq_length]
            sequences.append(seq)
            sequence_engine_ids.append(eng)
            sequence_end_cycles.append(cycles[start + seq_length - 1])  # last cycle of the sequence

    return np.array(sequences), np.array(sequence_engine_ids), np.array(sequence_end_cycles)
