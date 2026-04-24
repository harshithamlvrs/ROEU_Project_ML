# This script simulates a real-time sensor data stream for a single battery cell, 
# feeding it into the prediction pipeline and printing the results. 
# It includes a warmup period to allow the RUL predictions to stabilize before triggering any alerts. 
# The sensor readings are generated with some noise to mimic real-world conditions.

from collections import deque
import time

import numpy as np
import pandas as pd

WARMUP_CYCLES = 5
RUL_WINDOW_SIZE = 5
SENSOR_NOISE = 0.02

df = pd.read_csv('nasa_processed.csv')

def simulate_sensor_stream(cycle_idx, cell_id, noise=SENSOR_NOISE):
    row = df[(df['battery_id'] == cell_id) & (df['test_id'] == cycle_idx)]
    if row.empty:
        return None

    row = row.iloc[0]
    return {
        'Re': round(row['Re'] + np.random.normal(0, noise * 0.01), 5),
        'Rct': round(row['Rct'] + np.random.normal(0, noise * 0.01), 5),
        'total_impedance': round(row['total_impedance'] + np.random.normal(0, noise * 0.01), 5),
        'capacity_fade': round(row['capacity_fade'] + np.random.normal(0, noise * 0.001), 5),
        'cumulative_fade': round(row['cumulative_fade'], 5),
        'ambient_temperature': int(row['ambient_temperature']),
        'gas_ppm': round(row['gas_ppm'] + np.random.normal(0, 5), 2),
        'smoke_density': round(row['smoke_density'] + np.random.normal(0, 0.002), 5),
    }


def main():
    from predict_pipeline import predict

    cell = df['battery_id'].iloc[0]
    cycles = sorted(df[df['battery_id'] == cell]['test_id'].unique())
    rul_window = deque(maxlen=RUL_WINDOW_SIZE)

    for idx, cycle in enumerate(cycles):
        reading = simulate_sensor_stream(cycle, cell)
        if not reading:
            continue

        result = predict(reading)
        rul_window.append(result['cycles_remaining'])
        smoothed_rul = int(np.mean(rul_window))
        tier = result['fault_tier']

        in_warmup = idx < WARMUP_CYCLES
        if tier == 'short' and not in_warmup:
            alert = ' *** ALERT ***'
        elif in_warmup:
            alert = ' (warmup)'
        else:
            alert = ''

        print(
            f"Cycle {cycle:4} | Tier: {tier:6} | "
            f"RUL: {smoothed_rul:4} cycles (smoothed) | "
            f"Gas: {reading['gas_ppm']:.1f}ppm{alert}"
        )
        time.sleep(0.2)


if __name__ == '__main__':
    main()