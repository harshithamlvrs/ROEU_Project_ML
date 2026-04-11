# pipeline/sensor_simulator.py
import numpy as np, pandas as pd, time

df = pd.read_csv('nasa_processed.csv')

def simulate_sensor_stream(cycle_idx, cell_id, noise=0.02):
    row = df[(df['battery_id']==cell_id)&(df['test_id']==cycle_idx)]
    if row.empty: return None
    row = row.iloc[0]
    return {
        'Re':    round(row['Re'] + np.random.normal(0, noise*0.01), 5),
        'Rct':   round(row['Rct'] + np.random.normal(0, noise*0.01), 5),
        'total_impedance': round(row['total_impedance'] + np.random.normal(0, noise*0.01), 5),
        'capacity_fade':   round(row['capacity_fade'] + np.random.normal(0, noise*0.001), 5),
        'cumulative_fade': round(row['cumulative_fade'], 5),
        'ambient_temperature': int(row['ambient_temperature']),
        'gas_ppm':       round(row['gas_ppm'] + np.random.normal(0, 5), 2),
        'smoke_density': round(row['smoke_density'] + np.random.normal(0, 0.002), 5),
    }

if __name__ == '__main__':
    from predict_pipeline import predict
    cell = df['battery_id'].iloc[0]
    for cycle in sorted(df[df['battery_id']==cell]['test_id'].unique()):
        reading = simulate_sensor_stream(cycle, cell)
        if reading:
            result = predict(reading)
            print(f"Cycle {cycle:3} | {result} | Gas: {reading['gas_ppm']:.1f}ppm")
        time.sleep(0.2)