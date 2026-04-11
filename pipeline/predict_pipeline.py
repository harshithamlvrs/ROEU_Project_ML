# pipeline/predict_pipeline.py
#load the scalar.pkl and use all 8 features. Old version uses only 6 features without scaling.

import pandas as pd, joblib

FEATURES = ['Re','Rct','total_impedance','capacity_fade',
            'cumulative_fade','ambient_temperature',
            'gas_ppm','smoke_density']

clf    = joblib.load('pipeline/classifier.pkl')
reg    = joblib.load('pipeline/regressor.pkl')
le     = joblib.load('pipeline/label_encoder.pkl')
scaler = joblib.load('pipeline/scaler.pkl')

def predict(sensor_reading: dict) -> dict:
    row   = pd.DataFrame([sensor_reading])[FEATURES]
    row_s = pd.DataFrame(scaler.transform(row), columns=FEATURES)
    tier  = le.inverse_transform(clf.predict(row_s))[0]
    rul   = max(0, round(reg.predict(row_s)[0]))
    return {'fault_tier': tier, 'cycles_remaining': rul}

# Test with a sample reading — run with Ctrl+F5
sample = {
    'Re':0.08, 'Rct':0.25, 'total_impedance':0.33,
    'capacity_fade':-0.01, 'cumulative_fade':0.15,
    'ambient_temperature':24, 'gas_ppm':245.0, 'smoke_density':0.06
}
print(predict(sample))
# Expected output: {'fault_tier': 'long', 'cycles_remaining': ...}