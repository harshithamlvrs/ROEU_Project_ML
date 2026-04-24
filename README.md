# Smart Sensor and Machine Learning-Based Early Warning Framework for Electric Vehicle Battery Thermal Runaway 

This paper presents a multi-sensor and machine learning framework for early warning of thermal runaway in lithium-ion battery cells. Electrochemical, gas and impedance signals are analyzed using ML models to classify degradation states and estimate thermal-runaway risk. The system outputs a three-tier warning structure that provides actionable lead time to improve battery safety in EVs. Hardware integration is outlined as future work to enable real-time deployment. 

# System Design

Battery_Data_Cleaned.csv (raw Kaggle data)
                 ↓
         [Day1.ipynb]  ← Feature engineering, SoH calculation, data cleaning
                 ↓
        nasa_processed.csv  ← Intermediate processed dataset
                 ↓
         [Day2.ipynb]  ← ML model training
                 ↓
        [4 pickle files created]
           ├─ classifier.pkl      (fault tier classifier, Gradient Boosting)
           ├─ regressor.pkl       (RUL predictor, Gradient Boosting Regressor - 100 estimators)
           ├─ label_encoder.pkl   (encodes long/mid/short)
           └─ scaler.pkl          (StandardScaler for features)
                 ↓
        [pipeline/predict_pipeline.py]  ← Reusable prediction function
                 ↓
        [dashboard/app.py]  ← Streamlit web UI

# Commands to launch app.py on Streamlit (Windows) - For Demo
1. py -m pip install -r requirements.txt
2. py -m streamlit run dashboard/app.py
3. Ctrl + C (stop current app)

    ## With virtual environment:
    1. python -m pip install -r requirements.txt
    2. streamlit run app.py

