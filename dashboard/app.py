# First: pip install streamlit  (add to requirements.txt)
# Run: streamlit run dashboard/app.py

# dashboard/app.py
import streamlit as st, pandas as pd, joblib
import matplotlib.pyplot as plt

clf=joblib.load('pipeline/classifier.pkl')
reg=joblib.load('pipeline/regressor.pkl')
le =joblib.load('pipeline/label_encoder.pkl')
scaler=joblib.load('pipeline/scaler.pkl')
df=pd.read_csv('nasa_processed.csv')

FEATURES=['Re','Rct','total_impedance','capacity_fade',
          'cumulative_fade','ambient_temperature',
          'gas_ppm','smoke_density']

st.title('EV Battery Early Warning System')
st.markdown('Adjust sensor readings to simulate real-time fault detection.')

col1,col2=st.columns(2)
with col1:
    Re   =st.slider('Re — electrolyte resistance (Ω)',0.04,0.20,0.07)
    Rct  =st.slider('Rct — charge transfer resistance (Ω)',0.05,0.40,0.15)
    c_f  =st.slider('Capacity fade (Ah/cycle)',-0.05,0.0,-0.005)
    c_cum=st.slider('Cumulative fade (Ah)',0.0,0.5,0.1)
with col2:
    temp =st.selectbox('Ambient temperature (°C)',[4,24,43])
    gas  =st.slider('Gas level (ppm)',150,800,220)
    smoke=st.slider('Smoke density',0.0,0.3,0.05)

row=pd.DataFrame([[Re,Rct,Re+Rct,c_f,c_cum,temp,gas,smoke]],columns=FEATURES)
row_s=pd.DataFrame(scaler.transform(row),columns=FEATURES)
tier=le.inverse_transform(clf.predict(row_s))[0]
rul =max(0,round(reg.predict(row_s)[0]))

color={'long':'green','mid':'orange','short':'red'}[tier]
st.markdown(f'### Fault tier: :{color}[{tier.upper()}]')
if tier=='short':
    st.error('CRITICAL — Battery failure imminent. Take precautions immediately.')
st.metric('Estimated cycles remaining',rul)

st.subheader('Battery SoH trend')
cell=st.selectbox('Select battery cell',df['battery_id'].unique())
cell_df=df[df['battery_id']==cell].sort_values('test_id')
fig,ax=plt.subplots()
ax.plot(cell_df['test_id'],cell_df['SoH'])
ax.axhline(90,color='orange',linestyle='--',label='90% threshold')
ax.axhline(80,color='red',linestyle='--',label='80% threshold')
ax.set_xlabel('Cycle'); ax.set_ylabel('SoH (%)')
ax.legend(); st.pyplot(fig)