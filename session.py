import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from redcap import Project
import io

st.set_page_config(layout="wide", page_title="Hypoxia Lab Session Explorer")

# mode= st.radio('Offline mode?', ['Online', 'Offline'])
mode='Online'
if mode == 'Online':
    if 'df' not in st.session_state:
        api_url = 'https://redcap.ucsf.edu/api/'
        api_k = st.secrets['api_k']
        proj = Project(api_url, api_k)
        
        fdf = io.BytesIO(proj.export_file(record='4', field='file')[0])
        df = pd.read_csv(fdf)
        st.session_state['df'] = df

        fpt = io.BytesIO(proj.export_file(record='5', field='file')[0])
        patient = pd.read_csv(fpt)
        st.session_state['patient'] = patient

        fenc = io.BytesIO(proj.export_file(record='6', field='file')[0])
        encounter = pd.read_csv(fenc)
        st.session_state['encounter'] = encounter

        fkonica = io.BytesIO(proj.export_file(record='7', field='file')[0])
        konica = pd.read_csv(fkonica)
        st.session_state['konica'] = konica

    else:
        # st.caption('Using cached data')
        df = st.session_state['df']
        patient = st.session_state['patient']
        encounter = st.session_state['encounter']
        konica = st.session_state['konica']
else:
    df = pd.read_csv('spo2_sao2.csv')
    patient = pd.read_csv('patient.csv')
    encounter = pd.read_csv('devs.csv')
    konica = pd.read_csv('konica.csv')

colormap = {'Device1':'IndianRed',
            'Device1 bias':'IndianRed',
            'Device2 SpO2':'palegreen',
            'Device2 bias':'palegreen',
            'so2':'powderblue',
            'so2_range':'powderblue'}

def arms(spo2,sao2):
    return np.sqrt(np.mean((spo2-sao2)**2))

def create_scatter(frame, bias_thresh):
    fig = (px.scatter(
        frame, x='Sample', y=['Device1 SpO2', 'Device1 bias','Device2 SpO2','Device2 bias','so2'], color_discrete_map=colormap,)
        .update_traces(marker=dict(size=12,opacity=0.8, 
                        line=dict(width=2,color='DarkSlateGrey')))
        #horizontal legend
        .update_layout(legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1))
        .add_hline(y=bias_thresh, line_width=1, line_dash="dash", line_color="#FFBF00")
        # show every xtick
        .update_xaxes(dtick=1)
    )
    return fig

def ita_scatter(df):
    fig = (px.scatter(
        df[df['group']==group], x='patient_id', y='ita', title=f'ITA measurements at {group}, by subject')
        .update_xaxes(showticklabels=False, )
           )
    return fig

sessions = df.session.dropna().unique().tolist()
################################################################################ Layout begins

st.header('Controlled Desaturation Study Reports')

st.subheader('Overall Group Characteristics')
one, two = st.columns(2)
with one:
    st.write('Overall device 1 ARMS: ', arms(df['Device1 SpO2'],df['so2']).round(2))
    st.write('Overall device 2 ARMS: ', arms(df['Device2 SpO2'],df['so2']).round(2))
    st.write('Overall age:', encounter['age_at_encounter'].mean().round(2))
    st.write('Overall BMI:', patient['bmi'].mean().round(2))
    st.write('Sex:', patient['assigned_sex'].value_counts())
             


with two:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(['ITA', 'So2 Distribution', 'Age', 'Monk','BMI'])
    with tab1:
        group = st.selectbox('ITA Anatomic location', konica.group.unique().tolist())
        st.plotly_chart(ita_scatter(konica.groupby(['patient_id','group']).median(numeric_only=True).reset_index().sort_values(by='ita')))
    with tab2:
        st.plotly_chart(px.histogram(df, x='so2', title='so2 distribution', text_auto=True))
    with tab3:
        st.plotly_chart(px.histogram(encounter, x='age_at_encounter', title='Age distribution', text_auto=True))
    with tab4:
        monk_selected = st.selectbox('Measurement location', ['monk_forehead', 'monk_fingernail', 'monk_dorsal'])
        st.plotly_chart(px.histogram(encounter, x=encounter[f'{monk_selected}'], title='Monk Forehead', text_auto=True))
    with tab5:
        st.plotly_chart(px.histogram(patient, x='bmi', title='BMI distribution', text_auto=True))
# st.write(konica)

st.subheader('Individual session characteristics')
one, two = st.columns(2)

session_selected = two.selectbox('Select a session', sessions)
frame = df[df['session']==session_selected][['session', 'Sample', 'so2_range', 'so2',  'Device1 SpO2', 'Device1 bias','Device2 bias','Device2 SpO2']]
frame_enc = encounter[encounter['session']==session_selected]
frame_pt = patient[patient['subject_id'].isin(frame_enc['subject_id'])]

with one:
    st.plotly_chart(create_scatter(frame, 3.5))
with two:
    st.write('Device 1 Arms: ', round(arms(frame['Device1 SpO2'],frame['so2']),2))
    st.write('Device 2 Arms: ', round(arms(frame['Device2 SpO2'],frame['so2']),2))
    st.write('Patient age:', frame_enc['age_at_encounter'].values[0])
    st.write('Sex: ', frame_pt['assigned_sex'].values[0])
    st.write('BMI: ', frame_pt['bmi'].values[0])
    st.write('Monk (forehead):', frame_enc['monk_forehead'].values[0])
    st.write('Monk (fingernail):', frame_enc['monk_fingernail'].values[0])
    st.write('Monk (dorsal):', frame_enc['monk_dorsal'].values[0])

# st.write(frame_enc)
# st.write(frame_pt)
