import streamlit as st
from utils.init import init_session_vars
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall_array

init_session_vars()

field_names = st.session_state['field_names']

for k in field_names:
    st.session_state[k] = st.session_state.get(k, None)

pg = st.navigation([
                st.Page("pages/dashboard.py"),
                st.Page("pages/predict.py"),
                st.Page("pages/threshold_slider.py")
                ])
pg.run()