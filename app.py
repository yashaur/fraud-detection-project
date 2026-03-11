import streamlit as st

from utils.init import init_session_vars
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall_array
import time

init_session_vars()

field_names = st.session_state['field_names']

for k in field_names:
    st.session_state[k] = st.session_state.get(k, None)

pages = {"↔️ GO TO PAGE" :
               [
                st.Page(page = "pages/dashboard.py", title = '📊 Dashboard'),
                st.Page(page = "pages/predict.py", title = '🚨 Predict'),
                st.Page(page = "pages/threshold_slider.py", title = '🎛️ Threshold Slider')
                ]}
                
pg = st.navigation(pages, position = 'top')

pg.run()
