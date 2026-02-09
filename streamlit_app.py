import streamlit as st
from utils.init import init_session_vars
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall_array

init_session_vars()

pg = st.navigation([st.Page("pages/predict.py"), st.Page("pages/threshold_slider.py"), st.Page("pages/about_us.py")])
pg.run()