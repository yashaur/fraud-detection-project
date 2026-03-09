import streamlit as st
from utils.shap import shap_values

def shap_init():

    if st.session_state.get('_shap_initiated', False):
        return

    shap_values_init = False

    explainer = st.session_state['explainer']
    X = st.session_state['X']

    if 'shap_values' not in st.session_state:
        shap_values_init = True
        st.session_state['shap_values'] = shap_values(explainer, X, top_bottom = False, make_df = True)

    if shap_values_init:
        print('\n')
        print('~'*12, 'INITIALISATION', '~'*12)
        print('Initialising SHAP values')
        print('~'*40)