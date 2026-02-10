import streamlit as st
from utils.init import init_session_vars
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall_array

init_session_vars()

field_names = {
    'type': 'Transaction Type', 'amount': 'Amount', 'hour_of_day': 'Hour of Day',
    'oldbalanceOrg': 'Origin Account (Old Balance)', 'newbalanceOrig': 'Origin Account (New Balance)',
    'oldbalanceDest': 'Destination Account (Old Balance)', 'newbalanceDest': 'Destination Account (New Balance)'
}
for k in field_names:
    st.session_state[k] = st.session_state.get(k, None)


pg = st.navigation([st.Page("pages/dashboard.py") ,st.Page("pages/predict.py"), st.Page("pages/threshold_slider.py"), st.Page("pages/about_us.py")])
pg.run()