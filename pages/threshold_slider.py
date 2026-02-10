# IMPORTING LIBRARIES
from utils.init import init_session_vars
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall, precision_recall_array
from utils.charts import create_pr_doughnuts, create_pr_chart
import streamlit as st
import matplotlib.pyplot as plt

init_session_vars()

### STREAMLIT PAGE CODE ###

# Title of the page
st.title("Threshold Slider")

# Initialising the threshold

st.button(
            label = "Reset to 50%",
            on_click = lambda: st.session_state.update({'threshold': 50, 'threshold_slider': 50})
        )


threshold_int = st.slider(
                        label = "Choose threshold (%):",
                        min_value = 0,
                        max_value = 100,
                        step = 1,
                        key = 'threshold_slider',
                        value = st.session_state['threshold'],
                        on_change = lambda: st.session_state.update({'threshold': st.session_state['threshold_slider']})
)

threshold = threshold_int / 100

# ### INITIALISING SESSION STATE VARIABLES ###

X, y = st.session_state['X'], st.session_state['y']
model = st.session_state['model']
y_probs = st.session_state['y_probs']
pr_data = st.session_state['pr_data']

# Creating the precision and recall doughnut charts
precision, recall = precision_recall(thresh = threshold, y_probs = y_probs, y_test = y)
doughnuts = create_pr_doughnuts(precision, recall)
st.pyplot(doughnuts)

# Creating the precision-recall tradeoff plot
pr_chart = create_pr_chart(pr_data, precision, recall)
st.pyplot(pr_chart)

# Slider to control threshold
