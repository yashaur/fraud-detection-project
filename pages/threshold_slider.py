# IMPORTING LIBRARIES
from utils.data import load_preprocess
from utils.model import load_model, predict
from utils.precision_recall import precision_recall, precision_recall_array
from utils.charts import create_pr_doughnuts, create_pr_chart
import streamlit as st
import matplotlib.pyplot as plt


### INITIALISING SESSION STATE VARIABLES ###

# Initialising X and y
if 'X' not in st.session_state or 'y' not in st.session_state:
    st.session_state['X'], st.session_state['y'] = load_preprocess()
X, y = st.session_state['X'], st.session_state['y']

# Initiliasing the model
if 'model' not in st.session_state:
    st.session_state['model'] = load_model()
model = st.session_state['model']

# Initialising predicted probabilities
if 'y_probs' not in st.session_state:
    st.session_state['y_probs'] = predict(model, X)
y_probs = st.session_state['y_probs']

# Initialising the threshold
if 'threshold' not in st.session_state:
    threshold = 0.5
else:
    threshold = st.session_state['threshold'] / 100

# Initialising the precision-recall array at different thresholds
if 'pr_data' not in st.session_state:
    st.session_state['pr_data'] = precision_recall_array(X, y, model, y_probs)
pr_data = st.session_state['pr_data']


### STREAMLIT PAGE CODE ###

# Title of the page
st.title("Threshold Slider")

# Creating the precision and recall doughnut charts
precision, recall = precision_recall(thresh = threshold, y_probs = y_probs, y_test = y)
doughnuts = create_pr_doughnuts(precision, recall)
st.pyplot(doughnuts)

# Creating the precision-recall tradeoff plot
pr_chart = create_pr_chart(pr_data, precision, recall)
st.pyplot(pr_chart)

# Slider to control threshold
st.slider(label = "Choose threshold (%):", min_value = 0, max_value = 100, step = 1, value = 50, key = 'threshold')