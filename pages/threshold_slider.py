from utils.preprocess import load_preprocess
from utils.lgbm import load_model, predict
from utils.precision_recall import precision_recall
import streamlit as st
import matplotlib.pyplot as plt

data_path = 'data/'

if 'X' not in st.session_state or 'y' not in st.session_state:
   st.session_state['X'], st.session_state['y'] = load_preprocess(data_path)

X, y = st.session_state['X'], st.session_state['y']

if 'model' not in st.session_state:
    st.session_state['model'] = load_model('model/lgbm.pkl')

model = st.session_state['model']

if 'y_probs' not in st.session_state:
   st.session_state['y_probs'] = predict(model, X)

y_probs = st.session_state['y_probs']

if 'threshold' not in st.session_state:
   threshold = 0.5
else:
   threshold = st.session_state['threshold']

precision, recall = precision_recall(thresh = threshold, y_probs = y_probs, y_test = y)

st.title("Threshold Slider")

fig, ax = plt.subplots(1,2)

colours = ['lightblue', 'blue']

ax[0].pie(([1 - precision, precision]), startangle=90, wedgeprops=dict(width=.5), colors = colours)
ax[0].set_title('Precision')
ax[0].text(x = 0, y = 0, s = f'{precision*100: .1f}%', ha = 'center')

ax[1].pie(([1 - recall, recall]), startangle=90, wedgeprops=dict(width=.5), colors = colours)
ax[1].set_title('Recall')
ax[1].text(x = 0, y = 0, s = f'{recall*100: .1f}%', ha = 'center')

st.pyplot(fig)

st.slider(label = "Choose threshold:", min_value = 0.0, max_value = 1.0, step = 0.01, value = 0.5, key = 'threshold')

# st.write(f'Precision: {precision*100: .2f}%')
# st.write(f'Recall: {recall*100: .2f}%')