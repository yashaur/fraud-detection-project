import streamlit as st
from random import sample
from utils.init import init_session_vars
from utils.data import preprocess_input
from utils.model import load_model, predict

init_session_vars()

field_names = {
    'type': 'Transaction Type', 'amount': 'Amount', 'hour_of_day': 'Hour of Day',
    'oldbalanceOrg': 'Origin Account (Old Balance)', 'newbalanceOrig': 'Origin Account (New Balance)',
    'oldbalanceDest': 'Destination Account (Old Balance)', 'newbalanceDest': 'Destination Account (New Balance)'
}

types = ['Cash Out', 'Cash In', 'Transfer','Payment', 'Debit']

samples = st.session_state['pred_samples']

# for k in field_names:
#     st.session_state[k] = st.session_state.get(k, None)

### STREAMLIT PAGE CODE ###

st.title("Predict Fraud 🚨")


# Demo and reset buttons
with st.container(horizontal=True):

    if st.button("Cycle Random Demo Values"):
        random_sample = sample(samples, k = 1)
        # st.write(random_sample)
        for k in field_names:
            if k == 'hour_of_day':
                st.session_state[k] = random_sample[0][k] + 1
            else:
               st.session_state[k] = random_sample[0][k] 

    if st.button("Reset"):
        for k in field_names:
            st.session_state[k] = None


st.space('xxsmall')
st.write('**Transaction Details:**')

type, amount, hour_of_day = st.columns(3)

type.selectbox(
    field_names['type'],
    options = types,
    key = 'type'
)

amount.number_input(
    field_names['amount'],
    placeholder = 'Enter Amount',
    min_value = 0.0,
    step = 1000.0,
    value = None,
    key = 'amount',
)
# st.session_state['amount'] = st.session_state['amount_input']

hour_of_day.number_input(
    field_names['hour_of_day'],
    placeholder = 'Enter Hour',
    min_value = 1,
    max_value = 24,
    step = 1,
    value = None,
    key = 'hour_of_day'
)

st.space('xxsmall')
st.write('**Origin Account Details:**')

oldbalanceOrg, newbalanceOrig = st.columns(2)

oldbalanceOrg.number_input(
    field_names['oldbalanceOrg'],
    placeholder = 'Enter Amount',
    min_value = 0.0,
    step = 1000.0,
    value = None,
    key = 'oldbalanceOrg'
)

newbalanceOrig.number_input(
    field_names['newbalanceOrig'],
    placeholder = 'Enter Amount',
    min_value = 0.0,
    step = 1000.0,
    value = None,
    key = 'newbalanceOrig'
)

st.space('xxsmall')
st.write('**Destination Account Details:**')

oldbalanceDest, newbalanceDest = st.columns(2)

oldbalanceDest.number_input(
    field_names['oldbalanceDest'],
    placeholder = 'Enter Amount',
    min_value = 0.0,
    step = 1000.0,
    value = None,
    key = 'oldbalanceDest'
)

newbalanceDest.number_input(
    field_names['newbalanceDest'],
    placeholder = 'Enter Amount',
    min_value = 0.0,
    step = 1000.0,
    value = None,
    key = 'newbalanceDest'
)


threshold = st.session_state['threshold']
st.warning(f"The threshold is currently set at {threshold}%")

st.space('xsmall')


if st.button("Predict", width = 'stretch'):
    
    input = {col: st.session_state[col] for col in field_names}

    if None in input.values():
        st.error("❗️ Please fill out the following fields:")
        for col in field_names:
            if input[col] is None:
                st.write(field_names[col])

    else:
        print('~' * 16, 'PREDICT', '~' * 15)
        X = preprocess_input(input)

        model = load_model()

        probability = predict(model, X)[0] * 100
        
        print('~'*40)

        prediction = 1 if probability >= threshold else 0

        if prediction == 1:
            st.error(f"‼️ This transaction is fraudulent. Probability: {probability:.2f}%")
        else:
            st.success(f"✅ This transaction is not fraudulent. Probability: {probability:.2f}%")
