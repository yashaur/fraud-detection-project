import streamlit as st
from utils.data import load_preprocess, load_prediction_samples
from utils.model import load_model, predict
from utils.precision_recall import precision_recall_array

def init_session_vars():
    
    if st.session_state.get('_app_initialised', False):
        return
    
    init_state ={
                'Data': False,
                'Model': False,
                'Predictions': False,
                'Recall-Precision Data': False,
                'Threshold': False,
                'Prediction Samples': False,
                # 'Field Names': False,
                # 'Input Fields': False
    }

    if 'X' not in st.session_state or 'y' not in st.session_state:
        init_state['Data'] = True
        st.session_state['X'], st.session_state['y'] = load_preprocess()
        X, y = st.session_state['X'], st.session_state['y']

    if 'model' not in st.session_state:
        init_state['Model'] = True
        st.session_state['model'] = load_model()
        model = st.session_state['model']

    if 'y_probs' not in st.session_state:
        init_state['Predictions'] = True
        st.session_state['y_probs'] = predict(model, X)
        y_probs = st.session_state['y_probs']

    if 'pr_data' not in st.session_state:
        init_state['Recall-Precision Data'] = True
        st.session_state['pr_data'] = precision_recall_array(X, y, model, y_probs)

    if 'threshold' not in st.session_state:
        init_state['Threshold'] = True
        st.session_state['threshold'] = 50
        threshold = st.session_state['threshold']

    if 'pred_samples' not in st.session_state:
        init_state['Prediction Samples'] = True
        st.session_state['pred_samples'] = load_prediction_samples()

    # if 'field_names' not in st.session_state:
    #     init_state['Field Names'] = True
    #     st.session_state['field_names'] = {
    #                     'type': 'Transaction Type', 'amount': 'Amount', 'hour_of_day': 'Hour of Day',
    #                     'oldbalanceOrg': 'Origin Account (Old Balance)', 'newbalanceOrig': 'Origin Account (New Balance)',
    #                     'oldbalanceDest': 'Destination Account (Old Balance)', 'newbalanceDest': 'Destination Account (New Balance)'
    #                     }
    
    # field_names = st.session_state['field_names']

    # for k in field_names:
    #     if k not in st.session_state:
    #         init_state['Input Fields'] = True
    #         st.session_state[k] = st.session_state.get(k, None)


    if any(init_state.values()):
        print('\n')
        print('~'*12, 'INITIALISATION', '~'*12)
        
        for stage in init_state:
            if init_state[stage] and stage != 'Threshold':
                print('Initialising', stage)
            elif init_state[stage] and stage == 'Threshold':
                print(f'Initialising Threshold to {threshold}%')
        print('~'*40)
