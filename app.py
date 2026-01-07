

import streamlit as st
import pickle
import numpy as np


def load_model():
    try:
        with open("customer behaviour.pkl",'rb') as file:
            model = pickle.load(file)

        scaler = None
        try:
            with open('scaler.pkl','rb') as file:
                scaler = pickle.load(file)
        except:
            st.warning('Scaler not found or invalid')

        return model,scaler

    except FileNotFoundError as e:
        st.error(f'Model file not found: {e}')
        return None, None

    except Exception as e:
         st.error(f'Error loading artifacts: {e}')
         return None,None

model,scaler = load_model()


def genderInput(gender_input):
    gender_input = gender_input.lower().strip()
    if gender_input == 'male':
        return 0
    else:
        return 1

def resultoutput(result):
    if result == 1:
        return 'yes'
    else:
        return 'no'

def customer_satisfaction_prediction(gender_input,age_input,salary_input):
    try:
        gender_value = genderInput(gender_input)

        if gender_value is None:
            return 'Error : gender must be male or female ',None
        age_value = float(age_input)
        salary_value =float(salary_input)

        input_data = np.array([[gender_value,age_value,salary_value]])

        if scaler is None or not hasattr(scaler,'transform'):
            return 'Error : scaler not available or invalid.',None
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probablities = model.predict_prob(scaled_data)
        predicted_purchase = int(prediction[0])
        confidence = probablities[0][predicted_purchase]

        return predicted_purchase,confidence
    except Exception as e:
        return f'Prediction Error: {e}',None

def main():
    st.title('Customer behaviour prediction web app')

    gender_input = st.selectbox('select gender',['Male','Female'])
    age_input = st.number_input('Enter Age',max_value = 100,min_value = 18,value = 30)
    salary_input = st.number_input('Enter estimated salary ',min_value = 0,value = 50000,step = 1000)

    if st.button('predict customer purchase'):
        if model is None:
            st.error('model not loaded properly.') 
            return 

    result,confidence = customer_satisfaction_prediction(gender_input,age_input,salary_input)


    if isinstance(result,str) and (result.startswith('Error') or result.startswith('prediction')):
        st.error(result)
    else:
        result_output = resultoutput(result)
        st.success(f'Will the customer purchase?:{result_output}')
        if confidence is not None:
            st.info(f'confidence:{confidence:.2%}')

if __name__ == '__main__':
    main()
