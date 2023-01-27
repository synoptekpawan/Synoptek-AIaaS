import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
np.random.seed(10)
random.seed(10)
import sys
<<<<<<< HEAD
sys.path.insert(0, r"./retailChurnAnalytics/")
=======
sys.path.insert(0, r".../../")
>>>>>>> 88dfedf678801cd959e6e50a20449b88128ef647
from evaluateModelOnHoldData import evalModel
from RetailChurnPrediction import RetailChurnPrediction
from RetailChurnDashboard import RetailChurnDashboard

<<<<<<< HEAD
holdOuts = r"./retailChurnAnalytics/holdOutData/"
outputs = r"./retailChurnAnalytics/outputs/"
models = r"./retailChurnAnalytics/models/"
logs = r"./retailChurnAnalytics/logs/"
=======
holdOuts = r".../../inputs"
outputs = r".../../outputs/"
models = r".../../models/"
logs = r".../../logs/"
>>>>>>> 88dfedf678801cd959e6e50a20449b88128ef647


def main ():
    try:
        st.title("Welcome to Retail Churn Analysis & Prediction service")

        if st.checkbox("Retail Churn Prediction", key='1'):
            RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Retail Churn Dashboard", key='2'):
            RetailChurnDashboard()

                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


