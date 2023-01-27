import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
np.random.seed(10)
random.seed(10)
import sys
sys.path.insert(0, r".../../")
from evaluateModelOnHoldData import evalModel
from RetailChurnPrediction import RetailChurnPrediction
from RetailChurnDashboard import RetailChurnDashboard

holdOuts = r".../../inputs"
outputs = r".../../outputs/"
models = r".../../models/"
logs = r".../../logs/"


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


