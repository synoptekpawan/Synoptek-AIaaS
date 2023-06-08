import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
np.random.seed(10)
random.seed(10)
import sys
sys.path.insert(0, r"./retailChurnAnalytics/")
from evaluateModelOnHoldData import evalModel
from RetailChurnPrediction import RetailChurnPrediction
from RetailChurnDashboard import RetailChurnDashboard
#from RetailChurnTestResponse import RetailChurnTestResponse

holdOuts = r"./retailChurnAnalytics/holdOutData/"
outputs = r"./retailChurnAnalytics/outputs/"
models = r"./retailChurnAnalytics/models/"
logs = r"./retailChurnAnalytics/logs/"

# @st.cache_resource(suppress_st_warning=True)
def main ():
    try:
        st.title("Welcome to Retail Churn Analysis & Prediction service")

        # if st.checkbox("Retail Churn Test Response", key='1'):
        #     RetailChurnTestResponse()
            
            #RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Retail Churn Batch Prediction", key='2'):
            RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Retail Churn Dashboard", key='3'):
            RetailChurnDashboard()

                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


