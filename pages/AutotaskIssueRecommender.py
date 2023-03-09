import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
#np.random.seed(10)
#random.seed(10)
import sys
sys.path.insert(0, r"./AutotaskIssueRecommender-main/") # retailDynamicPricing\RetailDynamicPricing.py
# sys.path.insert(0, r"./utils/")
#from evaluateModelOnHoldData import evalModel
from autotask_app import autoTaskRec



def main ():
    try:
        st.title("Welcome to Autotask Issue Recommender service")

        # if st.checkbox("Retail Churn Test Response", key='1'):
        #     RetailChurnTestResponse()
            
            #RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Autotask Issue Recommender", key='2'):
            autoTaskRec()
        # if st.checkbox("Retail Churn Dashboard", key='3'):
        #     RetailChurnDashboard()

                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


