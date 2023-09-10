import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
np.random.seed(10)
random.seed(10)
# import sys
# sys.path.insert(0, r"./retailDynamicPricing/") # retailDynamicPricing\RetailDynamicPricing.py
# # sys.path.insert(0, r"./utils/")
# #from evaluateModelOnHoldData import evalModel
from retailDynamicPricing.RetailDynamicPricing import RetailDynamicPricing
#from RetailChurnDashboard import RetailChurnDashboard
#from RetailChurnTestResponse import RetailChurnTestResponse

holdOuts = r"./retailDynamicPricing/holdOutData/"
outputs = r"./retailDynamicPricing/outputs/"
models = r"./retailDynamicPricing/models/"
logs = r"./retailDynamicPricing/logs/"

# holdOuts = r"./holdOutData/"
# outputs = r"./outputs/"
# models = r"./models/"
# logs = r"./logs/"

# @st.cache_resource(suppress_st_warning=True)
def main ():
    try:
        st.title("Welcome to Retail Dynamic Pricing service")

        # if st.checkbox("Retail Churn Test Response", key='1'):
        #     RetailChurnTestResponse()
            
            #RetailChurnPrediction(holdOuts, outputs, models)
        if st.checkbox("Retail Dynamic Pricing Test Response", key='2'):
            RetailDynamicPricing(holdOuts, outputs, models)
        # if st.checkbox("Retail Churn Dashboard", key='3'):
        #     RetailChurnDashboard()

                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__main__':
    main()


