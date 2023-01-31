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


holdOuts = r"./retailChurnAnalytics/holdOutData/"
outputs = r"./retailChurnAnalytics/outputs/"
models = r"./retailChurnAnalytics/models/"
logs = r"./retailChurnAnalytics/logs/"

# -----------------------------------------------------------------------------------
## prepare logging config
logging.basicConfig(filename=logs+'modelEvalMain.log', 
                    level = logging.DEBUG,
                    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

# -----------------------------------------------------------------------------------
def RetailChurnPrediction (holdOuts, outputs, models):
    try:
        #st.write("get the test data from here [link](https://share.streamlit.io/mesmith027/streamlit_webapps/main/MC_pi/streamlit_app.py)")
        st.title("Please upload the data for which churn is to be predicted")
        # getting the input data from the user
        uploaded_file1 = st.file_uploader("Please upload user data in csv format here", key='4')
        uploaded_file2 = st.file_uploader("Please upload activity data in csv format here", key='5')
        if (uploaded_file1 is not None) & (uploaded_file2 is not None):
            if st.button("Upload Files & Run Model", key='3'):
                userdata  = pd.read_csv(uploaded_file1)
                userdata.to_csv(holdOuts+"userDataHdo.csv", index=False)
                actdata  = pd.read_csv(uploaded_file2)
                actdata.to_csv(holdOuts+"activityDataHdo.csv", index=False)

                churnPredDf = evalModel(userdata, actdata, holdOuts, outputs, models)

                st.dataframe(churnPredDf)

                st.success('Thanks for using the service')
                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__RetailChurnPrediction__':
    RetailChurnPrediction(holdOuts, outputs, models)

