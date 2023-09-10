import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
from streamlit_shap import st_shap
import shap
shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)
from retailChurnAnalytics_copy.evaluate import evalModel


holdOuts = r"./holdOutData/"
outputs = r"./outputs/"
models = r"./models/"

# -----------------------------------------------------------------------------------
def RetailChurnPrediction (holdOuts, outputs, models):
    try:
        st.write("[get the test data from here](https://synoptek-my.sharepoint.com/:f:/p/pchichghare/Eu9XEyzj9V5KrUv_bDHN0QwBiJCSx5zvkwegSjhsq2n8Ow?e=POv5Ea/)")
        st.write("Please upload the data for which churn is to be predicted")
        # getting the input data from the user
        userdata_  = pd.read_csv(holdOuts+"userDataHdo.csv")
        userdata_['UserId'] = userdata_['UserId'].astype(str)
        st.write("Sample User Data")
        st.dataframe(userdata_.head(1))
        uploaded_file1 = st.file_uploader("Please upload user data in csv format here", key='4')
        actdata_  = pd.read_csv(holdOuts+"activityDataHdo.csv")
        actdata_['TransactionId'] = actdata_['TransactionId'].astype(str)
        actdata_['UserId'] = actdata_['UserId'].astype(str)
        actdata_['ItemId'] = actdata_['ItemId'].astype(str)
        actdata_['year'] = actdata_['year'].astype(str)
        st.write("Sample Activity Data")
        st.dataframe(actdata_.head(1))
        uploaded_file2 = st.file_uploader("Please upload activity data in csv format here", key='5')
        if (uploaded_file1 is not None) & (uploaded_file2 is not None):
            if st.button("Upload Files & Run Model", key='6'):
                userdata  = pd.read_csv(uploaded_file1) #type: ignore
                #userdata.to_csv(holdOuts+"userDataHdo.csv", index=False)
                actdata  = pd.read_csv(uploaded_file2) #type: ignore
                #actdata.to_csv(holdOuts+"activityDataHdo.csv", index=False)

                churnPredDf, X_train, y_train, X_test, y_test, bestModel, selected_cols, holdSetFinal = evalModel(userdata, actdata, holdOuts, outputs, models)

                st.dataframe(churnPredDf)
                
                st.success('Thanks for using the service')
                
    except Exception as e:
        print(e)
# -----------------------------------------------------------------------------------
if __name__ == '__RetailChurnPrediction__':
    RetailChurnPrediction(holdOuts, outputs, models)