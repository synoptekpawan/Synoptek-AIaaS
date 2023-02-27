import streamlit as st
import pandas as pd
import numpy as np
import logging
import pickle
import random
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale = 1.1)
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score, accuracy_score, confusion_matrix, RocCurveDisplay
from streamlit_shap import st_shap
import shap
shap.initjs()
st.set_option('deprecation.showPyplotGlobalUse', False)
np.random.seed(10)
random.seed(10)
import sys
sys.path.insert(0, r"./retailChurnAnalytics/")
from evaluateModelOnHoldData import evalModel
# Need to load JS vis in the notebook



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
                
                # model evaluation metrics
                st.title("Model evaluation metrics")
                st.write("Best Model Selected: ",bestModel)
                predTr_ = bestModel.predict(X_train)
                predTe_ = bestModel.predict(X_test)
                
                accTr_ = accuracy_score(y_train, predTr_)
                accTe_ = accuracy_score(y_test, predTe_)
                st.write('Model Train Accuracy Score: ', round(accTr_,3), 'Model Test Accuracy Score: ', round(accTe_,3))
                #st.write('Model Test Accuracy Score: ', round(accTe_,3))
                
                # st.write('Below is a DataFrame:', data_frame, 'Above is a dataframe.')
                
                rocAucTr_ = roc_auc_score(y_train, predTr_)
                rocAucTe_ = roc_auc_score(y_test, predTe_)
                st.write('Model Train AUC Score: ', round(rocAucTr_,3), 'Model Test AUC Score: ', round(rocAucTe_,3)) #type: ignore
                #st.write('Model Test AUC Score: ', round(rocAucTe_,3))
            
                # Create a model explainer
                model_explainer = shap.Explainer(bestModel.predict, holdSetFinal, feature_names=selected_cols)

                # Shap values with explainer
                shap_values = model_explainer(holdSetFinal)

                # Feature Importance
                st.title("Feature importance")
                st_shap(shap.summary_plot(shap_values, holdSetFinal, feature_names=selected_cols, plot_type="bar"), height=500, width=1000)
                
                # Global feature importance
                st.title("Global feature interpretability")
                st_shap(shap.summary_plot(shap_values, holdSetFinal, feature_names=selected_cols), height=500, width=1000)
                
                # Local Interpretability
                st.title("Local feature interpretability for Non Churner")
                st_shap(shap.waterfall_plot(shap_values[0]), height=500, width=1000)
                
                # Local Interpretability
                st.title("Local feature interpretability for Churner")
                st_shap(shap.waterfall_plot(shap_values[2]), height=500, width=1000)
                
                # visualize all the predictions
                st.title("All the predictions")
                st_shap(shap.plots.force(shap_values), height=300, width=1000)
           
                st.success('Thanks for using the service')
                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__RetailChurnPrediction__':
    RetailChurnPrediction(holdOuts, outputs, models)

