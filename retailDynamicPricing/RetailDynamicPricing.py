import streamlit as st
# Import the reqiured libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style="ticks", color_codes=True)
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from tpot import TPOTRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from sklearn.linear_model import HuberRegressor, LinearRegression
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
import os
import sys
import pickle
sys.path.insert(0, r"./retailDynamicPricing/utils/")
# sys.path.insert(0, r"./utils/")
from evaluateModelOnHoldData import evalModel
# Need to load JS vis in the notebook

holdOuts = r"./retailDynamicPricing/holdOutData/"
outputs = r"./retailDynamicPricing/outputs/"
models = r"./retailDynamicPricing/models/"
logs = r"./retailDynamicPricing/logs/"

# holdOuts = r"./holdOutData/"
# outputs = r"./outputs/"
# models = r"./models/"
# logs = r"./logs/"

# -----------------------------------------------------------------------------------
# ## prepare logging config
# logging.basicConfig(filename=logs+'modelEvalMain.log', 
#                     level = logging.DEBUG,
#                     format='%(asctime)s -%(name)s - %(levelname)s - %(message)s', 
#                     datefmt='%d-%b-%y %H:%M:%S')

# -----------------------------------------------------------------------------------


def RetailDynamicPricing (holdOuts, outputs, models):
    try:
        #st.write("[get the test data from here](https://synoptek-my.sharepoint.com/:f:/p/pchichghare/Eu9XEyzj9V5KrUv_bDHN0QwBiJCSx5zvkwegSjhsq2n8Ow?e=POv5Ea/)")
        #st.write("Please upload the data for which churn is to be predicted")
        # getting the input data from the user
        itms = st.text_input('Enter item name in comma separated format',)
        itms_ = itms.split(",")
        #st.write(itms_)
        
        buying_price = st.text_input('Enter the buying price in comma separated format',)
        buying_price1_ = buying_price.split(",")
        buying_price1X_ = [float(price) for price in buying_price1_]
        #st.write(buying_price1X_)
        
        is_combo = st.text_input('Is item sold as combo (0/1) in comma separated format',)
        is_combo1_ = is_combo.split(",")
        is_combo1X_ = [int(val) for val in is_combo1_]
        #st.write(is_combo1X_)
        
        

        pricingDf, values_at_max_profit = evalModel(itms_, buying_price1X_, is_combo1X_, holdOuts, outputs, models)
        
        st.write("Predicted profits & qunatities against test prices")
        st.dataframe(pricingDf.style.format({"PRICE": "{:.2f}", "QUANTITY": "{:.0f}", "PROFIT": "{:.2f}"}))
        st.write("Max profit & qunatity against test price")
        st.dataframe(values_at_max_profit.style.format({"PRICE": "{:.2f}", "QUANTITY": "{:.0f}", "PROFIT": "{:.2f}"}))
        
        fig, ax = plt.subplots(figsize=(12,7))
        lns1 = ax.plot(pricingDf['PRICE'],pricingDf['QUANTITY'], label='PRICE vs QUANTITY', color='orange')
        ax.set_xlabel('PRICE')
        ax.set_ylabel('QUANTITY')
        ax2=ax.twinx()

        lns2 = ax2.plot(pricingDf['PRICE'],pricingDf['PROFIT'], label='PRICE vs PROFIT', color='blue')
        ax2.set_ylabel('PROFIT')

        # added these three lines
        lns = lns1+lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=0)

        ind = np.where(pricingDf['PROFIT'] == pricingDf['PROFIT'].max())[0][0]
        values_at_max_profit = pricingDf.iloc[[ind]]

        _ = ax.vlines(x=values_at_max_profit['PRICE'].values[0], ymin=0, 
                      ymax=values_at_max_profit['PROFIT'].values[0], colors='r', linestyle='--')

        
        st.pyplot(fig)


        st.success('Thanks for using the service')
                
    except Exception as e:
        logging.exception("Exception occurred")
        
        
# -----------------------------------------------------------------------------------
if __name__ == '__RetailDynamicPricing__':
    RetailDynamicPricing(holdOuts, outputs, models)

