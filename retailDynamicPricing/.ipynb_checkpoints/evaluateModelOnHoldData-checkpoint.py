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
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive
import os
import sys
import pickle
sys.path.insert(0, r"./retailDynamicPricing/utils/")
# sys.path.insert(0, r"./utils/")
from dropOutliers import drop_outliers
from model_selection import trainedModel
# -----------------------------------------------------------------------------------
## load date and folder variables
holdOuts = r"./retailChurnAnalytics/holdOutData/"
outputs = r"./retailChurnAnalytics/outputs/"
models = r"./retailChurnAnalytics/models/"

# holdOuts = r"./holdOutData/"
# outputs = r"./outputs/"
# models = r"./models/"

# today_ = dt.datetime.today().date()
# # print(today_)
# yesterday_ = today_ - timedelta(1)
# # print(yesterday_)

# -----------------------------------------------------------------------------------
# Load the data
# sold = pd.read_csv(holdOuts+'SaleMetaData_HD.csv')
# print('sold data: ',sold.shape)
# transactions = pd.read_csv(holdOuts+'transactionsN1_HD.csv')
# date_info = pd.read_csv(holdOuts+'DateInfo_HD.csv')

def evalModel (itms, buying_price, is_combo, holdOuts, outputs, models):
    for itm, price, combo in zip(itms, buying_price, is_combo):
        print(itm)
        start_price = round(price,2)
        end_price = start_price + 10
        end_price = round(end_price,2)
        
        
        if combo == 0:
            is_combo1 = 'withOutCombo'
            # load the model from disk
            f = models+itm+'_'+is_combo1+'.pkl'
            bestModel = pickle.load(open(f, 'rb'))
            
            test = pd.DataFrame(columns = ["PRICE", "QUANTITY"])
            test['PRICE'] = np.arange(start_price, end_price, 0.5)
            test['QUANTITY'] = bestModel.predict(test[['PRICE']])
            test['PROFIT'] = (test["PRICE"] - buying_price) * test["QUANTITY"]
            
            ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
            values_at_max_profit = test.iloc[[ind]]
            
            return test, values_at_max_profit
            
        else:
            is_combo1 = 'withCombo'
            # load the model from disk
            f = models+itm+'_'+is_combo1+'.pkl'
            bestModel = pickle.load(open(f, 'rb'))
            
            test = pd.DataFrame(columns = ["PRICE", "QUANTITY"])
            test['PRICE'] = np.arange(start_price, end_price, 0.5)
            test['QUANTITY'] = bestModel.predict(test[['PRICE']])
            test['PROFIT'] = (test["PRICE"] - buying_price) * test["QUANTITY"]
            
            ind = np.where(test['PROFIT'] == test['PROFIT'].max())[0][0]
            values_at_max_profit = test.iloc[[ind]]
            
            return test, values_at_max_profit
            




