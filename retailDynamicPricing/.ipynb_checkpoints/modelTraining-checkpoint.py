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
# sys.path.insert(0, r"./retailDynamicPricing/utils/")
sys.path.insert(0, r"./utils/")
from dropOutliers import drop_outliers
from model_selection import trainedModel


## load folder variables
# inputs = r"../retailDynamicPricing/inputs/"
# outputs = r"../retailDynamicPricing/outputs/"
# models = r"../retailDynamicPricing/models/"

inputs = r"./inputs/"
outputs = r"./outputs/"
models = r"./models/"

# Load the data
sold = pd.read_csv(inputs+'SaleMetaData_.csv')
print('sold data: ',sold.shape)
transactions = pd.read_csv(inputs+'transactionsN1_.csv')
date_info = pd.read_csv(inputs+'DateInfo.csv')

# data preparation
transactions['CALENDAR_DATE'] = pd.to_datetime(transactions['CALENDAR_DATE'])
print('transactions data: ',transactions.shape)

date_info.drop(['IS_SCHOOLBREAK', 'AVERAGE_TEMPERATURE', 'IS_OUTDOOR'], axis=1, inplace=True)
date_info['CALENDAR_DATE'] = pd.to_datetime(date_info['CALENDAR_DATE'])
date_info['HOLIDAY'] = date_info['HOLIDAY'].fillna("No Holiday")
print('date_info data: ',date_info.shape)

# combine sold and transaction data
data1 = pd.merge(sold, transactions, on =  'SELL_ID')
b = data1.groupby(['SELL_ID', 'ITEM_ID','ITEM_NAME', 'CALENDAR_DATE','PRICE','IS_COMBO']).QUANTITY.sum()
intermediate_data = b.reset_index()
print('intermediate_data data: ',intermediate_data.shape)

# combine intermediate data with date info
combined_data = pd.merge(intermediate_data, date_info, on = 'CALENDAR_DATE')
print('combined_data data: ',combined_data.shape)

# prepare bau data by dropping holidays and weekends
bau_data = combined_data[(combined_data['HOLIDAY']=='No Holiday') & \
                         (combined_data['IS_WEEKEND']==0)]
bau_data.reset_index(drop=True, inplace=True)
print('bau_data data: ',bau_data.shape)

# model training
#elasticities = {}
for itm in bau_data.ITEM_NAME.unique():
    print(itm)
    ch = bau_data[bau_data['ITEM_NAME']==itm]
    ch.reset_index(drop=True, inplace=True)
    
    # drop outliers
    chNew = drop_outliers(ch)
    print(chNew.shape, ch.shape)
    
    withOutCombo = chNew[chNew['IS_COMBO']==0]
    if len(withOutCombo) > 1:
        
        # train the model on data
        regModel, mse_tr, mae_tr, mape_tr, r2score_tr, mse_te, mae_te, mape_te, r2score_te = trainedModel(withOutCombo)
        
        # save the best features to disk
        f = models+itm+'_withOutCombo.pkl'
        pickle.dump(regModel, open(f, 'wb'))
        
        # recording elasticities
        #elasticities[itm] = elasticity
        
        # prepare metrcis dataframe
        df = pd.DataFrame()
        df['Train'] = [mse_tr, mae_tr, mape_tr, r2score_tr]
        df['Test'] = [mse_te, mae_te, mape_te, r2score_te]
        df = df.rename(index={0: 'mse', 1: 'mae', 2: 'mape', 3: 'r2score'})
        df.to_csv(outputs+itm+'_withOutCombo.csv', index=False)
        print(df)
        
        
    else:
        pass
    
    withCombo = chNew[chNew['IS_COMBO']==1]
    if len(withCombo) > 1:
        # train the model on data
        regModel, mse_tr, mae_tr, mape_tr, r2score_tr, mse_te, mae_te, mape_te, r2score_te = trainedModel(withCombo)
        
        # save the best features to disk
        f = models+itm+'_withCombo.pkl'
        pickle.dump(regModel, open(f, 'wb'))
        
        # recording elasticities
        #elasticities[itm] = elasticity
        
        # prepare metrcis dataframe
        df = pd.DataFrame()
        df['Train'] = [mse_tr, mae_tr, mape_tr, r2score_tr]
        df['Test'] = [mse_te, mae_te, mape_te, r2score_te]
        df = df.rename(index={0: 'mse', 1: 'mae', 2: 'mape', 3: 'r2score'})
        df.to_csv(outputs+itm+'_withCombo.csv', index=False)
        print(df)
    else:
        pass
    
    