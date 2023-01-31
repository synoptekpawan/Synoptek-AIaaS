## import required packages
import pandas as pd
import numpy as np
import datetime as dt
from dateutil import parser
import logging
from datetime import timedelta
import sys
import warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r"./retailChurnAnalytics/utils/")
from churnUtility import *

# main script
###
# This script assign churn status to each subscriber in the data
# To assign the churn status it uses the churn period start date passed as input as dataframe2( either from the uploaded dataset or the web service)

###


def dataTaggingMain(dataframe1 = None, dataframe2 = None):
    key_column='UserId'
    activity_column='TransactionId'
    
    ChurnPeriod=int(dataframe2.iloc[0]['churnPeriod'])
    ChurnThreshold=int(dataframe2.iloc[0]['churnThreshold'])
  
    #dataframe1['Timestamp'] = dataframe1.apply(lambda x : dt.datetime.fromtimestamp(x['Timestamp']).strftime('%Y-%m-%d'), axis=1)
    dataframe1['Timestamp'] = dataframe1['Timestamp'].astype('datetime64[ns]')
    
    # 
    dataframe1['Parsed_Date'] = pd.to_datetime(dataframe1['Timestamp'])
    ## Assigning Churn Status
    churnUtil=churnUtility(ChurnPeriod,ChurnThreshold)
    outdataframe=churnUtil.assign_churn_status(dataframe1,key_column=key_column,activity_column=activity_column) 
    #print('outdataframe.head()', outdataframe.head())
    #print('outdataframe.dtypes\n', outdataframe.dtypes)
    
    
    outdataframe.fillna('Unknown', inplace=True)
    outdataframe['churnPeriod']=ChurnPeriod
    to_keep_list=[each for each in outdataframe.columns if each!='Parsed_Date']
    
    ## Return value must be of a sequence of pandas.DataFrame
    return outdataframe[to_keep_list]