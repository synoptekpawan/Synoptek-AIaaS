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
from dataTagging import dataTaggingMain

# ---------------------------------------------------------------------------
def dataLabelingMain(inputs=None, f1=None, f2=None, churnPeriod_=0, churnThreshold_=0):
    ## load required data
    ## reading user data
    userData = f1 #pd.read_csv(inputs+f1)
    #logging.info(f'userdata shape: {userData.shape}')
    ## reading activity data
    activityData = f2 #pd.read_csv(inputs+f2)
    #logging.info(f'activitydata shape: {activityData.shape}')

    # -----------------------------------------------------------------------------------
    ## enter churn period & chrn threshold
    churnPeriod = churnPeriod_
    #logging.info(f'churn period: {churnPeriod}')
    churnThreshold = churnThreshold_
    #logging.info(f'churn threshold: {churnThreshold}')
    
    # -----------------------------------------------------------------------------------
    ## data cleaning
    ## drop duplicate rows
    userData.drop_duplicates(inplace=True)
    userData.reset_index(drop=True, inplace=True)
    ## fill missing values
    userData.fillna(-1, inplace=True)
    #print(userData.head())
    #logging.info(f'userdata shape: {userData.shape}')
    ## drop duplicate rows
    activityData.drop_duplicates(inplace=True)
    activityData.reset_index(drop=True, inplace=True)
    ## fill missing values
    activityData.fillna(-1, inplace=True)
    #print(activityData.head())
    #logging.info(f'activitydata shape: {activityData.shape}')
    
    # -----------------------------------------------------------------------------------
    ## data preparation
    ## join user & activity data
    userActData = pd.merge(userData, activityData, on='UserId', how='left')
    ## create churn DF
    churnDf = pd.DataFrame()
    churnDf['churnPeriod'] = [churnPeriod]
    churnDf['churnThreshold'] = [churnThreshold]
    
    # -----------------------------------------------------------------------------------
    ## data tagging   
    taggedDf = dataTaggingMain(userActData, churnDf)
    ## join joined data & check df data
    userActData_ = pd.merge(userActData, taggedDf, on='UserId', how='left')
    ## save the tagged for further process
    userActData_.to_csv(inputs+"allTaggedData_.csv", index=False)

# -----------------------------------------------------------------------------------
    return userActData_