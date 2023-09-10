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
# sys.path.insert(0, r"./utils/")
# sys.path.insert(0, r"./retailChurnAnalytics/utils/")
from retailChurnAnalytics_copy.utils.churnUtility import *
# from retailChurnAnalytics.utils.churnUtility import *

def createNumericFeatures(dataframe1 = None, dataframe2 = None):
    key_column='UserId'
    IsDevelopment=True

    #specifying churn start period
    churn_period=dataframe1.iloc[0]['churnPeriod']   

    #       Feature Engineering
    churnUtil=churnUtility(churn_period=churn_period)
    #       Feature Engineering
    dataframe2=churnUtil.calculateNumericalDataFeatures(dataframe1,key_column, 
                 summable_columns=dataframe1.columns,
                        rename_label='overall', IsDevelopment=IsDevelopment)
    
    return dataframe2



def createStrFeatures(dataframe1 = None, dataframe2 = None):
    key_column='UserId'
    IsDevelopment=True

    #specifying churn start period
    churn_period=dataframe1.iloc[0]['churnPeriod']   

    #       Feature Engineering

    churnUtil=churnUtility(churn_period=churn_period)
    dataframe2=churnUtil.calculateStringDataFeatures(dataframe1,key_column, uniquable_columns=dataframe1.columns,
                 rename_label='overall', IsDevelopment=IsDevelopment)
    
    return dataframe2



def calculateAverages_(dataframe1 = None, dataframe2 = None):
    key_column='UserId'
    IsDevelopment=True

    #       Feature Engineering
    churnUtil=churnUtility()
    df=churnUtil.calculateAverages(dataframe1,dataframe2,
        key_column, uniquable_columns=dataframe2.columns, 
        summable_columns=dataframe1.columns)
    
    # Return value must be of a sequence of pandas.DataFrame
    return df