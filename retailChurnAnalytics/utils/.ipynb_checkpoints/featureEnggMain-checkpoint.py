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
sys.path.insert(0, r"C:/Users/pawanc/Desktop/retailAnalytics/retailChurnAnalytics/utils/")
from churnUtility import *
from featureEngg import *

# -----------------------------------------------------------------------------------
def featureEnggMain(inputs, filename_):
    ## load required data
        ## reading user data
    allTaggedData = pd.read_csv(inputs+filename_) # allTaggedData_.csv
    #logging.info(f'allTaggedData shape:: {allTaggedData.shape}')

    # -----------------------------------------------------------------------------------
    ## data preparation
    cols=allTaggedData.select_dtypes(exclude = ['int','float', 'datetime']).columns.to_list()
    allTaggedData[cols] = allTaggedData[cols].astype('category')

    allTaggedData_ = allTaggedData.copy()
    allTaggedData_ = allTaggedData_[['Age', 'Address', 'Gender', 'UserType', 'UserId', 'Label']]
    allTaggedData_.drop_duplicates(inplace=True, ignore_index=True)

    allTaggedData1_ = allTaggedData.copy()
    allTaggedData1_ = allTaggedData1_[['Quantity', 'Value', 'UserId', 'Timestamp', 'churnPeriod']]

    allTaggedData2_ = allTaggedData.copy()
    allTaggedData2_ = allTaggedData2_[['ProductCategory', 'ItemId', 'Location', 'TransactionId', 'UserId', 'churnPeriod', 'Timestamp']]

    # -----------------------------------------------------------------------------------
    ## create numeric features
    cnf_ = createNumericFeatures(allTaggedData1_)

    ## create str features
    csf_ = createStrFeatures(allTaggedData2_)

    ## calculate means
    ca_ = calculateAverages_(cnf_, csf_)

    # -----------------------------------------------------------------------------------
    ## combine all feature datasets
    cnsf_ = pd.merge(cnf_, csf_, on='UserId', how='outer')
    cnsf_.drop_duplicates(inplace=True, ignore_index=True)

    cnsf1_ = pd.merge(cnsf_, ca_, on='UserId', how='outer')
    cnsf1_.drop_duplicates(inplace=True, ignore_index=True)

    allFeatData_ = pd.merge(allTaggedData_, cnsf1_, on='UserId', how='inner')
    allFeatData_.drop_duplicates(inplace=True, ignore_index=True)

    ## save all feature engineered data
    allFeatData_.to_csv(inputs+"allFeaturesData_.csv", index=False)

# -----------------------------------------------------------------------------------   
    return allFeatData_