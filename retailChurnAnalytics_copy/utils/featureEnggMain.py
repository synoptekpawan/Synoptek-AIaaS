import pandas as pd
import numpy as np
from retailChurnAnalytics_copy.utils.churnUtility import *
from retailChurnAnalytics_copy.utils.featureEngg import *

import warnings
warnings.filterwarnings('ignore')


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
    allTaggedData1_ = allTaggedData1_[['Quantity', 'Value', 'UserId', 'Timestamp_', 'churnPeriod']]

    allTaggedData2_ = allTaggedData.copy()
    allTaggedData2_ = allTaggedData2_[['ProductCategory', 'ItemId', 'Location', 'TransactionId', 'UserId', 'churnPeriod', 'Timestamp_']]

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
    allFeatData_ = allFeatData_[['UserId', 'Age', 'Address', 'Gender', 'UserType',
       'Total_Quantity', 'Total_Value', 'Total_churnPeriod', 'StDev_Quantity',  
       'StDev_Value', 'StDev_churnPeriod', 'AvgTimeDelta', 'Recency',
       'Unique_ProductCategory', 'Unique_ItemId', 'Unique_Location',
       'Unique_TransactionId', 'Unique_churnPeriod',
       'Total_Quantity_per_Unique_ProductCategory',
       'Total_Quantity_per_Unique_ItemId',
       'Total_Quantity_per_Unique_Location',
       'Total_Quantity_per_Unique_TransactionId',
       'Total_Value_per_Unique_ProductCategory',
       'Total_Value_per_Unique_ItemId', 'Total_Value_per_Unique_Location',      
       'Total_Value_per_Unique_TransactionId','Label']]

    # ## save all feature engineered data
    # allFeatData_.to_csv(inputs+"allFeaturesData_.csv", index=False)

    return allFeatData_

