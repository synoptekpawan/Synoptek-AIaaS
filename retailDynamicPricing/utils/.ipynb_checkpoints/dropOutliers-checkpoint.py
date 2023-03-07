import numpy as np
import pandas as pd
from scipy import stats

def outliers(data):
    # calculate z-scores for each data point
    z_scores = np.abs((data - np.mean(data)) / np.std(data))

    # define a threshold for outlier detection
    threshold = 3

    # identify outliers
    outliers = np.where(z_scores > threshold)

    # print the outliers
    print("Outliers:", outliers[0])
    
    return outliers[0]

    # remove outliers
    
def drop_outliers(data):
    newDf = []
    for cmbs in data.IS_COMBO.unique():
        xx = data[['SELL_ID','PRICE','QUANTITY']][data['IS_COMBO']==cmbs]
        xx.reset_index(drop=True, inplace=True)
        #print(xx.shape)
        #print(cmbs)
        ch = outliers(xx)
        we = xx.drop(ch.tolist(), axis=0)
        #print(we.shape)
        newDf.append(we)

    dataNew = pd.concat(newDf, ignore_index=True)
    #dataNew.head()
    dataNew = pd.merge(dataNew, data, on=['SELL_ID','PRICE','QUANTITY'], how='left')
    dataNew.drop_duplicates(ignore_index=True, inplace=True)
    
    return dataNew