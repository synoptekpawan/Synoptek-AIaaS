import pandas as pd
import numpy as np
import pickle
from retailChurnAnalytics_copy.utils.churnUtility import *
from retailChurnAnalytics_copy.utils.dataLabelingMain import dataLabelingMain
from retailChurnAnalytics_copy.utils.featureEnggMain import featureEnggMain

# ## load date and folder variables
# holdOuts = r"retailChurnAnalytics/holdOutData/"
# outputs = r"retailChurnAnalytics/outputs/"
# models = r"retailChurnAnalytics/models/"

holdOuts = r"./holdOutData/"
outputs = r"./outputs/"
models = r"./models/"

def evalModel (f1, f2, holdOuts, outputs, models):

    # f1 = pd.read_csv(holdOuts+'userDataHdo.csv')
    # f2 = pd.read_csv(holdOuts+'activityDataHdo.csv')

    churnPeriod_=21
    churnThreshold_=0

    ## data tagging
    allTaggedData = dataLabelingMain(inputs=holdOuts, f1=f1, f2=f2, churnPeriod_=churnPeriod_, churnThreshold_=churnThreshold_)

    ## feature engg
    filename_ = 'allTaggedData_.csv'
    allFeatData = featureEnggMain(holdOuts, filename_)

    ## load the best features from disk
    f = models+'bestFeatures_.pkl'
    selected_cols = pickle.load(open(f, 'rb'))
    # print(selected_cols)
    # print(len(selected_cols))

    ## load the X_train from disk
    f = models+'X_train.pkl'
    X_train = pickle.load(open(f, 'rb'))
    # print(X_train.shape)

    ## load the X_train from disk
    f = models+'y_train.pkl'
    y_train = pickle.load(open(f, 'rb'))
    # print(y_train.shape)

    ## load the X_test from disk
    f = models+'X_test.pkl'
    X_test = pickle.load(open(f, 'rb'))
    # print(X_test.shape)

    ## load the y_test from disk
    f = models+'y_test.pkl'
    y_test = pickle.load(open(f, 'rb'))
    # print(y_test.shape)

    allFeatData_ = pd.get_dummies(allFeatData, columns = ['Address', 'Gender','UserType','Label'], drop_first=True)

    holdSet = allFeatData_.copy()

    selected_cols_ = set(selected_cols)
    found_cols = set(holdSet.columns)
    not_found_cols = list(selected_cols_ - found_cols)
    # print(not_found_cols)

    for cols in not_found_cols:
        holdSet[cols] = len(holdSet)*[0]

    holdSetFinal = holdSet[selected_cols]

    # load the best model disk
    f = models+'bestModel_.pkl' # today_, yesterday_
    bestModel = pickle.load(open(f, 'rb'))
    print(bestModel)

    predOnHoldSet = bestModel.predict(holdSetFinal)
    churnPredDf = allFeatData.copy()

    churnPredDf['Churn'] = predOnHoldSet
    churnPredDf = churnPredDf[['UserId','Age','Address','Gender','UserType','Churn']]

    churnPredDf.to_csv(outputs+'churnPredDf_.csv')
    
    
    return churnPredDf, X_train, y_train, X_test, y_test, bestModel, selected_cols, holdSetFinal


# f1 = pd.read_csv(holdOuts+'userDataHdo.csv')
# f2 = pd.read_csv(holdOuts+'activityDataHdo.csv')
# predf, X_train, y_train, X_test, y_test, bestModel, selected_cols, holdSetFinal = evalModel (f1, f2, holdOuts, outputs, models)
# print(predf)
