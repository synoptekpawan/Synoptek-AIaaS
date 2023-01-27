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

# compare different numbers of features selected using anova f-test
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import random
import numpy as np
import pickle


np.random.seed(10)
random.seed(10)

sys.path.insert(0, r".../../utils/")
from churnUtility import *
from dataLabelingMain import dataLabelingMain
from featureEnggMain import featureEnggMain
from featureSelectionMain import trainTestSplitWithBestFeatMain
 
# -----------------------------------------------------------------------------------
## load date and folder variables
holdOuts = r".../../holdOutData/"
outputs = r".../../outputs/"
models = r".../../models/"

today_ = dt.datetime.today().date()
# print(today_)
yesterday_ = today_ - timedelta(1)
# print(yesterday_)

# -----------------------------------------------------------------------------------

def evalModel (f1, f2, holdOuts, outputs, models):
    churnPeriod_=21
    churnThreshold_=0
    
    ## data tagging
    #f1 = pd.read_csv(holdOuts+'userDataHdo.csv')
    #f2 = pd.read_csv(holdOuts+'activityDataHdo.csv')
    allTaggedData = dataLabelingMain(inputs=holdOuts, f1=f1, f2=f2, churnPeriod_=churnPeriod_, churnThreshold_=churnThreshold_)
    #print(allTaggedData.shape)
    #print(allTaggedData.columns)

    # -----------------------------------------------------------------------------------
    ## feature engg
    filename_ = 'allTaggedData_.csv'
    allFeatData = featureEnggMain(holdOuts, filename_)
    #print(allFeatData.shape)
    #print('all Features Data', allFeatData.columns)
    #print(allFeatData.head())

    # -----------------------------------------------------------------------------------
    # load the best features from disk
    f = models+'bestFeatures_.pkl'
    selected_cols = pickle.load(open(f, 'rb'))
    print(selected_cols)

    allFeatData_ = pd.get_dummies(allFeatData, columns = ['Address', 'Gender','UserType','Label'], drop_first=True)
    #print(allFeatData_.columns.tolist())

    featureSet = list(set(selected_cols).intersection(allFeatData_.columns.tolist()))
    print(featureSet)

    diffSet = list(set(selected_cols) - set(allFeatData_.columns.tolist()))
    print(diffSet)

    holdSet = allFeatData_.copy()
    holdSetFinal = holdSet[featureSet]

    # holdSetFinal['Age_E'] = len(holdSetFinal)*[0]
    # holdSetFinal['Address_C'] = len(holdSetFinal)*[0]
    # holdSetFinal['Age_I'] = len(holdSetFinal)*[0]
    # holdSetFinal['Age_G'] = len(holdSetFinal)*[0]
    # holdSetFinal['Address_D'] = len(holdSetFinal)*[0]

    #print(holdSetFinal.shape)

    # -----------------------------------------------------------------------------------
    ## Get model preodiction hold out set

    # load the best model disk
    f = models+'bestModel_.pkl' # today_, yesterday_
    bestModel = pickle.load(open(f, 'rb'))

    predOnHoldSet = bestModel.predict(holdSetFinal)

    #print(predOnHoldSet)

    churnPredDf = allFeatData.copy()
    #churnPredDf['UserId'] = allFeatData['UserId']
    churnPredDf['Churn'] = predOnHoldSet
    churnPredDf = churnPredDf[['UserId','Age','Address','Gender','UserType','Churn']]

    #print(churnPredDf)

    churnPredDf.to_csv(outputs+'churnPredDf_.csv')
    
    return churnPredDf



