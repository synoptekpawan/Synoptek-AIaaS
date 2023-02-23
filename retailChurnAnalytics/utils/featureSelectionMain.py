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


np.random.seed(10)
random.seed(10)
#sys.path.insert(0, r"./utils/")
sys.path.insert(0, r"./retailChurnAnalytics/utils/")
from churnUtility import *
from featureEngg import *
from featSelection import *

# ---------------------------------------------------------------------------
## function used for feature selection in model training script
def trainTestSplitWithBestFeatMain(inputs, filename_, test_size=0.0, random_state=0):
    ## find the number of features required for model training
    # define dataset
    X, y, data = load_dataset(inputs+filename_) # allFeaturesData_.csv
    #print(data.columns)
    # define number of features to evaluate
    num_features = [i+1 for i in range(X.shape[1])]
    # enumerate each number of features
    resultsD_ = dict()
    #results_ = []
    for k in num_features:
        # create pipeline
        lr = LogisticRegression(random_state=random_state)
        fs = SelectKBest(score_func=my_func, k=k)
        pipeline = Pipeline(steps=[('mic',fs), ('lr', lr)])
        # evaluate the model
        scores = evaluate_model(pipeline, X, y)
        q3, q1 = np.percentile(scores, [75 ,25])
        iqr_ = q3 - q1
        resultsD_[k] = iqr_
        bestK_ = min(resultsD_, key=resultsD_.get)
        #results_.append(scores)

    ## split the data set into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    ## selecting final best features
    fs = SelectKBest(score_func = my_func, k=bestK_) 
    X_train_fs = fs.fit_transform(X_train, y_train)
    X_test_fs = fs.transform(X_test)
    ## print the selected columns
    cols = fs.get_support(indices=True)
    selected_cols = data.iloc[:, 1:-1].iloc[:, cols].columns.tolist()
    #print(selected_cols)
        
# ---------------------------------------------------------------------------       
    return (X_train_fs, X_test_fs, y_train, y_test, bestK_, selected_cols)