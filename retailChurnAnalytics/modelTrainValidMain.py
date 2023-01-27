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
from sqlalchemy import create_engine
from sqlalchemy import types
import psycopg2

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
from dotenv import load_dotenv
import os

np.random.seed(10)
random.seed(10)

sys.path.insert(0, r"D:/reatilanalytics/retailChurnAnalytics/utils/")
from churnUtility import *
from dataLabelingMain import dataLabelingMain
from featureEnggMain import featureEnggMain
from featureSelectionMain import trainTestSplitWithBestFeatMain
 

# -----------------------------------------------------------------------------------
## prepare logging config
logging.basicConfig(filename='./logs/modelTrainValidMain.log', 
                    level = logging.DEBUG,
                    format='%(asctime)s -%(name)s - %(levelname)s - %(message)s', 
                    datefmt='%d-%b-%y %H:%M:%S')

# -----------------------------------------------------------------------------------
## load date and folder variables
inputs = r"D:/reatilanalytics/retailChurnAnalytics/inputs/"
outputs = r"D:/reatilanalytics/retailChurnAnalytics/outputs/"
models = r"D:/reatilanalytics/retailChurnAnalytics/models/"

today_ = dt.datetime.today().date()
# print(today_)
yesterday_ = today_ - timedelta(1)
# print(yesterday_)

# -----------------------------------------------------------------------------------
## load the credentials here
def load_envFile():
    try:
        ## load the dotenv
        load_dotenv()
        
        ## fetch the credentials
        host = os.environ["host"] 
        user = os.environ["user"]
        passwd = os.environ["passwd"]
        db_name = os.environ["db_name"]
        port = os.environ["port"]
        
    except Exception as e:
        logging.exception("Exception occurred")
    
    else:
        return (host, user, passwd, db_name, port)
# -----------------------------------------------------------------------------------
## pull skubssmapping from database here
def pullDataFromDb(tbName, host, user, passwd, db_name, port):
    
    try:
        ## create the connection
        host = host
        user = user
        passwd = passwd
        db_name = db_name
        port = port

        engine = create_engine('postgresql://'+user+':'+passwd+'@'+host+':'+str(port)+'/'+db_name, echo=False)

        res = engine.execute(f'SELECT * FROM public."{tbName}"')

        result = res.fetchall()

        data = pd.DataFrame(result)
        data.columns = result[0].keys()

    except Exception as e:
        logging.exception("Exception occurred")
    
    else:
        return data

# -----------------------------------------------------------------------------------
## get db creds
host, user, passwd, db_name, port = load_envFile()
tbName1 = 'churnUserData'
f1 = pullDataFromDb(tbName1, host, user, passwd, db_name, port)
f1 = f1.drop(['Id'], axis=1)
f1.to_csv(inputs+"userData.csv", index=False)
#print(f1.head())

tbName2 = 'churnActivityData'
f2 = pullDataFromDb(tbName2, host, user, passwd, db_name, port)
f2 = f2.drop(['Id'], axis=1)
f2.to_csv(inputs+"activityData.csv", index=False)
#print(f2.head())


# -----------------------------------------------------------------------------------
## data tagging
# f1 = 'userData.csv'
# f2 = 'activityData.csv'
allTaggedData = dataLabelingMain(inputs=inputs, f1=f1, f2=f2, churnPeriod_=21, churnThreshold_=0)
print(allTaggedData.shape)
print(allTaggedData.columns)

# -----------------------------------------------------------------------------------
## feature engg
filename_ = 'allTaggedData_.csv'
allFeatData = featureEnggMain(inputs, filename_)
print(allFeatData.shape)
print(allFeatData.columns)

# -----------------------------------------------------------------------------------
## feature selection
filename_ = 'allFeaturesData_.csv'
X_train_fs, X_test_fs, y_train, y_test, bestK_, selected_cols = trainTestSplitWithBestFeatMain(inputs, 
                                                                                               filename_, 
                                                                                               test_size=0.3, 
                                                                                               random_state=42)
print(X_train_fs.shape, X_test_fs.shape, y_train.shape, y_test.shape)
print(bestK_)
print(selected_cols)

# save the best features to disk
f = models+'bestFeatures_.pkl'
pickle.dump(selected_cols, open(f, 'wb'))


# -----------------------------------------------------------------------------------

try:
    ## train, validate ML model1
    lr = LogisticRegression(random_state=42)

    dist = dict(C=np.arange(0, 10, 1), penalty=['l2', 'l1'], tol=np.arange(1e-4, 0.1, 0.1))

    clf = GridSearchCV(lr, dist, cv=5, scoring='accuracy')

    search_space = clf.fit(X_train_fs, y_train)

    lr_ = LogisticRegression(C=search_space.best_params_['C'], 
                             penalty= search_space.best_params_['penalty'], 
                             tol= search_space.best_params_['tol'], 
                             random_state=42)

    lr_.fit(X_train_fs, y_train)

    pred_lr = lr_.predict(X_test_fs)

    acc_lr = accuracy_score(y_test, pred_lr)

    rocAuc_lr = roc_auc_score(y_test, pred_lr)
    print(f'logistic regression - accuracy score: {acc_lr} , rocAuc score: {rocAuc_lr}')

except Exception as e:
    logging.exception("Exception occurred")

# ---------------------------------------------------------------------------
try:
    ## train, validate ML model2
    bdt = AdaBoostClassifier(
        DecisionTreeClassifier(min_samples_leaf = 10, random_state = 42, max_depth = 1), random_state=42
        )

    dist = dict(algorithm = ["SAMME", "SAMME.R"], n_estimators = np.arange(10, 100, 10))
    clf = GridSearchCV(bdt, dist, cv = 5, scoring = 'accuracy')

    search_space = clf.fit(X_train_fs, y_train)

    dt_ = AdaBoostClassifier(DecisionTreeClassifier(min_samples_leaf = 10, random_state = 42, max_depth = 1), 
                             n_estimators = search_space.best_params_['n_estimators'], 
                             algorithm = search_space.best_params_['algorithm'], 
                             random_state=42)

    dt_.fit(X_train_fs, y_train)

    pred_dt = dt_.predict(X_test_fs)

    acc_dt = accuracy_score(y_test, pred_dt)

    rocAuc_dt = roc_auc_score(y_test, pred_dt)
    print(f'boosted decision tree - accuracy score: {acc_dt} , rocAuc score: {rocAuc_dt}')
except Exception as e:
    logging.exception("Exception occurred")

# ---------------------------------------------------------------------------
try:
    ## Select best model of 2 and save it
    modelDict = {}
    modelDict['lr'] = (rocAuc_lr, acc_lr, lr_)
    modelDict['dt'] = (rocAuc_dt, acc_dt, dt_)
    #print(modelDict)
    
    bestModel = max(modelDict, key = modelDict.get)
    
    print(modelDict[bestModel][2])
    
    # save the best ML model to disk
    f = models+'bestModel_.pkl'
    pickle.dump(modelDict[bestModel][2], open(f, 'wb'))

except Exception as e:
    logging.exception("Exception occurred")

# ---------------------------------------------------------------------------



