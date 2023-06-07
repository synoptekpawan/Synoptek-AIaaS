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
from tpot import TPOTClassifier
import random
import numpy as np
import pickle
from dotenv import load_dotenv
import os

np.random.seed(10)
random.seed(10)

sys.path.insert(0, r"./retailChurnAnalytics/utils/")
# sys.path.insert(0, r"./utils/")
from churnUtility import *
from dataLabelingMain import dataLabelingMain
from featureEnggMain import featureEnggMain
from featureSelectionMain import trainTestSplitWithBestFeatMain
 

# -----------------------------------------------------------------------------------
# ## prepare logging config
# logging.basicConfig(filename='./logs/modelTrainValidMain.log', 
#                     level = logging.DEBUG,
#                     format='%(asctime)s -%(name)s - %(levelname)s - %(message)s', 
#                     datefmt='%d-%b-%y %H:%M:%S')

# -----------------------------------------------------------------------------------
## load date and folder variables
inputs = r"retailChurnAnalytics/inputs/"
outputs = r"retailChurnAnalytics/outputs/"
models = r"retailChurnAnalytics/models/"

# inputs = r"./inputs/"
# outputs = r"./outputs/"
# models = r"./models/"

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
host, user, passwd, db_name, port = load_envFile()  # type: ignore
tbName1 = 'churnUserData'
f1 = pullDataFromDb(tbName1, host, user, passwd, db_name, port)
#f1 = f1.drop(['Id'], axis=1)
f1.to_csv(inputs+"userData.csv", index=False)  # type: ignore
#print(f1.head())

tbName2 = 'churnActivityData'
f2 = pullDataFromDb(tbName2, host, user, passwd, db_name, port)
#f2 = f2.drop(['Id'], axis=1)
f2.to_csv(inputs+"activityData.csv", index=False)  # type: ignore
#print(f2.head())


# -----------------------------------------------------------------------------------
## data tagging
# f1 = 'userData.csv'
# f2 = 'activityData.csv'
allTaggedData = dataLabelingMain(inputs=inputs, f1=f1, f2=f2, churnPeriod_=30, churnThreshold_=0)
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
print(X_train_fs.shape, X_test_fs.shape, y_train.shape, y_test.shape)  # type: ignore
print(bestK_)
print(selected_cols)

# save the best features to disk
f = models+'bestFeatures_.pkl'
pickle.dump(selected_cols, open(f, 'wb'))

# save the X_train to disk
f = models+'X_train.pkl'
pickle.dump(X_train_fs, open(f, 'wb'))

# save the y_train to disk
f = models+'y_train.pkl'
pickle.dump(y_train, open(f, 'wb'))

# save the X_train to disk
f = models+'X_test.pkl'
pickle.dump(X_test_fs, open(f, 'wb'))

# save the y_train to disk
f = models+'y_test.pkl'
pickle.dump(y_test, open(f, 'wb'))

# -----------------------------------------------------------------------------------
## model training
try:
    ## train, validate ML model1
    tpot = TPOTClassifier(generations=5, verbosity=2,  population_size=40, random_state=42)
    tpot.fit(X_train_fs, y_train)

    pred_tpot = tpot.predict(X_test_fs)

    acc_tpot = accuracy_score(y_test, pred_tpot)

    rocAuc_tpot = roc_auc_score(y_test, pred_tpot)
    print(f'accuracy score: {acc_tpot} , rocAuc score: {rocAuc_tpot}')
    for idx, (name, transform) in enumerate(tpot.fitted_pipeline_.steps, start=1):
        best_model = transform
        #print(f'{idx}. {transform}')
    print(best_model)
    # save the best ML model to disk
    f = models+'bestModel_.pkl'
    pickle.dump(best_model, open(f, 'wb'))

except Exception as e:
    logging.exception("Exception occurred")

# ---------------------------------------------------------------------------



