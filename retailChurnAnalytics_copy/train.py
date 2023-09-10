import pandas as pd
import numpy as np
import pickle
from utils.dataLabelingMain import dataLabelingMain
from utils.featureEnggMain import featureEnggMain
from utils.featureSelectionMain import trainTestSplitWithBestFeatMain
from tpot import TPOTClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings('ignore')

## load date and folder variables
# inputs = r"./retailChurnAnalytics/inputs/"
# outputs = r"./retailChurnAnalytics/outputs/"
# models = r"./retailChurnAnalytics/models/"

inputs = r"./inputs/"
outputs = r"./outputs/"
models = r"./models/"

f1 = pd.read_csv(inputs+"userData.csv")
# print(f1.columns)

f2 = pd.read_csv(inputs+"activityData.csv")
# print(f2.columns)


allTaggedData = dataLabelingMain(inputs=inputs, f1=f1, f2=f2, churnPeriod_=30, churnThreshold_=0)
# print(allTaggedData.shape)
# print(allTaggedData.columns)

## feature engg
filename_ = 'allTaggedData_.csv'
allFeatData = featureEnggMain(inputs, filename_)
# print(allFeatData.shape)
# print(allFeatData.columns)

## feature selection
filename_ = 'allFeaturesData_.csv'
X_train_fs, X_test_fs, y_train, y_test, bestK_, selected_cols = trainTestSplitWithBestFeatMain(inputs, 
                                                                                               filename_, 
                                                                                               test_size=0.3, 
                                                                                               random_state=42)
# print(X_train_fs.shape, X_test_fs.shape, y_train.shape, y_test.shape)  # type: ignore
# print(bestK_)
# print(selected_cols)

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

## model training
try:
    ## train, validate ML model1
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    rf.fit(X_train_fs, y_train)

    pred_rf = rf.predict(X_test_fs)

    acc_rf = accuracy_score(y_test, pred_rf)

    rocAuc_rf = roc_auc_score(y_test, pred_rf)
    print(f'accuracy score: {acc_rf} , rocAuc score: {rocAuc_rf}')
    # save the best ML model to disk
    f = models+'bestModel_.pkl'
    pickle.dump(rf, open(f, 'wb'))

except Exception as e:
    print(e)

