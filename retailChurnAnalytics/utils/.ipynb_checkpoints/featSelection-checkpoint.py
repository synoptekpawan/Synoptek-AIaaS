# compare different numbers of features selected using anova f-test
from numpy import mean
from numpy import std
from pandas import read_csv
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from matplotlib import pyplot
import random
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
np.random.seed(10)
random.seed(10)


# load the dataset
def load_dataset(filename):
    # load the dataset as a pandas DataFrame
    data = read_csv(filename)
    data = pd.get_dummies(data, columns = ['Address', 'Gender','UserType','Label'], drop_first=True)
    # retrieve numpy array
    dataset = data.values
    # split into input (X) and output (y) variables
    X = dataset[:, 1:-1]
    y = dataset[:,-1]
    #print(X.shape, y.shape)
    return X, y, data
 
# evaluate a give model using cross-validation
def evaluate_model(model, X, y):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=42)
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1, error_score='raise')
    return scores

def my_func(X, y):
    return mutual_info_classif(X, y, random_state=42, n_neighbors=3, discrete_features='auto')