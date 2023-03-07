from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import train_test_split, RepeatedKFold
from tpot import TPOTRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVR
from tpot.builtins import StackingEstimator, ZeroCount
from tpot.export_utils import set_param_recursive


def trainedModel(data):
    X_train, X_test, y_train, y_test = train_test_split(data[['PRICE']], data[['QUANTITY']], test_size=0.3, random_state=42)
    #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    # define evaluation procedure
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define search
    tpotModel = TPOTRegressor(generations=5, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=42, n_jobs=-1)
    # perform the search
    tpotModel.fit(X_train, y_train)

    bestModel_ = []
    for idx, (name, transform) in enumerate(tpotModel.fitted_pipeline_.steps, start=1):
        bestModel = transform
        #print(f'{idx}. {transform}')
        bestModel_.append(bestModel)
    bestModel1_ = tuple(bestModel_)

    exported_pipeline = make_pipeline(*bestModel1_)
    #print(exported_pipeline)
    
    # # save the best features to disk
    # f = models+itm+'_withOutCombo.pkl'
    # pickle.dump(exported_pipeline, open(f, 'wb'))

    # Fix random state for all the steps in exported pipeline
    set_param_recursive(exported_pipeline.steps, 'random_state', 42)

    exported_pipeline.fit(X_train, y_train)
    
    #elasticity = exported_pipeline.steps[-1][1].coef_[-1]

    pred_tpot_tr = exported_pipeline.predict(X_train)
    pred_tpot_te = exported_pipeline.predict(X_test)

    mse_tr = mean_squared_error(y_train, pred_tpot_tr)
    mse_te = mean_squared_error(y_test, pred_tpot_te)
    #print('mse_tr',mse_tr, 'mse_te',mse_te)

    mae_tr = mean_absolute_error(y_train, pred_tpot_tr)
    mae_te = mean_absolute_error(y_test, pred_tpot_te)
    #print('mae_tr',mae_tr, 'mae_te',mae_te)

    mape_tr = mean_absolute_percentage_error(y_train, pred_tpot_tr)
    mape_te = mean_absolute_percentage_error(y_test, pred_tpot_te)
    #print('mape_tr',mape_tr, 'mape_te',mape_te)

    r2score_tr = r2_score(y_train, pred_tpot_tr)
    r2score_te = r2_score(y_test, pred_tpot_te)
    #print('r2score_tr',r2score_tr, 'r2score_te',r2score_te)
    
    return exported_pipeline, mse_tr, mae_tr, mape_tr, r2score_tr, mse_te, mae_te, mape_te, r2score_te 