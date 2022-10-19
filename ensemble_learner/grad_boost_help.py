###############
# Import all modules
##############

import sklearn
import functions as fun

from sklearn.ensemble import GradientBoostingRegressor


##################
# Walk forward testing
#################



def walk_forward_gb(tscv, all_models, target, features, test,alpha_low, alpha_high,n_features, method, refit:bool):
    """"
    walk_forward_gb: walk forward validation for GradientBoostingRegressor in sklaern, with probabilsitic forecasts. 

    Parameters
    -------
    tscv: time series split form sklearn
    all_models: dict of predifend gradient boosting models
    target: given data-set for the target variable 
    test: test set of the data set
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: number of features
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed
    method: type of ensemble method

    Returns
    -------
    pred: average prediction results
    report: report of the evalutation metrics on the test set
    lower: lower quantile predictions
    uppper: upper quantile predictions

    """

    pred = [] # list for saving the average predictions
    lower = [] # list for saving the lower percentiles
    upper = [] # list for saving the upper perecenitls
    
    # fit the models before the walk forward validation if refit = False
    if refit == False:
        train_len = int(len(target) - len(test))
        gbr_low = all_models['q %1.3f' % alpha_low].fit(features.values[:train_len],target[:train_len])
        gbr_mean = all_models['mean'].fit(features.values[:train_len],target[:train_len])
        gbr_high = all_models['q %1.3f' % alpha_high].fit(features.values[:train_len],target[:train_len])

    #make wal forward validation
    for train_index, test_index in tscv.split(target):
        # refitting each of the models each time the training set is increased fi refit = True
        if refit == True:
            gbr_low = all_models['q %1.3f' % alpha_low].fit(features.values[train_index], target[train_index])
            gbr_mean = all_models['mean'].fit(features.values[train_index], target[train_index])
            gbr_high = all_models['q %1.3f' % alpha_high].fit(features.values[train_index], target[train_index])

        # make uncertainty prediction based on the current training set
        lower_temp = gbr_low.predict(features.values[test_index])
        mean_temp= gbr_mean.predict(features.values[test_index])
        upper_temp = gbr_high.predict(features.values[test_index])

        # save the predictions in lists 
        for m in lower_temp:
            lower.append(m)
        for k in mean_temp:
            pred.append(k)
        for l in upper_temp:
            upper.append(l)
        
    #evalutae the uncetainty prediction
    report  = fun.evaluation(pred,lower,upper, test,alpha_low, alpha_high,n_features, method)
    return pred, report, lower, upper