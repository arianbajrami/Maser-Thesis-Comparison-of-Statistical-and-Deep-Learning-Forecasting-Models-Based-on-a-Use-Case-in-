from cgi import test
from distutils import errors
from pyexpat import model
from turtle import width
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_pinball_loss
from sklearn.metrics import r2_score
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAXSpecification
from sklearn.linear_model import LinearRegression



###################
# Benchmarking
###################

## Simple Average
def simple_average(data):
    """
    SA: make a simple average


    parameter
    ---------
    data: dataset of time series

    returns
    -----------
    simple average of dataset
    """

    return np.mean(data)

def pred_avg(data, simple_average):
    """
    pred_avg: make a simple aveage prediction

    parameter
    ---------
    data: dataset of time series
    simple_average: mean of the data set
    returns
    -----------
    simple average prediction of dataset
    """
    preds = []
    for x in range(len(data)):
        x = simple_average
        preds.append(x)
    return preds


## Naive 1
def naive (data):
    """
    return last value of dataset

    parameter
    ---------
    data: dataset of time series
    simple_average: mean of the data set
    returns
    -----------
    return last value of dataset
    """
    return data[len(data)-1]

def naive_forecast(data,train):
    """
    make naive forecast, the last value is the prediction

    parameter
    ---------
    data: dataset of time series
    simple_average: mean of the data set
    
    train: train dataset 

    returns
    -----------
    naive forecast of the last value of the data set
    """
    naive_co2 = naive(train)
    preds = []
    for x in range(len(data)):
        x = naive_co2
        preds.append(x)
    return preds

## Naive 2

def naive_forecast_2(train, test):
    """
    make naive forecast, the 24 last values are the prediction

    parameter
    ---------
    data: dataset of time series
    simple_average: mean of the data set
    
    train: train dataset 

    returns
    -----------
    naive forecast of the values of the previous day
    """
    train_lag = train.shift(24)
    naive_pred = train_lag[len(train) - len(test) :]
    return naive_pred


## Simple moving aveage
def simple_moving_average(data, sma_window, forecast_period):
    """
    calculate moving average

    parameter
    ---------
    data: dataset of time series
    sma_windwo: sliding window for the moving aveage
    forecast_period: length of forecast period
    
    returns
    -----------
    preds: latest moving average of the data
    """
    sma = data.rolling(sma_window).mean()
    preds = []
    for x in range(forecast_period):
        x = sma.values[-1]
        #print(sma.values[-1])
        preds.append(x)
    return preds

def sma_forecast(train, test, data,sma_window, forecast_period):
    """
    make simple moving average predictions for the given forecast period

    parameter
    ---------
    train: train dataset of time series
    test: test dataset of time series
    data: dataset of time series
    sma_windwo: sliding window for the moving aveage
    forecast_period: length of forecast period
    
    returns
    -----------
    pred: simple moving average prediction for the lenth of the given test set
    report: evaluation metrics for forecast
    """
    pred = []
    for j in range(int(test.index[0]), test.index[-1], forecast_period):
        pred_update = simple_moving_average(train, sma_window, forecast_period)
        for k in pred_update:
            pred.append(k)
        train = data[j:j+forecast_period]
        assert(len(train) == len(pred_update))
    report = evaluation_without_uncertainty(pred, test, 'simple moving average')
    return pred,report

#########################
# Statistics
########################

## Exponential smoothing
def walk_forwad_validation_hw(test,train,data, forecast_period, alpha_low, alpha_high):
    """
    make hold winter exponential smoothin predictions for the given forecast period

    parameter
    ---------
    train: train dataset of time series
    test: test dataset of time series
    data: dataset of time series
    forecast_period: length of forecast period
    alpha_low: lower percentile
    alpha_high: higher percentile
    
    returns
    -----------
    pred: simple moving average prediction for the length of the given test set
    lower: lower quantile prediction 
    upper: upper quantile predction
    report: evaluation metrics for forecast
    """

    pred = []
    lower = []
    upper = []
    simulations = 1000 # number of simulated forecasts

    # start walk forward for length of data set
    for j in range(int(test.index[0]), test.index[-1], forecast_period):
        # define the model
        model = ExponentialSmoothing(train, seasonal_periods= 24,  seasonal = 'add' )
        #fit the model
        res = model.fit(optimized= True)
        # simulate a number of prediction with horizon = forecast_period
        sim = res.simulate(forecast_period,repetitions=simulations, error="mul")
        mean = np.mean(sim, axis = 1) # mean of simulaed predictions
        low_pred = np.quantile(sim,alpha_low,axis =1) # lower quantile 
        upp_pred = np.quantile(sim,alpha_high, axis =1) # upper quantile
        # safe current predictions
        for k in mean:
            pred.append(k)
        for l in low_pred:
            lower.append(l)
        for m in upp_pred:
            upper.append(m)
        # add next 24 hours from test to train 
        train = train.append(data[j:j+forecast_period])
    
    # calcualte the error metrics
    report = evaluation(pred, test, 'Hold Winters exp. smoothing')
    # return everything
    return pred,lower,upper, report


# ARIMA
def walk_forwad_validation_arima(test, forecast_period, res,alpha_low, alpha_high,n_features, method):
    """
    make (S)ARIMA predictions for the given forecast period

    parameter
    ---------
    test: test dataset of time series
    forecast_period: length of forecast period
    res: already fitted ARIMA model
    alpha_low: lower percentile
    alpha_high: higher percentile
    n_features: number of features
    method: type of prediction model

    returns
    -----------
    pred: simple moving average prediction for the length of the given test set
    lower: lower quantile prediction 
    upper: upper quantile predction
    report: evaluation metrics for forecast
    """
    pred = []
    upper = []
    lower = []
    # start walk forward for length of data set
    for j in range(0,  len(test), forecast_period):
        # forecast the next forecasst horizon
        forecast_update = res.get_forecast(forecast_period)
        # get average prediction
        f_mean = forecast_update.predicted_mean
        # get the confidence interval
        f_conf = forecast_update.conf_int()

        # save the predictions
        for k in f_mean:
            pred.append(k)
        for l in range(len(f_conf.values[:,0])):
            lower.append(f_conf.values[l,0])
        for m in range(len(f_conf.values[:,1])):
            upper.append(f_conf.values[m,1])
        # update the train data set with the current forecast horizon of the training set 
        train_update = test[j:j+forecast_period]
        # update the model -> set refit = True for refitting of model 
        res = res.append(train_update, refit = False)

    # get error metrics
    report =  evaluation(pred,lower,upper,test,alpha_low,alpha_high,n_features, method)
    return pred,report,lower,upper

# SARIMAX
def walk_forward_validation_sarimax(test_x,test_y, forecast_period, res, alpha_low, alpha_high,n_features, method):
    """
    make (S)ARIMAX predictions for the given forecast period

    parameter
    ---------
    test_x: test dataset of features for time series
    test_y: test dataset for target of prediction
    forecast_period: length of forecast period
    res: already fitted ARIMA model
    alpha_low: lower percentile
    alpha_high: higher percentile
    n_features: number of features
    method: type of prediction model

    returns
    -----------
    pred: simple moving average prediction for the length of the given test set
    lower: lower quantile prediction 
    upper: upper quantile predction
    report: evaluation metrics for forecast
    """
    pred = []
    upper = []
    lower = []
    # start walk forward for length of data set
    for j in range(0,  len(test_y), forecast_period):
        # make forecast for current forecast period
        forecast_update = res.get_forecast(forecast_period,exog = test_x[j:j+forecast_period])
        # get averaage prediction
        f_mean = forecast_update.predicted_mean
        # get confidence interval
        f_conf = forecast_update.conf_int()

        # save current predictions
        for k in f_mean:
            pred.append(k)
        for l in range(len(f_conf.values[:,0])):
            lower.append(f_conf.values[l,0])
        for m in range(len(f_conf.values[:,1])):
            upper.append(f_conf.values[m,1])
        
        # update the train set with current period from test data 
        train_update = test_y[j:j+forecast_period]

        # update the feature train set with features from current period
        x_update = test_x[j:j+forecast_period]
        
        # update the prediction model with new data
        # Select refit = TRUE if you want to refit the model
        res = res.append(train_update,exog = x_update, refit = False)
    
    # get error metrics
    report =  evaluation(pred,lower,upper, test_y,alpha_low,alpha_high,n_features, method)
    return pred,report,lower,upper


##################################
# Machine Learning
##################################

def predict_uncertaitny(model,test_x, alpha_low, alhpa_high ):
    """
    predict_uncertainty: calculate the aveage, the lower perecentile and upper percentile prediction of a shallow ML modell.
    model: predifined model that consists of an ensemble of base estimators.

    paramaters
    -------
    test_x: test set of the features that the model is trained on
    alpha_low: lower percentile
    alpha_high: upper percentile
    example: prediction of a 95% confidence interval means calculating the 0.025th and 0.975th percentile of the
             predictions of the base estimators. Than alpha_high - alpha_low = 0.95 
    
    returns
    -------
    pred: average prediction results
    lower: lower quantile predictions
    uppper: upper quantile predictions

    """ 
    lower = [] # list for saving the lower percentiles
    upper = [] # list for saving the upper perecenitls
    avg_pred = [] # list for saving the average predictions

    # make predictions
    for i in range(len(test_x)):
        preds = [] # list for saving the individual predicitons of the estimators
        for pred in model.estimators_:
            preds.append(pred.predict(test_x[i].reshape(1, -1))[0])
        #calculate average predictions ase well as lower and upper quantile
        lower.append(np.quantile(preds, alpha_low))
        upper.append(np.quantile(preds, alhpa_high))
        avg_pred.append(np.mean((preds)))
    return lower, avg_pred, upper


# Walk forward for non-probabilistic model
def walk_forward_ML(train_x,test_x,train_y,test_y,model,forecast_period,method):
    """
    make predictions for the given forecast period for a given ensemble ML method

    parameter
    ---------
    train_x: train dataset of features for time series
    test_x: test dataset of features for time series
    train_y: train dataset for target of prediction
    test_y: test dataset for target of prediction
    model: predifined prediction model
    forecast_period: length of forecast period
    method: type of prediction model

    returns
    -----------
    pred: simple moving average prediction for the length of the given test set
    report: evaluation metrics for forecast
    """
    length = len(test_y)
    pred = []

    # walk forward validation for lenght of dataset
    for j in range(0,  length, forecast_period):
        # fit the model
        model_fit = model.fit(train_x, train_y)
        # make prediction
        pred_upd = model_fit.predict(test_x[j:j+forecast_period])
        # save predictions
        for k in pred_upd:
            pred.append(k)
        # update train data
        train_y = pd.concat([train_y,test_y[j:j+forecast_period]])
        train_x = pd.concat([train_x,test_x[j:j+forecast_period]])
        
    # get error metrics
    report =  evaluation_without_uncertainty(pred, test_y, method)
    return pred,report 

# walk forwrd for ensemble model
def walk_forward_shallow(tscv, model, data, features, test,alpha_low, alpha_high,n_features, method):
    """"
    walk_forward_shallow: make a walk forward validation for a given ensemble method
    tscv: object that contains walk forward split of the data set

    Parameters
    -------
    model: predifined ensemble method
    data: given data-set
    test: test set of the data set
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: number of features
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

    #make wal forward validation
    for train_index, test_index in tscv.split(data):
        # refitting of the model each time the training set is increased
        model_fit = model.fit(features.values[train_index],data[train_index])
        # make uncertainty prediction based on the current training set
        lower_temp, pred_temp, upper_temp = predict_uncertaitny(model_fit,features.values[test_index],alpha_low, alpha_high)
        # save the predictions in lists 
        for k in pred_temp:
            #print(k)
            pred.append(k)
        for l in upper_temp:
            upper.append(l)
        for m in lower_temp:
            lower.append(m)

    #evalutae the uncetainty prediction
    report  = evaluation(pred,lower,upper, test,alpha_low, alpha_high,n_features, method)
    return pred, report, lower, upper


########################################
# Performance metrics
######################################## 


def interval_score(test_y, alpha, pred_low, pred_high):
    """
    IS: evaluate sharpeness of prediction intervall

    parameters
    ---------
    test_y: test data of the target
    alpha: width of prediction intervall
    pred_low: lower quantile predictions
    pred_high: higher quantile predictions

    returns
    --------
    IS-score: consists of sharpenes and calibration according to Gneiting et al.
    """
    #transform to arrays:
    test_y_ar = np.array(test_y)
    pred_low_ar = np.array(pred_low)
    pred_high_ar = np.array(pred_high)
    

    sharpeness = pred_high_ar - pred_low_ar
    calibration = (
        (np.clip(pred_low_ar - test_y_ar, a_min= 0, a_max=None  )
         +np.clip(test_y_ar - pred_high_ar, a_min= 0, a_max= None)
        ) * 2 / alpha
    )
    # inverval score is the average of sharpeness + calibration
    inverval_score = np.mean(sharpeness + calibration)
    return inverval_score


def evaluation_without_uncertainty(pred, test, method):
    """
    ealutaion: evaluate the given method

    paramaters
    -------
    pred: average prediction results
    test: test set of the target variable
    method: type of method
    
    returns
    -------
    report: report of the evalutaion metrics

    """  
    # start evaluation
    mae = mean_absolute_error(test, pred)
    mape = mean_absolute_percentage_error(test, pred)
    rmse = mean_squared_error(test, pred, squared = False)
    R2 = r2_score(test,pred)

    # safe the evaluations
    report = []
    report.append(f'MAE for {method}: {np.round(mae,2)} g_CO2/kWh')
    report.append(f'MAPE for {method}: {np.round(mape,4)*100} %')
    report.append(f'RMSE for {method}: {np.round(rmse,2)} g_CO2/kWh')
    report.append(f'R2 {method}: {np.round(R2,4)*100 } %')
    report = '\n'.join(report)
    return report


def evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    """
    ealutaion: evaluate the given method

    paramaters
    -------
    pred: average prediction results
    lower: lower quantile predictions
    uppper: upper quantile predictions
    test: test set of the target variable
    alpha_low: lower percentile
    alpha_hihg: higher percentile
    n_features: Number of features
    method: type of method
    
    returns
    -------
    report: report of the evalutaion metrics

    """  
    # calculate width of prediction intevall:
    alpha = alpha_high - alpha_low

    # start evaluation
    mae = mean_absolute_error(test, pred)
    mape = mean_absolute_percentage_error(test, pred)
    rmse = mean_squared_error(test, pred, squared = False)
    R2 = r2_score(test,pred)
    Adj_r2 = 1-(1-R2)*(len(test)-1)/(len(test)-n_features-1)
    pinball_low = mean_pinball_loss(test,lower,alpha = alpha_low)
    pinball_high = mean_pinball_loss(test,upper,alpha = alpha_high)
    ival_score = interval_score(test, alpha, lower, upper)
    #width = np.abs(np.mean(np.array(lower) - np.array(upper)))
    inside = round((np.sum([lower[i] <= test.values[i] <= upper[i] for i in range(len(test))]) / len(test)) * 100, 2)

    # add results to  report
    report = []
    report.append(f'MAE for {method}: {np.round(mae,2)} g_CO2/kWh')
    report.append(f'MAPE for {method}: {np.round(mape,4)*100} %')
    report.append(f'RMSE for {method}: {np.round(rmse,2)} g_CO2/kWh')
    report.append(f'Pinball for lower Quantile {method}: {np.round(pinball_low,2)} g_CO2/kWh')
    report.append(f'Pinball for higher Quantile {method}: {np.round(pinball_high,2)} g_CO2/kWh')
    report.append(f'Interval Score {method}: {np.round(ival_score,4)} g_CO2/kWh')
    #report.append(f'Average width of PI {method}: {np.round(width,2)} g_CO2/kWh')
    report.append(f'Real Values insisde PI {method}: {np.round(inside,4)} %')
    report.append(f'R2 {method}: {np.round(R2,4)*100 } %')
    report.append(f'Adjusted R2 {method}: {np.round(Adj_r2,4)*100} %')
    report = '\n'.join(report)
    return report

def evaluation_lstm(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    """
    ealutaion: evaluate the given method

    paramaters
    -------
    pred: average prediction results
    lower: lower quantile predictions
    uppper: upper quantile predictions
    test: test set of the target variable
    alpha_low: lower percentile
    alpha_hihg: higher percentile
    n_features: Number of features
    method: type of method
    
    returns
    -------
    report: report of the evalutaion metrics

    """  
    # calculate width of prediction intevall:
    alpha = alpha_high - alpha_low

    # start evaluation
    mae = mean_absolute_error(test, pred)
    mape = mean_absolute_percentage_error(test, pred)
    rmse = mean_squared_error(test, pred, squared = False)
    R2 = r2_score(test,pred)
    Adj_r2 = 1-(1-R2)*(len(test)-1)/(len(test)-n_features-1)
    pinball_low = mean_pinball_loss(test,lower,alpha = alpha_low)
    pinball_high = mean_pinball_loss(test,upper,alpha = alpha_high)
    ival_score = interval_score(test, alpha, lower, upper)
    #width = np.abs(np.mean(np.array(lower) - np.array(upper)))
    #inside = round((np.sum([lower[i] <= test[i] <= upper[i] for i in range(len(test))]) / len(test)) * 100, 2)

    # add results to  report
    report = []
    report.append(f'MAE for {method}: {np.round(mae,2)} g_CO2/kWh')
    report.append(f'MAPE for {method}: {np.round(mape,4)*100} %')
    report.append(f'RMSE for {method}: {np.round(rmse,2)} g_CO2/kWh')
    report.append(f'Pinball for lower Quantile {method}: {np.round(pinball_low,2)} g_CO2/kWh')
    report.append(f'Pinball for higher Quantile {method}: {np.round(pinball_high,2)} g_CO2/kWh')
    report.append(f'Interval Score {method}: {np.round(ival_score,4)} g_CO2/kWh')
    #report.append(f'Average width of PI {method}: {np.round(width,2)} g_CO2/kWh')
    #report.append(f'Real Values insisde PI {method}: {np.round(inside,4)} %')
    report.append(f'R2 {method}: {np.round(R2,4)*100 } %')
    report.append(f'Adjusted R2 {method}: {np.round(Adj_r2,4)*100} %')
    report = '\n'.join(report)
    return report


##########################
## Plotting
##########################


# define ffe style colurs
colours = {'black': [0,0,0],
             'darkblue': '#1F4E79', 
             'middle_blue': '#3795D5',
             'light_blue' : '#D7E6F5',
             '1B' : '#356CA5',
             '1D' : '#8AB5E1',
             'dark_red': '#AB2626',
             'dark_orange':'#B77201',
             'gold':'#F7D507',
             'middle_orange' : '#EC9302',
             'dark_green' : '#41641A',
             'middle_green' : '#92D050',
             'dark_gray' : '#515151'
             }


# plot non - probablistic models
def plot_model(test, pred, slice_start, slice_end, method):
    """
    plot the prediction results

    paramaters
    -------
    test: test set of the target variable
    pred: average prediction results
    slice_start: beginn of plotting
    slice_end: end of plotting
    method: type of method
    """  
    plt.figure(figsize = (24,8))
    test_slice = test.iloc[slice_start:slice_end]
    pred_slice = pred.iloc[slice_start:slice_end]
    plt.plot(test_slice, label = 'Test Data', color = colours.get('gold'))
    plt.plot(pred_slice, label = method, color = colours.get('1B'))
    plt.title(method)
    plt.xlabel('prediction step / h')
    plt.ylabel('co2 emmsiosn factor / g_CO2/kWh')
    plt.grid()
    plt.legend(loc = 'best')
    plt.show()


# plot probabilistic models
def plot_model_uncertainty(test, pred, lower, upper, date, slice_start, slice_end, method):
    """
    plot the prediction results

    paramaters
    -------
    test: test set of the target variable
    pred: average prediction results
    lower: lower quantile predictions
    uppper: upper quantile predictions
    date: datetime pandas object
    slice_start: beginn of plotting
    slice_end: end of plotting
    method: type of method
    """  

    test_plot = test.copy()
    test_plot.index = date.values
    pred_plot = pred.copy()
    pred_plot.index = date.values
    lower_plot = lower.copy()
    lower_plot.index = date.values
    upper_plot = upper.copy()
    upper_plot.index = date.values
    


    plt.figure(figsize = (24,8))
    test_slice = test_plot.loc[slice_start:slice_end]
    pred_slice = pred_plot.loc[slice_start:slice_end]
    lower_slice = lower_plot.loc[slice_start:slice_end]
    upper_slice = upper_plot.loc[slice_start:slice_end]
    #print(lower_slice.values)
    plt.fill_between(x= lower_slice.index, y1= lower_slice.values[:,0], y2=upper_slice.values[:,0], alpha=0.5,label = 'Prediction interval', color = colours.get('1D'))
    #plt.fill_between(x= range(slice_start,slice_end,1),y1= lower[slice_start:slice_end],y2=upper[slice_start:slice_end],alpha=0.5,label = 'Confidence interval', color = colours.get('1D'))
    plt.plot(test_slice, label = 'Test data', color = colours.get('gold'))
    plt.plot(pred_slice, label = method, color = colours.get('1B'))
    plt.title(method)
    plt.xlabel('Time in $h$')
    plt.ylabel('$CO_2$ emission factor in $g_{CO2}/kWh$')
    plt.rcParams.update({'font.size': 18})
    plt.grid()
    plt.legend(loc = 'best')
    plt.show()
    
def plot_model_uncertainty2(test, pred, lower, upper, date, slice_start, slice_end, method):
    """
    plot the prediction results

    paramaters
    -------
    test: test set of the target variable
    pred: average prediction results
    lower: lower quantile predictions
    uppper: upper quantile predictions
    date: datetime pandas object
    slice_start: beginn of plotting
    slice_end: end of plotting
    method: type of method
    """  

    test_plot = test.copy()
    test_plot.index = date.values
    pred_plot = pred.copy()
    pred_plot.index = date.values
    lower_plot = lower.copy()
    lower_plot.index = date.values
    upper_plot = upper.copy()
    upper_plot.index = date.values
    


    plt.figure(figsize = (24,8))
    test_slice = test_plot.loc[slice_start:slice_end]
    pred_slice = pred_plot.loc[slice_start:slice_end]
    lower_slice = lower_plot.loc[slice_start:slice_end]
    upper_slice = upper_plot.loc[slice_start:slice_end]
    #print(lower_slice.values)
    plt.fill_between(x= lower_slice.index, y1= lower_slice.values[:,0], y2=upper_slice.values[:,0], alpha=0.5,label = 'Prediction interval', color = colours.get('1D'))
    #plt.fill_between(x= range(slice_start,slice_end,1),y1= lower[slice_start:slice_end],y2=upper[slice_start:slice_end],alpha=0.5,label = 'Confidence interval', color = colours.get('1D'))
    plt.plot(test_slice, label = 'Validation data', color = colours.get('gold'))
    plt.plot(pred_slice, label = method, color = colours.get('1B'))
    plt.title(method)
    plt.xlabel('Time in $h$')
    plt.ylabel('$CO_2$ emission factor in $g_{CO2}/kWh$')
    plt.rcParams.update({'font.size': 18})
    plt.grid()
    plt.legend(loc = 'best')
    plt.show()




########################################
#  Feature Engineering
########################################

## create lagged features
def make_lags(ts, lags):
    """
    create lags for times series 
    paramters:
    ---------------
    ts: time series
    lags: number of lags that should be created
    
    returns:
    ---------------
    Datafram with number of lags = lags of time series
    
    """
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(1, lags + 1)
        },
        axis=1)

