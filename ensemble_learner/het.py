import keras
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, MaxPooling1D, BatchNormalization, LSTM, Dropout
from scipy.special import jn_zeros
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import functions_2_ as fun
import tensorflow as tf
import numpy as np
#import window_generator as 
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample


def MLP(n_features, activation, loss, optimizer):
    model = Sequential([
        Dense(n_features, input_dim = n_features, kernel_initializer='normal', kernel_regularizer='l2', activation = activation),
        keras.layers.Dropout(0.2),
        Dense(64,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #keras.layers.Dropout(0.2),
        Dense(128,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #keras.layers.Dropout(0.2),
        Dense(256,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #keras.layers.Dropout(0.2),
        Dense(512,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #keras.layers.Dropout(0.2),
        Dense(1024,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #keras.layers.Dropout(0.2),
        Dense(2048,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        keras.layers.Dropout(0.6),
        Dense(4096,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        Dense(32,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        Dense(1, kernel_initializer='random_normal', kernel_regularizer='l2')
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model

def LSTM_model(n_features, loss, optimizer, dp_rate, initializer_lstm):
    model = Sequential([
        LSTM(1250,kernel_initializer= initializer_lstm,  return_sequences = True, input_shape = (24, n_features)),
        Dense(8, kernel_initializer= initializer_lstm),#, activation = 'relu'),
        Dropout(0.2),
        Dense(8, kernel_initializer= initializer_lstm),#, activation = 'relu'),
        Dense(1, kernel_initializer= initializer_lstm) # random_norma
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model



def walk_forward_validation_het(target, features ,test, target_lstm, features_lstm,scaler_target, prediction_step, sarma, sarmax, machine_learning, alpha_low, alpha_high, n_features, activation, loss, optimizer_mlp,optimizer_lstm, method):
    """
    make (S)ARIMAX predictions for the given forecast period

    parameter
    ---------
    target: given data-set for the target variable <- train set
    features: features for the given model < - train 
    test: test set of the data set
    features_lstm: LSTM features must be normalized
    target_lstm: LSTM target must be normalized
    scaler_target: Scaler function of LSTM target
    scaler: Scaler function for normilization
    prediction_step: prediction lenght
    sarma: Sarma model
    sarmax: sarmax model
    machine_learning: Machine Learning models
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: Number fo features
    activation: activaiton function
    loss: function
    optimizer_mlp: optimizer mlp
    optimizer_lstm: optimizer_lstm
    method: Ensemble method


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

    train_len = len(target) - len(test)
    whole_len =  len(target)

    learning_rate = 0.3
    loss = 'mse'
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    #activation = 'relu'
    dp_rate = 0.1
    metrics=[tf.keras.metrics.RootMeanSquaredError()]
    initializer_lstm = tf.keras.initializers.GlorotNormal()
    #initializer_lstm = tf.keras.initializers.GlorotNormal(seed=42)
    initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
    #initializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=42)

    callback = keras.callbacks.EarlyStopping(
    monitor = 'loss',
    verbose = 0,
    patience = 15,
    mode = 'auto',
    restore_best_weights = True
    )


    i = int(target_lstm.shape[0]) - int(len(test) / 24)
    print(i)
    # start walk forward for length of data set
    for j in range(train_len, whole_len, prediction_step):
        pred_temp = []
        upper_temp = []
        lower_temp = []


        ####### MLP prediction ########### 
        #mlp_model = MLP(n_features, activation, loss, optimizer_mlp)
        #mlp_model.fit( features[:j], target[:j], batch_size = 16, epochs = 100, shuffle = True)
        #pred_mlp = mlp_model.predict(features[j:j+prediction_step])
        #pred_mlp = pred_mlp.reshape(prediction_step,)
        #pred_temp.append(pred_mlp)
        #print(pred_temp)

        ####### LSTM prediction ###########
        target_temp_lstm = resample(target_lstm[:i], replace= True, random_state=42)
        features_temp_lstm = resample(features_lstm[:i], replace= True, random_state=42)
      
        assert(prediction_step % 24 == 0)
        lstm_step = int (prediction_step / 24)
      
        lstm = LSTM_model(n_features, loss, optimizer, dp_rate, initializer_lstm)
        lstm.fit(features_temp_lstm[:i], target_temp_lstm[:i], validation_data = (features_lstm[i:i+lstm_step], target_lstm[i:i+lstm_step]), batch_size = 4, epochs = 40, callbacks = callback, shuffle = True)
      
        print(i)
       
        pred_lstm = lstm.predict(features_lstm[i:i+lstm_step])
        i = i+lstm_step
        print(i)
        pred_lstm = pred_lstm.reshape(pred_lstm.shape[0]*24,)
        print(pred_lstm.shape)
        pred_lstm = scaler_target.inverse_transform([pred_lstm])
        pred_lstm = pred_lstm.reshape(prediction_step,)
        print(pred_lstm.shape)
        #print(pred_lstm)
        pred_temp.append(pred_lstm),

        ######## Machine Learning prediction ###############
        for member in machine_learning:
          target_temp = resample(target[:j], replace= True, random_state=member)
          features_temp = resample(features[:j], replace= True, random_state=member)
          machine_learning[member].fit(features_temp, target_temp)
          pred_temp.append(machine_learning[member].predict(features[j:j+prediction_step]))
        
        ######## Sarmax prediction ######################
        preds_sarmax = sarmax.get_forecast(prediction_step, exog = features[j:j+prediction_step])
        pred_sarmax = np.array(preds_sarmax.predicted_mean)
        pred_temp.append(pred_sarmax)
        
        sarmax = sarmax.append(target[j:j+prediction_step],exog = features[j:j+prediction_step], refit = False)
        ######## Sarma prediction ######################
        preds_sarma = (sarma.get_forecast(prediction_step))
        pred_sarma = np.array(preds_sarma.predicted_mean)
        pred_temp.append(pred_sarma)
        sarma = sarma.append(target[j:j+prediction_step], refit = True)
        
        print(pred_temp)
        pred_temp = np.array(pred_temp)
        #print(pred_temp)

        mean_temp = np.mean(pred_temp, axis = 0)
        #print(mean_temp.shape)
        lower_temp = np.quantile(pred_temp, 0.025, axis = 0)
        upper_temp = np.quantile(pred_temp, 0.975, axis = 0)

        for k in mean_temp:
            #print(k)
            pred.append(k)
        for l in upper_temp:
            upper.append(l)
        for m in lower_temp:
            lower.append(m)

    pred = np.array(pred).reshape(len(pred),)
    lower = np.array(lower).reshape(len(lower),)
    upper = np.array(upper).reshape(len(upper),)
    print(pred.shape)
    print(test.shape)
    print(lower.shape)
    # get error metrics
    #report = fun.evaluation_without_uncertainty(pred,test,method)
    report =  fun.evaluation(pred,lower,upper, test,alpha_low,alpha_high,n_features, method)
    return pred,report,lower,upper, pred_temp

















