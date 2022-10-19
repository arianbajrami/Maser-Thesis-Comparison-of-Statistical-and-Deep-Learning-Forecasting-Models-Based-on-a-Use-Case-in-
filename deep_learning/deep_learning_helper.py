import keras
from keras.models import Model, Sequential
from keras import backend as K
from keras.layers import Dense
from keras.layers import Conv1D, Flatten, MaxPooling1D, BatchNormalization, LSTM
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import functions_2_ as fun
import tensorflow as tf
from numpy.random import seed
from sklearn.utils import resample

#seed(42)
#tf.random.set_seed (42)

## weird gpu stuff
#physical_devices = tf.config.experimental.list_physical_devices('GPU')
#if len(physical_devices) > 0:
#    tf.config.experimental.set_memory_growth(physical_devices[0], True)

################
# layer definition
###########
class Dropout(keras.layers.Dropout):
    """Applies Dropout to the input.
    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](
           http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """
    import keras.backend as K
    def __init__(self, rate, training=None, noise_shape=None, seed=None, **kwargs):
        super(Dropout, self).__init__(rate, noise_shape=None, seed=None,**kwargs)
        self.training = training

        
    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return K.dropout(inputs, self.rate, noise_shape,
                                 seed=self.seed)
            if not training: 
                return K.in_train_phase(dropped_inputs, inputs, training=self.training)
            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs


##################
# Define the model
################# 

####### MLP
def MLP(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer):
    model = Sequential([
        Dense(n_features, input_dim = n_features, kernel_initializer='normal', activation = activation),
        keras.layers.Dropout(dp_rate, seed = 42),
        Dense(64,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(128,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(256,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(256,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(512,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(1024,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42, training = True),
        Dense(2048,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, seed = 42,  training = True),
        Dense(4096,kernel_initializer= initializer, kernel_regularizer='l2',  activation = activation),
        Dense(32,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dense(1, kernel_initializer= initializer)
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model

def MLP_2(n_features, learning_rate, activation, loss, optimizer, dp_rate):
    model = Sequential([
        Dense(n_features, input_dim = n_features, kernel_initializer='normal', kernel_regularizer='l2', activation = activation),
        keras.layers.Dropout(0.2),
        Dense(64,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        Dense(128,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        #Dense(256,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        Dense(256,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        Dense(512,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        Dense(1024,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        #Dropout(dp_rate, training = True),
        #keras.layers.Dropout(0.2),
        Dense(2048,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, training = True),
        Dense(4096,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        Dense(32,kernel_initializer='random_normal', kernel_regularizer='l2', activation = activation),
        Dense(1, kernel_initializer='random_normal', kernel_regularizer='l2')
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model

#### CNN

def CNN(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer, initializer_cnn):
    model = Sequential([
        Conv1D(128,3,kernel_initializer= initializer_cnn,  kernel_regularizer='l2',  activation = activation, input_shape = (32,1)),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128,3,kernel_initializer= initializer_cnn, kernel_regularizer='l2', activation = activation),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128,3,kernel_initializer= initializer_cnn, kernel_regularizer='l2', activation = activation),
        BatchNormalization(),
        MaxPooling1D(2),
        Conv1D(128,2,kernel_initializer= initializer_cnn, kernel_regularizer='l2', activation = activation),
        Flatten(),
        Dense(512,kernel_initializer= initializer,kernel_regularizer='l2',  activation = activation),
        Dropout(dp_rate, training = True),
        Dense(512,kernel_initializer= initializer,kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, training = True),
        Dense(512,kernel_initializer= initializer, kernel_regularizer='l2', activation = activation),
        Dropout(dp_rate, training = True),
        Dense(512,kernel_initializer= initializer, kernel_regularizer='l2',  activation = activation),
        Dropout(dp_rate, training = True),
        Dense(512,kernel_initializer= initializer, kernel_regularizer='l2',  activation = activation),
        Dense(1, kernel_initializer= initializer,  kernel_regularizer='l2')
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model

#########
# LSTM
########
def LSTM_model(n_features, loss, optimizer, dp_rate, initializer_lstm):
    model = Sequential([
        LSTM(1250,kernel_initializer= initializer_lstm,  return_sequences = True, input_shape = (24, n_features)),
        Dense(32, kernel_initializer= initializer_lstm),#, activation = 'relu'),
        Dropout(dp_rate, training = True),
        Dense(128, kernel_initializer= initializer_lstm),#, activation = 'relu'),
        Dense(1, kernel_initializer= initializer_lstm) # random_norma
    ])
    model.compile(loss = loss, optimizer = optimizer)
    return model


###################
# walk forward validation
###############
def predict_dist(x, model, num_predictions):
    preds = []
    for _ in range(num_predictions):
        preds += [(model.predict(x, verbose  = False))]
    return preds

def predictions(x, model, num_predictions, alpha_low, alpha_upp):
    pred_dist = predict_dist(x, model, num_predictions)
    lower = np.quantile(np.array(pred_dist), alpha_low, axis = 0)
    avg = np.mean(np.array(pred_dist), axis = 0)
    upper = np.quantile(np.array(pred_dist), alpha_upp, axis = 0)
    return lower, avg, upper



def walk_forward_deep_learning_point_pred(tscv, model, target, features, test, callback ,learning_rate, epochs, method, refit:bool):
    """
    walk_forward_gb: walk forward validation for GradientBoostingRegressor in sklaern, with probabilsitic forecasts. 

    Parameters
    -------
    tscv: time series split form sklearn
    NN: Neural Network model
    target: given data-set for the target variable <- train set
    features: features for the given model < - train set
    test: test set of the data set
    callback: callback instance of keras
    learning_rate: learning rate for updating the model with new data
    epochs: number of training epochs
    method: type of ensemble method
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed

    Returns
    -------
    pred: average prediction results
    report: report of the evalutation metrics on the test set
    
    """
    
    pred = [] # list for saving the average predictions
    # fit the models before the walk forward validation if refit = False
    
    # define new optimizer
    optimizer = keras.optimizers.Adamax(0.0005)
    if refit == False:
        train_len = int(len(target) - len(test))
        # define configuration for update of learning rate
        model.compile(loss = 'mean_absolute_error', optimizer = optimizer)
        # updat the model wtih new data
        model.fit(features.values[:train_len],target[:train_len],  epochs = epochs, callbacks = callback, shuffle = False)

    #make wal forward validation
    for train_index, test_index in tscv.split(target):
        # refitting each of the models each time the training set is increased fi refit = True
        if refit == True:
            model.compile(loss = loss, optimizer = optimizer)
            model.fit(features.values[train_index], target[train_index],epochs = epochs, callbacks = callback, shuffle = False)
        # make uncertainty prediction based on the current training set
        pred_temp = model.predict(features.values[test_index])

        # save the predictions in lists 
        for l in pred_temp:
            pred.append(l)
        
    #evalutae the uncetainty prediction
    report  = evaluation(mlp_val, lower, upper, co2_val,alpha_low,alpha_upp,n_features, method = 'MLP')
    # evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    return pred, report


def walk_forward_deep_learning_mlp(tscv, target, features, test, alpha_low, alpha_high, n_features, callback ,learning_rate, loss, optimizer, activation, dp_rate, initializer, epochs, batch_size, num_preds, method, refit:bool, validation:bool):
    """
    walk_forward_gb: walk forward validation for keras deep learning architectures, with probabilsitic forecasts. 

    Parameters
    -------
    tscv: time series split form sklearn
    target: given data-set for the target variable <- train set
    features: features for the given model < - train 
    test: test set of the data set
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: Number fo features
    callback: callback instance of keras
    learning_rate: learning rate for updating the model with new data
    loss: Loss function
    optimizer: optimizer functio
    activation: Activation function
    dp_rate: Dropout reat
    initializer: Weight initilalizer
    epochs: number of training epochs
    batch_size: Batch size
    num_preds: Number of Monte Carlo predictions
    method: Deep Learning mdoel
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed
    validation->Bool: True if validation phase, False For test phase

    Returns
    -------
    pred: average prediction results
    lower: lower qunatile predictions
    upper: upper qunatile predicitons
    report: report of the evalutation metrics on the test set
    
    """
    
    pred = [] # list for saving the average predictions
    lower = []
    upper = []
    # fit the models before the walk forward validation if refit = False
    
    # define new optimizer
    #optimizer = keras.optimizers.Adamax(learning_rate)
    if refit == False:
        train_len = int(len(target) - len(test) - int(0.05*len(target)))
        print(train_len)
        val_len = int(0.05*len(target)) + train_len
        print(val_len)
        # define configuration for update of learning rate
        model = MLP_2(n_features, learning_rate, activation, loss, optimizer, dp_rate)
        model.compile(loss = loss, optimizer = optimizer)
        # updat the model wtih new data
        # make usage of validation callback if validation ist true
        if validation == False:
            model.fit(features[:train_len],target[:train_len], validation_data = (features[train_len:val_len],target[train_len:val_len]), batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 1) #shuffle = False)
        else:
            model.fit(features[:train_len],target[:train_len], validation_data = (features[train_len:],target[train_len:]), batch_size = batch_size,  epochs = epochs, callbacks = callback, verbose = 1)#, shuffle = False)

    #make wal forward validation
    for train_index, test_index in tscv.split(target):
        # refitting each of the models each time the training set is increased if refit = True
        train_len = len(train_index) - int(len(test_index))
        print(train_len)
        val_len = int(len(test_index)) + train_len
        print(val_len)
        if refit == True:
            model = MLP_2(n_features, learning_rate, activation, loss, optimizer, dp_rate)
            model.compile(loss = loss, optimizer = optimizer)
            if validation == False:
                model.fit(features[:train_len], target[:train_len], validation_data = (features[train_len:val_len],target[train_len:val_len]), batch_size = batch_size,epochs = epochs, callbacks = callback, verbose = 0 )#, shuffle = False)
            else:
                model.fit(features.values[train_index], target[train_index], validation_data = (features.values[test_index], target[test_index]),batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 0)#, shuffle = False)
        # make uncertainty prediction based on the current training set
        #print(features.values[:])
        lower_temp, avg_temp, upper_temp = predictions(features.values[test_index], model, num_preds, alpha_low, alpha_high)

        # save the predictions in lists 
        for l in lower_temp:
            lower.append(l)
            
        for a in avg_temp:
            pred.append(a)
            
        for u in upper_temp:
            upper.append(u)
            
    lower = np.array(lower)
    upper = np.array(upper)
    #evalutae the uncetainty prediction
    report  = fun.evaluation(pred, lower[:,0], upper[:,0], test,alpha_low,alpha_high,n_features, method = 'MLP')
     #evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    return lower,pred,upper, report

def walk_forward_deep_learning_cnn(tscv, target, features, test, alpha_low, alpha_high, n_features, callback ,learning_rate, loss, optimizer, activation, dp_rate, initializer, initializer_cnn, epochs, batch_size, num_preds, method, refit:bool, validation:bool):
    """
    walk_forward_gb: walk forward validation for keras deep learning architectures, with probabilsitic forecasts. 

    Parameters
    -------
    tscv: time series split form sklearn
    target: given data-set for the target variable <- train set
    features: features for the given model < - train 
    test: test set of the data set
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: Number fo features
    callback: callback instance of keras
    learning_rate: learning rate for updating the model with new data
    loss: Loss function
    optimizer: optimizer functio
    activation: Activation function
    dp_rate: Dropout reat
    initializer: MLP initilalizer
    initizilizer_cnn: CNN initizalizer
    epochs: number of training epochs
    batch_size: Batch size
    num_preds: Number of Monte Carlo predictions
    method: Deep Learning mdoel
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed
    validation->Bool: True if validation phase, False For test phase

    Returns
    -------
    pred: average prediction results
    lower: lower qunatile predictions
    upper: upper qunatile predicitons
    report: report of the evalutation metrics on the test set
    
    """
    
    pred = [] # list for saving the average predictions
    lower = []
    upper = []
    # fit the models before the walk forward validation if refit = False
    
    # define new optimizer
    # optimizer = keras.optimizers.Adam(learning_rate)
    if refit == False:
        train_len = int(len(target) - len(test))
        # define configuration for update of learning rate
        model = CNN(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer, initializer_cnn)
        model.compile(loss = loss, optimizer = optimizer)
        # updat the model wtih new data
        # make usage of validation callback if validation ist true
        if validation == False:
            model.fit(features[:train_len],target[:train_len], batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 1) #shuffle = False)
        else:
            model.fit(features[:train_len],target[:train_len], validation_data = (features[train_len:],target[train_len:]), batch_size = batch_size,  epochs = epochs, callbacks = callback, verbose = 1)#, shuffle = False)

    #make wal forward validation
    for train_index, test_index in tscv.split(target):
        # refitting each of the models each time the training set is increased if refit = True
        if refit == True:
            model = CNN(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer, initializer_cnn)
            model.compile(loss = loss, optimizer = optimizer)
            if validation == False:
                model.fit(features[train_index], target[train_index], batch_size = batch_size,epochs = epochs, callbacks = callback, verbose = 0 )#, shuffle = False)
            else:
                model.fit(features[train_index], target[train_index], validation_data = (features[test_index], target[test_index]),batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 0)#, shuffle = False)
        # make uncertainty prediction based on the current training set
        #print(features.values[:])
        lower_temp, avg_temp, upper_temp = predictions(features[test_index], model, num_preds, alpha_low, alpha_high)

        # save the predictions in lists 
        for l in lower_temp:
            lower.append(l)
            
        for a in avg_temp:
            pred.append(a)
            
        for u in upper_temp:
            upper.append(u)
    
    lower = np.array(lower)
    upper = np.array(upper)
    #evalutae the uncetainty prediction
    report  = fun.evaluation(pred, lower[:,0], upper[:,0], test,alpha_low,alpha_high,n_features, method = 'CNN')
     #evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    return lower, pred, upper, report

def walk_forward_deep_learning_cnn2(tscv, target, features, test, alpha_low, alpha_high, n_features, callback ,learning_rate, loss, optimizer, activation, dp_rate, initializer, initializer_cnn, epochs, batch_size, num_preds, method, refit:bool, validation:bool):
    """
    walk_forward_gb: walk forward validation for keras deep learning architectures, with probabilsitic forecasts. 

    Parameters
    -------
    tscv: time series split form sklearn
    target: given data-set for the target variable <- train set
    features: features for the given model < - train 
    test: test set of the data set
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: Number fo features
    callback: callback instance of keras
    learning_rate: learning rate for updating the model with new data
    loss: Loss function
    optimizer: optimizer functio
    activation: Activation function
    dp_rate: Dropout reat
    initializer: MLP initilalizer
    initizilizer_cnn: CNN initizalizer
    epochs: number of training epochs
    batch_size: Batch size
    num_preds: Number of Monte Carlo predictions
    method: Deep Learning mdoel
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed
    validation->Bool: True if validation phase, False For test phase

    Returns
    -------
    pred: average prediction results
    lower: lower qunatile predictions
    upper: upper qunatile predicitons
    report: report of the evalutation metrics on the test set
    
    """
    
    pred = [] # list for saving the average predictions
    lower = []
    upper = []
    # fit the models before the walk forward validation if refit = False
    
    # define new optimizer
    # optimizer = keras.optimizers.Adam(learning_rate)
    if refit == False:
        train_len = int(len(target) - len(test) - int(0.05*len(target)))
        print(train_len)
        val_len = int(0.05*len(target)) + train_len
        print(val_len)
        # define configuration for update of learning rate
        model = CNN(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer, initializer_cnn)
        model.compile(loss = loss, optimizer = optimizer)
        # updat the model wtih new data
        # make usage of validation callback if validation ist true
        if validation == False:
            model.fit(features[:train_len],target[:train_len], validation_data = (features[train_len:val_len],target[train_len:val_len]), batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 1) #shuffle = False)
        else:
            model.fit(features[:train_len],target[:train_len], validation_data = (features[train_len:],target[train_len:]), batch_size = batch_size,  epochs = epochs, callbacks = callback, verbose = 1)#, shuffle = False)

    #make wal forward validation
    i = 0
    for train_index, test_index in tscv.split(target):
        # refitting each of the models each time the training set is increased if refit = True
        train_len = len(train_index) - int(len(test_index))
        print(train_len)
        val_len = int(len(test_index)) + train_len
        print(val_len)

        if refit == True:
            model = CNN(n_features, learning_rate, activation, loss, optimizer, dp_rate, initializer, initializer_cnn)
            model.compile(loss = loss, optimizer = optimizer)
            if validation == False:
                model.fit(features[:train_len], target[:train_len], validation_data = (features[train_len:val_len],target[train_len:val_len]), batch_size = batch_size,epochs = epochs, callbacks = callback, verbose = 0 )#, shuffle = False)
            else:
                model.fit(features[train_index], target[train_index], validation_data = (features[test_index], target[test_index]),batch_size = batch_size, epochs = epochs, callbacks = callback, verbose = 0)#, shuffle = False)
        # make uncertainty prediction based on the current training set
        #print(features.values[:])
        lower_temp, avg_temp, upper_temp = predictions(features[test_index], model, num_preds, alpha_low, alpha_high)
        print(avg_temp.shape)
        print(i)
        i = i+1
        # save the predictions in lists 
        for l in lower_temp:
            lower.append(l)
            
        for a in avg_temp:
            pred.append(a)
            
        for u in upper_temp:
            upper.append(u)
    
    lower = np.array(lower)
    upper = np.array(upper)
    #evalutae the uncetainty prediction
    report  = fun.evaluation(pred, lower[:,0], upper[:,0], test,alpha_low,alpha_high,n_features, method = 'CNN')
     #evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    return lower, pred, upper, report




#################
# LSTM testing
################
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def ensemble_lstm(n, n_features, loss, optimizer, dp_rate, initializer_lstm):
  ensemble = {}
  #n = 15
  for i in range(n):
    ensemble[i] = LSTM_model(n_features, loss, optimizer, dp_rate, initializer_lstm)
  return ensemble

def ensemble_fit_val(ensemble, inputs_train, labels_train, inputs_val, labels_val, batch_size, epochs, callback):
  for member in ensemble:
    #boot_label = resample(labels_train, replace= True, random_state=member)
    #boot_input = resample(inputs_train, replace= True, random_state=member)
    ensemble[member].fit(inputs_train, labels_train, validation_data = (inputs_val, labels_val), batch_size = batch_size, epochs = epochs, callbacks = callback, shuffle = True)

def ensemble_fit(ensemble, inputs_train, labels_train, batch_size, epochs, callback):
  for member in ensemble:
    ensemble[member].fit(inputs_train, labels_train, batch_size = batch_size, epochs = epochs, callbacks = callback, shuffle = True)


def ensemble_predict(n, ensemble, test):
  preds = []
  for j in range(n):
    print(j)
    pred = ensemble[j].predict(test)
    pred = pred.reshape(test.shape[0]*24,)
    #for j in pred:
    preds.append(pred)
  return preds

from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler(feature_range=(0, 1))

def walk_forward_deep_learning_lstm(target, features, test, scaler, alpha_low, alpha_high, n_features, n, prediction_step, callback ,learning_rate, loss, optimizer, dp_rate, initializer, epochs, batch_size, method, refit:bool, validation:bool):
    """
    walk_forward_gb: walk forward validation for keras deep learning architectures, with probabilsitic forecasts. 

    Parameters
    -------
    target: given data-set for the target variable <- train set
    features: features for the given model < - train 
    test: test set of the data set
    scaler: Scaler function for normilization
    alpha_low: lower percentile
    alpha_high: upper percentile
    n_features: Number fo features
    n: >1 for LSTM ensemble
    callback: callback instance of keras
    learning_rate: learning rate for updating the model with new data
    loss: Loss function
    optimizer: optimizer functio
    activation: Activation function
    dp_rate: Dropout reat
    initializer: MLP initilalizer
    initizilizer_cnn: CNN initizalizer
    epochs: number of training epochs
    batch_size: Batch size
    num_preds: Number of Monte Carlo predictions
    method: Deep Learning mdoel
    refit->Bool: if trure refit model, if flase dont refit model after each prediction horizon is completed
    validation->Bool: True if validation phase, False For test phase

    Returns
    -------
    pred: average prediction results
    lower: lower qunatile predictions
    upper: upper qunatile predicitons
    report: report of the evalutation metrics on the test set
    
    """
    
    pred = [] # list for saving the average predictions
    lower = []
    upper = []
    # fit the models before the walk forward validation if refit = False
    train_len = len(target) - len(test)
    whole_len =  len(target)
    # define new optimizer
    # optimizer = keras.optimizers.Adam(learning_rate)
    if refit == False:
        print('no refit')
        # define configuration for update of learning rate
        ensemble = ensemble_lstm(n, n_features, loss, optimizer, dp_rate, initializer)
        print(ensemble)
        # updat the model wtih new data
        # make usage of validation callback if validation ist true
        if validation == False:
          ensemble_fit(ensemble, features[:train_len], target[:train_len], batch_size, epochs, callback)
        else:
          print('validation time')
          ensemble_fit_val(ensemble, features[:train_len], target[:train_len], features[train_len:whole_len], target[train_len:whole_len], batch_size, epochs, callback)

    #make wal forward validation
    i = 0
    for j in range(train_len, whole_len, prediction_step):
        # refitting each of the models each time the training set is increased if refit = True
        if refit == True:
            ensemble = ensemble_lstm(n, n_features, loss, optimizer, dp_rate, initializer)
            if validation == False:
              ensemble_fit(ensemble, features[:j], target[:j], batch_size, epochs, callback)
            else:
              ensemble_fit_val(ensemble, features[:j], target[:j], features[j:j+prediction_step], target[j:j+prediction_step], batch_size, epochs, callback)
        # make uncertainty prediction based on the current training set
        #print(features.values[:])
        #preds  = ensemble_predict(n, ensemble, features[j:j+prediction_step])
        lower_temp, avg_temp, upper_temp = predictions(features[j:j+prediction_step], ensemble[0], 75, alpha_low, alpha_high)
        #print(i)
        #i = i+1
        # mae point and prediction interval prediction
        #avg_temp = np.mean(preds, axis = 0)
        #lower_temp = np.quantile(preds, alpha_low, axis = 0)
        #upper_temp = np.quantile (preds, alpha_high, axis = 0)


        # save the predictions in lists 
        for l in lower_temp:
            lower.append(l)
            
        for a in avg_temp:
            pred.append(a)
            
        for u in upper_temp:
            upper.append(u)
    
    pred = np.array(pred)
    lower = np.array(lower)
    upper = np.array(upper)

    #pred = scaler.inverse_transform([pred])
    #lower = scaler.inverse_transform([lower])
    #upper = scaler.inverse_transform([upper])

    pred = pred.reshape(test.shape[0]*24,)
    lower = lower.reshape(test.shape[0]*24,)
    upper = upper.reshape(test.shape[0]*24,)
    test = test.reshape(test.shape[0]*24,)

    pred = scaler.inverse_transform([pred])
    lower = scaler.inverse_transform([lower])
    upper = scaler.inverse_transform([upper])

    test = scaler.inverse_transform([np.array(test).reshape(len(test),)])
    test = test.T
    pred = pred.T
    lower = lower.T
    upper = upper.T

    print(pred.shape)
    #lower = np.array(lower)
    #upper = np.array(upper)
    #evalutae the uncetainty prediction
    report  = fun.evaluation_lstm(pred, lower, upper, test,alpha_low,alpha_high,n_features, method = 'LSTM')
     #evaluation(pred, lower, upper, test,alpha_low,alpha_high,n_features, method):
    return lower, pred, upper, report




#####################
# Plotting
#######################

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

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss', color = colours.get('1B'))
  plt.plot(history.history['val_loss'], label='val_loss',  color = colours.get('gold'))
  #plt.ylim([0, 10])
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend(loc = 'best')
  plt.title('Train and validation loss')
  plt.grid(True)