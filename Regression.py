#This file contains the regression model (artificial neural networks) that predicts car ownership per LSOA
#Input is the dataset file for the change in cars between two points in time

#import dependencies
import pickle
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
from scipy.stats import gaussian_kde
import geopandas as gpd
import time


import tensorflow as tf
import keras
from keras import layers
from keras import models
from keras import utils
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Activation
from keras.regularizers import l2
from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras import datasets

from keras.callbacks import LearningRateScheduler
from keras.callbacks import History

from keras import losses
from sklearn.utils import shuffle

#_build model function
def build_model(NN_config, input_dim, weighting_estimator, learning_rate):

    model = Sequential()

    for i in range(len(NN_config)):

        if i == 0:
            # after input layer
            model.add(
                Dense(NN_config[i], input_shape=[input_dim], kernel_initializer=weighting_estimator, activation='relu'))
        else:
            model.add(Dense(NN_config[i], kernel_initializer=weighting_estimator, activation='relu'))

    # output layer
    model.add(Dense(1, kernel_initializer='normal'))

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['mae', 'mse'])
    return model

#fit_model function
def fit_model(NN_model, X, y, batch_size, EPOCHS):
    # early_stop function stops training if error doesn't change much
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

    history = NN_model.fit(X, y,
                        batch_size=batch_size,
                        epochs=EPOCHS,
                        validation_split=0.2,
                        verbose=0,
                        shuffle=True,
                        callbacks=[early_stop])

    return history

#normalise function
def norm(dataframe, training_stats):
    return (dataframe - training_stats['mean']) / training_stats['std']

#regression function
def regression(df):

    # return independent variables
    cols = df.columns.tolist()
    x = [vj for vj in [vi for vi in [v for v in cols if v != 'TotalCars'] if vi != 'GEO_CODE'] if vj != 'geometry']

    for xval in x:
        df[xval] = pd.to_numeric(df[xval])

    # test/train split; produce train_stats summary
    df_train = df.sample(frac=0.8, random_state=0)  # all data is 'training data' - validation is done within fit_model
    df_test = df.drop(df_train.index)
    train_stats = df_train[x].describe().transpose()

    #return X_train and X_test --> training/testing input data
    X_train = norm(df_train[x], train_stats)
    X_test = norm(df_test[x], train_stats)

    #return dependent variables
    y_train = df_train['TotalCars']
    y_test = df_test['TotalCars']

    #X_all and y_all - concatenate train & test datasets
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])

    #hard-coded in NN specs
    NN_config = [40, 40]
    input_dim = len(df_train[x].keys())
    weighting_estimator = 'normal'
    learning_rate = 0.001
    batch_size = int(len(X_train) / 100)
    EPOCHS = 10000

    #build model
    model = build_model(NN_config, input_dim, weighting_estimator, learning_rate)

    #fit model
    history = fit_model(model, X_all, y_all, batch_size, EPOCHS)

    #record config
    config = {'NN config': NN_config,
              'Weighting estimator': weighting_estimator,
              'Learning rate': learning_rate,
              'Batch size': batch_size}

    y_pred = model.predict(X_all)
    y_predlist = []
    for y in y_pred:
        y_predlist.append(np.round(y[0]))

    results_df = pd.DataFrame({'Predicted': y_predlist, 'Actual': y_all})

    #merge this with the input dataset, based on index
    for i in list(df.index.values):
        df.at[i, 'Predicted'] = results_df.Predicted[i]

    #report error
    print('MAE, all dataset')
    print('Mean Absolute Error:', metrics.mean_absolute_error(y_all, y_predlist))

    #return dataset with the predicted column
    return df

#Save model config
