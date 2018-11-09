# -*- coding: utf-8 -*-
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
import random
import pandas as pd
import csv
import time
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

def MAPE(y_true, y_pred):
    errors = 0
    for i in range(len(y_true)):
        errors = errors + (abs((float(y_true[i]) - float(y_pred[i]))) / float(y_true[i]))
    return errors / len(y_pred)

def modelNN(n,shape,epochs,learning_rate):
    cf = 'custom_activation'
    model = Sequential()
    get_custom_objects().update({cf: Activation(custom_activation)})
    model.add(Dense(units = n,input_shape = shape ,activation = cf, kernel_initializer='normal'))
    model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
    model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
    model.add(Dense(units = n, activation = cf, kernel_initializer = 'normal'))
    model.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'normal'))
    decay_rate = learning_rate/epochs;
    adam = Adam(lr = learning_rate,decay = decay_rate)
    model.compile(loss='mean_squared_error', optimizer=adam)
    return model


def custom_activation(x):
    global alpha
    return 1 / (1 + K.exp(-alpha * x))

# Loading Data
df = pd.read_csv("pems.csv", header=0)

minMapeMPL1 = 100.0

# Indexing the data
df['date'] = pd.to_datetime(df['date'])
df.index = df['date']
del df['date']

# Variaveis Fixadas
MAX_Q = 5
p = 13
q = 4
epochsN = 1 # modificar talvez

# Variaveis a ser otimizadas
n = 100
learning_rate = 0.005
alpha = 1

with open('NN_PEMS.csv', 'w', 1) as nn_file:
    # Reading CSV
    nnwriter  = csv.writer(nn_file, delimiter=',', quoting=csv.QUOTE_MINIMAL)

    # Writing results headers
    nnwriter.writerow(['P', 'Q', 'N', 'H', 'Avg Mape', 'Min MAPE', 'Avg time'])

    # Using historic data (Q) from the same time and weekday
    for i in range (1, MAX_Q + 1):
        df['count-{}'.format(i)] = df['count'].shift(i * 7 * 24)

    # Change n/a to 1
    df = df.fillna(0)

    # Normalizing the data
    df_max = max(df['count'])
    df_min = min(df['count'])
    df['count'] = df['count'] / (df_max - df_min)
    for i in range (1, MAX_Q + 1):
        df['count-{}'.format(i)] = df['count-{}'.format(i)] / (df_max - df_min)

    aux_df = df

    # Shifiting the data set by Q weeks
    df = df[q * 7 * 24:]

    print('Running for params P = {}, Q = {}, N = {}'.format(p, q, n))
    print('Pre-processing...')

        # Initializing the data
    X1 = list()
    Y1 = list()

    # Mapping each set of variables (P and Q) to their correspondent value
    for i in range(len(df) - p - 1):
        X = list()
        for j in range (1, q + 1):
            X.append(df['count-{}'.format(j)][i + p + 1])

        X1.append(X + list(df['count'][i:(i + p)]))
        Y1.append(df['count'][i + p + 1])

    print('   Splitting in train-test...')
        # Train/test/validation split
    rows1 = random.sample(range(len(X1)), int(len(X1)//3))
    
    X1_test = np.array( [X1[j] for j in rows1] )
    Y1_test = np.array( [Y1[j] for j in rows1] )
    X1_train = np.array( [X1[j] for j in list(set(range(len(X1))) - set(rows1))] )
    Y1_train = np.array([Y1[j] for j in list(set(range(len(Y1))) - set(rows1))] )
    
    print('   Initializing the models...')
    results_nn1 = list()
    avg_mlp_time1 = 0
    # Initializing the model
    shape = X1_train.shape[1:]
    MLP1 = modelNN(n,shape,epochsN,learning_rate)

    print('Running tests...')
    for test in range(0, 30):
        if(test % 6 == 5):
            print('T = {}%'.format(int(((test + 1)*100)/30)))

        start_time = time.time()
        
        MLP1.fit(X1_train, Y1_train, epochs = epochsN)
        predicted1_nn = MLP1.predict(X1_test)
        avg_mlp_time1 = avg_mlp_time1 + time.time() - start_time
        results_nn1.append(MAPE(Y1_test, predicted1_nn))
        print(test)
        if(minMapeMPL1 > MAPE(Y1_test, predicted1_nn)):
            trueValue = pd.DataFrame(Y1_test)
            bestMLP1value = pd.DataFrame(predicted1_nn)
            minMapeMPL = MAPE(Y1_test, predicted1_nn)
            trueValue.to_csv("TrueValue.csv")
            bestMLP1value.to_csv("BestMLP1value.csv")


    nnwriter.writerow([p, q, n, 1, np.mean(results_nn1), min(results_nn1), avg_mlp_time1 / 30])
    print('- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -')
df = aux_df
