import numpy as np
import pandas as pd
import os
import warnings
import torch
warnings.filterwarnings("ignore")
from keras.layers import Dense, Activation
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import BatchNormalization

from keras import backend as K
K.set_floatx('float32')
import json


# ANN structure
N_neurons=20
N_layers=1

model = Sequential()
if N_layers==1:
     model.add(Dense(N_neurons, activation = 'sigmoid', input_dim = 4))
elif N_layers==2 :
    model.add(Dense(N_neurons, activation = 'sigmoid', input_dim = 4))
    model.add(Dense(units = N_neurons, activation = 'relu'))
elif N_layers==3 :
    model.add(Dense(N_neurons, activation = 'sigmoid', input_dim = 4))
    model.add(Dense(units = N_neurons, activation = 'relu'))
    model.add(Dense(units = N_neurons, activation = 'relu'))
else:  
    print("Check no of layers")
model.add(Dense(units = 4))
model.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics=['mae'])


# Load the model
with open('fischer_ann_weights.json', 'r') as f:
    weights_list = json.load(f)

weights = [np.array(w) for w in weights_list]
model.set_weights(weights)

# Test the model
df = pd.read_csv("fischer_data_processed.csv",sep=",")
X = df.iloc[:,0:4].astype('float32')
y = df.iloc[:,4:8].astype('float32')


def predict_fischer_model(X):
    y_pred = model.predict(X)
    y_pred=torch.Tensor(y_pred)
    return y_pred
