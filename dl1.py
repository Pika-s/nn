#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pandas import read_csv


# In[ ]:


import keras
from keras.models import Sequential #To create sequential neural network
from keras.layers import Dense #To create hidden layers


# In[ ]:


from keras import Sequential
from keras import Dense


# In[ ]:


from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


# In[ ]:


dataframe = read_csv("housing.csv", delim_whitespace=True, header=None)
dataset = dataframe.values


# In[ ]:


X = dataset[:,0:13]
Y = dataset[:,13]


# In[ ]:


def baseline_model():
    model = Sequential()
    model.add(Dense(13, input_shape=(13,), kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


estimator = KerasRegressor(model=baseline_model, epochs=100, batch_size=5, verbose=0)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, Y, cv=kfold, scoring='neg_mean_squared_error')
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))

