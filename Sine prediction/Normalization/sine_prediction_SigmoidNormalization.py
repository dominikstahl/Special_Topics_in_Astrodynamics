# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:26:23 2020

@author: salva
"""
# Sigmoid normalization
# In[194]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[195]:


import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense

print(tf.__version__)


# ## Generate a dataset for training & validation

# Generate your data
# * Examine sine waves with a few different amplitudes and frequencies to assess peformance and robustness.
# * Start with `sin(x)`.

# In[196]:


data = []
for x in np.arange(0,200):
    y_pi = np.sin(float(x-50)*(2*np.pi)/100)
    y = np.sin(float(x)*(2*np.pi)/100)
    data.append([x,y_pi,y])
data = np.array(data)
data2 = data


# Create a `pandas` dataframe for it

# In[197]:

pdata = pd.DataFrame({'x':data[:,0],'y_pi':data[:,1],'y':data[:,2]})

# In[198]:

plt.figure(1)
plt.scatter(pdata['x'],pdata['y'])


# Split the dataset into training and testing sets

# In[199]:


train_dataset = pdata.sample(frac=0.8, random_state=0)
test_dataset = pdata.drop(train_dataset.index)


# Split features from labels

# In[200]:


train_labels = train_dataset.pop('y')
test_labels = test_dataset.pop('y')


# In[201]:


train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats


# In[202]:
# Sigmoid normalization
def norm(x):
    result = 1 / (1 + np.exp(x))
    return (result)

normed_train_dataset = norm(train_dataset)
normed_test_dataset = norm(test_dataset)

# In[203]:

timeSteps = 1
batchSize = 32
n_features = 2

# ##### Create a model

# In[204]:

model = Sequential()
model.add(LSTM(32, activation='selu')),
model.add(Dense(1, activation='selu'))
model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
# In[206]:


# reshape from [samples, features] into [samples, features, timesteps]
normed_train_x_reordered = np.array(normed_train_dataset).reshape((normed_train_dataset.shape[0], normed_train_dataset.shape[1], timeSteps))
normed_test_x_reordered = np.array(normed_test_dataset).reshape((normed_test_dataset.shape[0], normed_test_dataset.shape[1], timeSteps))
print("normed_train_x_reordered.shape = {}".format(normed_train_x_reordered.shape))
print("normed_test_x_reordered.shape = {}".format(normed_test_x_reordered.shape))


# In[207]:


train_labels_y = np.array(train_labels[0:len(normed_train_x_reordered)])
test_labels_y = np.array(test_labels[0:len(normed_test_x_reordered)])


# #### Train the model

# In[208]:


history = model.fit(normed_train_x_reordered, train_labels_y, validation_split=0.2, epochs=100)


# Visualize the model's training progress via the history object

# In[209]:


hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()


# ## Basic regression: predict future values of the sine function

# #### Predict with the model

# If you trained on `n` points (eg. `n = 49`), now predict the value of point `n+1`

# Once you have predicted a single future point, develop a 'rolling' predictor which drops the first point and predicts point `n+2` based on the set of points `[2,n+1]`.

# Examine the quality of the results for the mechanism to predict points into the future.

# ____

# #### Plot the history

# In[210]:

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim([-0.1,0.7])
plt.legend(['loss','val_loss'],loc='upper right')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
plt.show()


# Are `10` epochs too few? If so, try training for more epochs.

# ### Make predictions
# 
# Finally, predict values using data in the test set:

# In[211]:


test_predictions = model.predict(normed_test_x_reordered).flatten()

plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels_y, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
_ = plt.plot([-2,2], [-2,2])
plt.show()


# Take a look at the error distribution.


# In[213]:

plt.figure()
plt.scatter(np.array(test_dataset)[:,0], test_predictions)
plt.title('Case activation function SELU', fontsize=14, fontweight='bold')
plt.xlabel('epoch')
plt.ylabel('sin(epoch)')


# In[214]:
#Statistics

plt.figure()
error = test_predictions - test_labels_y
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.title("Sigmoid normalization results")
 
rms = np.sqrt(np.mean(error**2))
print("Root-mean-square prediction error Sigmoid = {}".format(rms))

