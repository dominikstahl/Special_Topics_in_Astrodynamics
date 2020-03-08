#!/usr/bin/env python
# coding: utf-8

# 
# # Special Topics in Astrodynamics | ae4889
# 
# ## Time-series prediction on a simple function

# #### Import the required packages

# In[319]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[320]:


import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)


# ## Generate a dataset for training & validation

# Generate your data
# * Examine sine waves with a few different amplitudes and frequencies to assess peformance and robustness.
# * Start with `sin(x)`.

# In[321]:


data = []
for x in np.arange(0,200):
    #y_10p = np.sin(float(x-10)*(2*np.pi)/100)
    #y_11p = np.sin(float(x-11)*(2*np.pi)/100)
    #y_12p = np.sin(float(x-12)*(2*np.pi)/100)
    #y_13p = np.sin(float(x-13)*(2*np.pi)/100)
    #y_14p = np.sin(float(x-14)*(2*np.pi)/100)
    #first = 2*y_10p+y_11p-y_13p-2*y_14p
    #second = 4*y_10p+y_11p+y_13p+4*y_14p
    #third = 9*y_10p+y_11p-y_13p-9*y_14p
    y_pi = np.sin(float(x-50)*(2*np.pi)/100)
    y = np.sin(float(x)*(2*np.pi)/100)
    data.append([x,y_pi,y])
data = np.array(data)
data2 = data
print(data)


# Create a `pandas` dataframe for it

# In[322]:


pdata = pd.DataFrame({'x':data[:,0],'y_pi':data[:,1],'y':data[:,2]})
#pdata # This displays the DataFrame if uncommented.


# Visual check of the dataset

# In[323]:


plt.scatter(pdata['x'],pdata['y'])


# Split the dataset into training and testing sets

# In[324]:


train_dataset = pdata.sample(frac=0.8, random_state=0)
test_dataset = pdata.drop(train_dataset.index)


# Split features from labels

# In[325]:


train_labels = train_dataset.pop('y')
test_labels = test_dataset.pop('y')


# In[326]:


train_stats = train_dataset.describe()
train_stats = train_stats.transpose()
train_stats


# In[327]:


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_dataset = norm(train_dataset)
normed_test_dataset = norm(test_dataset)


# ##### Create a model

# In[328]:


def build_model():
    model = keras.Sequential([
        layers.Dense(40, activation='selu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(40, activation='selu'),
        layers.Dense(1)
    ])

    model.compile(loss='mse',
                optimizer='adam',
                metrics=['accuracy'])
    return model


# In[329]:


model = build_model()


# In[330]:


model.summary()


# #### Train the model

# In[331]:


history = model.fit(normed_train_dataset, train_labels, validation_split=0.2, epochs=50)


# Visualize the model's training progress via the history object

# In[332]:

plt.figure()
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
hist.tail()
plt.show()

# ## Basic regression: predict future values of the sine function

# #### Predict with the model

# If you trained on `n` points (eg. `n = 49`), now predict the value of point `n+1`

# Once you have predicted a single future point, develop a 'rolling' predictor which drops the first point and predicts point `n+2` based on the set of points `[2,n+1]`.

# Examine the quality of the results for the mechanism to predict points into the future.

# ____

# #### Plot the history

# In[333]:

plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss','val_loss'],loc='upper right')
plt.show()


# Are `10` epochs too few? If so, try training for more epochs.

# ### Make predictions
# 
# Finally, predict values using data in the test set:

# In[334]:


print(normed_test_dataset)
test_predictions = model.predict(normed_test_dataset).flatten()

plt.figure()
a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.xlim([-1.5,1.5])
plt.ylim([-1.5,1.5])
_ = plt.plot([-2,2], [-2,2])
plt.show()

# Take a look at the error distribution.

# In[335]:

plt.figure()
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")
plt.show()

#Calculate the error obtained with Dense layers
rms = np.sqrt(np.mean(error**2))
print("Root-mean-square prediction error  = {}".format(rms))

# In[336]:


test_dataset
plt.scatter(np.array(test_dataset)[:,0], test_predictions)


# 

# In[337]:


for i in range (1,10):
    data = data[10:]
    print(data)
    for j in range (1,10):
        x = data[-1,0]+1.0
        y_pi = data[-50,2]
        no = norm([x,y_pi])
        input_for_new_prediction = pd.DataFrame({'x':[no[0]],'y_pi':[no[1]]})
        data = np.append(data,[[x,y_pi,model.predict(input_for_new_prediction)[0,0]]],axis=0)
        data2 = np.append(data2,[[x,y_pi,model.predict(input_for_new_prediction)[0,0]]],axis=0)
    print(data)

    #plt.scatter(pdata['x'],pdata['y'])

    train_dataset = pdata.sample(frac=0.8, random_state=0)
    test_dataset = pdata.drop(train_dataset.index)

    train_labels = train_dataset.pop('y')
    test_labels = test_dataset.pop('y')

    train_stats = train_dataset.describe()
    train_stats = train_stats.transpose()
    train_stats

    def norm(x):
      return (x - train_stats['mean']) / train_stats['std']

    normed_train_dataset = norm(train_dataset)
    normed_test_dataset = norm(test_dataset)

    def build_model():
        model = keras.Sequential([
            layers.Dense(40, activation='selu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(40, activation='selu'),
            layers.Dense(1)
        ])

        model.compile(loss='mse',
                    optimizer='adam',
                    metrics=['accuracy'])
        return model

    model = build_model()
    model.summary()
    history = model.fit(normed_train_dataset, train_labels, validation_split=0.2, epochs=50)


# In[340]:

plt.figure()
pdata = pd.DataFrame({'x':data2[:,0],'y':data2[:,2]})
plt.scatter(pdata['x'],pdata['y'])
plt.title("Prediction of 100 points of the sine-wave series using Dense layers")
plt.show()
# ____

# ## Questions to consider
# 
# * How well does your ANN predict beyond the range of data it was trained on?
# 
# See the Phase 1 assignment description for additional guidance.

# ____

# ## Copyright & license details

# In[339]:


print(data)

