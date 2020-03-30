#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


#Loading the data from csv
x_train = pd.read_csv('X_train_update.csv', ',') 
y_train = pd.read_csv('Y_train_CVw08PX.csv', ',')
x_test = pd.read_csv('X_test_update.csv', ',')


# In[3]:


#Dropping the unwanted columns
x_train = x_train.drop(['description'], axis = 1)
x_test = x_test.drop(['description'], axis = 1)


# In[9]:


#Making changes to the id to map the id to name of the image in the train data
x_train['Unnamed: 0'] = 'image_' + x_train['imageid'].map(str) + '_product_' + x_train['productid'].map(str) + '.jpg'


# In[5]:


#Making changes to the id to map the id to name of the image in the train data
y_train['Unnamed: 0'] = 'image_' + x_train['imageid'].map(str) + '_product_' + x_train['productid'].map(str) + '.jpg'


# In[6]:


y_train


# In[36]:


#merging to a single dataframe
train = pd.merge(x_train, y_train, on='Unnamed: 0', how='outer')
train['prdtypecode'] = train['prdtypecode'].apply(str)


# In[37]:


train


# In[51]:


#list of the labels
labels = list(y_train.prdtypecode.unique())
labels.sort()
print(labels)
print(len(set(labels)))


# In[52]:


#distribution of labels across the train data
plt.figure(figsize=(14,6))
y_train.prdtypecode.value_counts().plot(kind='bar')
plt.show()


# In[13]:



from keras_preprocessing.image import ImageDataGenerator

# create a data generator
datagen=ImageDataGenerator(rescale=1./255.)


# In[38]:


# load and iterate training dataset
train_generator = datagen.flow_from_dataframe(
dataframe = train,
directory = "/Users/prannoynoel/Documents/DS/Ensemble /project/images/image_train",
x_col = 'Unnamed: 0',
y_col = 'prdtypecode',
batch_size=32,
seed=42,
class_mode='categorical', 
shuffle=True,
target_size=(32,32))


# In[17]:


train_generator


# In[18]:


#Making changes to the id to map the id to name of the image in the test data
x_test['Unnamed: 0'] = 'image_' + x_test['imageid'].map(str) + '_product_' + x_test['productid'].map(str) + '.jpg'


# In[20]:


test_datagen=ImageDataGenerator(rescale=1./255.)

# load and iterate test dataset
test_generator=test_datagen.flow_from_dataframe(
dataframe=x_test,
directory="/Users/prannoynoel/Documents/DS/Ensemble /project/images/image_test",
x_col="Unnamed: 0",
y_col=None,
batch_size=32,
seed=42,
shuffle=False,
class_mode=None,
target_size=(32,32))


# In[25]:


from keras.utils import to_categorical
train_labels = to_categorical(labels)


# In[26]:


print(train_labels.shape)


# In[31]:


from keras.layers import Reshape, Flatten, Dense, Dropout
from keras.layers.embeddings import Embedding
import keras
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.models import Sequential,Input,Model
from keras.layers import Conv2D, MaxPooling2D

#creating a neural network
model = Sequential()
model.add(Conv2D(input_shape=(32,32,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Flatten())
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=4096,activation="relu"))
model.add(Dense(units=27, activation="softmax"))


# In[32]:


model.summary()


# In[33]:


from keras.optimizers import SGD, Adam
# sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
adam = Adam(lr=0.0001)
model.compile(optimizer= adam, loss='categorical_crossentropy', metrics=['accuracy'])


# In[41]:


# fit model
model.fit_generator(train_generator,
                   epochs = 1)


# In[42]:


#reset the test_generator  make a prediction
test_generator.reset()
pred = model.predict_generator(test_generator,
                                verbose=1)


# In[43]:


predicted_class_indices=np.argmax(pred,axis=1)


# In[44]:


#map the predicted labels with their unique ids
labels = (train_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in predicted_class_indices]


# In[49]:


#labels predicted
k = np.array(predictions)
np.unique(k)


# In[45]:


#importing into a csv file
filenames=test_generator.filenames
results=pd.DataFrame({"Filename":filenames,
                      "Predictions":predictions})
results.to_csv("results.csv",index=False)


# In[ ]:




