#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np


# In[2]:


import tensorflow_datasets as tfds


# In[3]:


mnist_data,mnist_info = tfds.load(name='mnist',with_info=True,as_supervised=True)


# In[4]:


mnist_train,mnist_test = mnist_data['train'],mnist_data['test']


# In[5]:


mnist_info


# In[6]:


num_validation_sample =  0.1 * mnist_info.splits['train'].num_examples
num_validation_sample =  tf.cast(num_validation_sample,tf.int64)
num_test_samples      =  mnist_info.splits['test'].num_examples
num_test_samples      =  tf.cast(num_test_samples,tf.int64)


# In[7]:


def scale(image,label):
    image = tf.cast(image,tf.float32)
    image /= 255.
    return image,label


# In[8]:


scaled_train_validate_data = mnist_train.map(scale)
test_data           = mnist_test.map(scale)


# In[9]:


BUFFER_SIZE = 10000

shuffled_train_validate_data = scaled_train_validate_data.shuffle(BUFFER_SIZE)
validation_data              = shuffled_train_validate_data.take(num_validation_sample)
train_data                   = shuffled_train_validate_data.skip(num_validation_sample)


# In[10]:


BATCH_SIZE = 100
train_data                 = train_data.batch(BATCH_SIZE)
validation_data            = validation_data.batch(num_validation_sample)
test_data                  = test_data.batch(num_test_samples)


# In[11]:


validation_inputs, validation_targets = next(iter(validation_data))


# In[12]:


input_size        = 784
output_size       = 10
hidden_layer_size = 200
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(hidden_layer_size,activation='relu'),
    tf.keras.layers.Dense(hidden_layer_size,activation='tanh'),
    tf.keras.layers.Dense(10,activation='softmax')
])


# In[13]:


opt = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=opt,loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[14]:


#TRAINGING
NUM_EPOCHS = 5
model.fit(train_data,epochs=NUM_EPOCHS,validation_data=(validation_inputs,validation_targets),verbose=2)


# In[15]:


test_loss,test_accuracy = model.evaluate(test_data)


# In[17]:


print('Test loss={0:.2f}, Test Accuracy ={1:.2f}%'.format(test_loss,test_accuracy*100.))


# In[ ]:




