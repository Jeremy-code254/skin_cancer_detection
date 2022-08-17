#!/usr/bin/env pythonC:\Users\hp\\
# coding: utf-8

# In[214]:

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
import seaborn as sns
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np
import os
#%%
import keras
from keras.utils.np_utils import to_categorical
from keras import Sequential
from keras.layers import Dense,Dropout, Flatten, Conv2D, MaxPool2D
from sklearn.model_selection import train_test_split
from tensorflow import lite
#%%

read = pd.read_csv('Desktop/HM10000/HAM10000_metadata.csv')


# In[196]:


le = LabelEncoder()
le.fit(read["dx"])
LabelEncoder()
le.classes_


# In[197]:
read['label'] = le.transform(read["dx"])
read.sample()
# In[199]:

r0 =read[read['label'] == 0]
r1 =read[read['label'] == 1]
r2 =read[read['label'] == 2]
r3 =read[read['label'] == 3]
r4 =read[read['label'] == 4]
r5 =read[read['label'] == 5]
r6 =read[read['label'] == 6]
samples = 500


# In[200]:


r0_balanced = resample(r0, replace = True, n_samples = samples, random_state = 42)
r1_balanced = resample(r1, replace = True, n_samples = samples, random_state = 42)
r2_balanced = resample(r2, replace = True, n_samples = samples, random_state = 42)
r3_balanced = resample(r3, replace = True, n_samples = samples, random_state = 42)
r4_balanced = resample(r4, replace = True, n_samples = samples, random_state = 42)
r5_balanced = resample(r5, replace = True, n_samples = samples, random_state = 42)
r6_balanced = resample(r6, replace = True, n_samples = samples, random_state = 42)


# In[201]:


balanced_data = pd.concat([r0_balanced,r1_balanced,r2_balanced,
                           r3_balanced,r4_balanced,r5_balanced,
                          r6_balanced])


# In[202]:


balanced_data['label'].value_counts()

# In[204]:


img_path = {os.path.splitext(os.path.basename(x))[0]: x
                            for x in glob(os.path.join('Desktop/HM10000/', '*', '*.JPG'))
                            }
                            
# In[209]:

balanced_data['path'] = read['image_id'].map(img_path.get)

# In[ ]:
    
balanced_data['image'] = balanced_data['path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))
# In[ ]:
X = np.asarray(balanced_data['image'].tolist())
X  = X/255
y = balanced_data['label']
y_cat = to_categorical(y, num_classes = 7)
np.random.seed(42)
X_train, X_test, y_train, y_test = train_test_split( X, y_cat, test_size=0.25, random_state=42)

#%%
num_classes = 7
model = Sequential()
model.add(Conv2D(256,(3,3), activation ="relu", input_shape=(32,32,3)))

model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(128,(3,3), activation ="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))

model.add(Conv2D(64,(3,3), activation ="relu"))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.3))
model.add(Flatten())

model.add(Dense(32))
model.add(Dense(7,activation  = 'softmax'))
model.summary()
#%%

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

#%%

model.fit( X_train, y_train, epochs=50,  batch_size = 16,
                    validation_data = (X_test, y_test) , verbose = 2)
#%%
# score= model.evaluate(X_test, y_test)
# print('test accuracy: ', score[1])
#%%
y_predict= model.predict(X_test[[400]])
y_predict_classes = np.argmax(y_predict, axis = 1)
y_true= np.argmax(y_test, axis=1)
#%%

import cv2
picture = cv2.imread('Desktop/im1.jpg')
picture = picture/255
input_image = np.asarray(picture)
input_image.resize(1,32,32,3,refcheck=False)
pred = model.predict(input_image)
prediction = np.argmax(pred, axis = 1)

labels = ["Actinic keratosis","Basa cell carcinoma","Benign keratosis-like lesions","Dermatofibroma","Melanoma","Melanocity nevi","Vascular"]

labels = np.asarray(labels)

labels[prediction]
        
 #%%
# skin_care = 'desktop/skin_care3.0.h5'
# keras.models.save_model(model,skin_care)
# converter=lite.TFLiteConverter.from_keras_model(model)
# tfmodel = converter.convert()
# open("skin_care2.0.tflite","wb").write(tfmodel)
 





























