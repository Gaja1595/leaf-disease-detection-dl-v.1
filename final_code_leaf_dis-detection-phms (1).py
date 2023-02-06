#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import keras
from keras.preprocessing.image import ImageDataGenerator, img_to_array , load_img
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19,preprocess_input,decode_predictions
import tensorflow as tf


# In[74]:


from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)


# In[75]:


len(os.listdir("C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage"))


# In[76]:


train_datagen=ImageDataGenerator(zoom_range=0.5 ,shear_range=0.3,horizontal_flip=True,preprocessing_function=preprocess_input)
val_datagen=ImageDataGenerator(preprocessing_function=preprocess_input)


# In[77]:


train=train_datagen.flow_from_directory(directory="/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage",
                                        target_size=(256,256),
                                        batch_size=32)
val=val_datagen.flow_from_directory(directory="/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage",
                                        target_size=(256,256),
                                        batch_size=32)


# In[78]:


t_img,label=train.next()


# In[79]:


t_img.shape


# In[80]:


def plotImage(img_arr,label):
    for im ,l in zip(img_arr,label):
        plt.figure(figsize=(5,5))
        plt.imshow(im/256)
        plt.show()


# In[81]:


plotImage(t_img[:10],label[:10])


# In[82]:


from keras.layers import Dense,Flatten
from keras.models import Model
from keras.applications.vgg19 import VGG19
import keras


# In[83]:


base_model=VGG19(input_shape=(256,256,3),include_top=False)


# In[84]:


for layer in base_model.layers:
    layer.trainable=False


# In[85]:


base_model.summary()


# In[86]:


X=Flatten()(base_model.output)
X=Dense(units=15,activation='softmax')(X)
model=Model(base_model.input,X)


# In[87]:


model.summary()


# In[88]:


model.compile(optimizer='adam', loss=keras.losses.categorical_crossentropy,metrics=['accuracy'])


# In[89]:


his=model.fit_generator(train,steps_per_epoch=16,
              epochs=50,
            verbose=1,
              
                shuffle=True)
        


# In[90]:


h=his.history
h.keys()


# In[91]:


print(h)


# In[92]:


for i in h:
    print(i)
    for j in h[i]:
        print(j)


# In[93]:


plt.plot(h['loss'])
plt.plot(h['accuracy'],c="red")
plt.title("loss vs accuracy")
plt.show()


# In[94]:


ref=dict(zip(list(train.class_indices.values()),list(train.class_indices.keys())))
print(ref)


# In[95]:


train.class_indices.values()


# In[96]:


train.class_indices.keys()


# In[97]:


def prediction(path):
    img=load_img(path,target_size=(256,256))
    i=img_to_array(img)
    im=preprocess_input(i)
    img= np.expand_dims(im , axis=0)
 
    
    print(im.shape)
    pred= np.argmax(model.predict(img))
    print(f"the image belongs to {ref[pred]}")
   


# In[98]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato__Tomato_YellowLeaf__Curl_Virus/00a538f3-8421-43ab-9e6f-758d36180dd3___YLCV_NREC 2667.JPG"

prediction(path)


# In[99]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Septoria_leaf_spot/00f16858-f392-4d9e-ad9f-efab8049a13f___JR_Sept.L.S 8368.JPG"

prediction(path)


# In[100]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Potato___Late_blight/1d05837e-11e4-40a4-8bd5-cfe5ad67365a___RS_LB 5428.JPG"

prediction(path)


# In[101]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Leaf_Mold/9ced7d05-026f-45ab-986e-6af8eb2a13d6___Crnl_L.Mold 8995.JPG"

prediction(path)


# In[102]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Septoria_leaf_spot/00f16858-f392-4d9e-ad9f-efab8049a13f___JR_Sept.L.S 8368.JPG"

prediction(path)


# In[103]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Septoria_leaf_spot/00f16858-f392-4d9e-ad9f-efab8049a13f___JR_Sept.L.S 8368.JPG"

prediction(path)


# In[104]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Septoria_leaf_spot/00f16858-f392-4d9e-ad9f-efab8049a13f___JR_Sept.L.S 8368.JPG"

prediction(path)


# In[105]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Septoria_leaf_spot/00f16858-f392-4d9e-ad9f-efab8049a13f___JR_Sept.L.S 8368.JPG"

prediction(path)


# In[106]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Spider_mites_Two_spotted_spider_mite/0ad80523-3f34-41ae-aec8-e676546ddde5___Com.G_SpM_FL 8489.JPG"

prediction(path)


# In[107]:


print(h['accuracy'])


# In[108]:


path="C:/Users/gaja1/anaconda3/envs/hello-tf/Dataset/PlantVillage/Tomato_Spider_mites_Two_spotted_spider_mite/0bfd3a50-768e-4661-a134-43aaee7a1c1d___Com.G_SpM_FL 8837.JPG"

prediction(path)


# In[112]:





# In[110]:


print(acc)


# In[ ]:




