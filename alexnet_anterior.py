# -*- coding: utf-8 -*-

""" AlexNet.

Applying 'Alexnet' to Oxford's 17 Category Flower Dataset classification task.

References:
    - Alex Krizhevsky, Ilya Sutskever & Geoffrey E. Hinton. ImageNet
    Classification with Deep Convolutional Neural Networks. NIPS, 2012.
    - 17 Category Flower Dataset. Maria-Elena Nilsback and Andrew Zisserman.

Links:
    - [AlexNet Paper](http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
    - [Flower Dataset (17)](http://www.robots.ox.ac.uk/ ~vgg/data/flowers/17/)
"""

from __future__ import division, print_function, absolute_import

import tflearn
import xlwt
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
from keras.utils import np_utils
import os
import numpy as np
from numpy import *
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from PIL import Image
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
#import tflearn.datasets.oxflower17 as oxflower17
#X, Y = oxflower17.load_data(one_hot=True, resize_pics=(227, 227))
# input image dimensions
img_rows, img_cols = 227, 227

# number of channels
img_channels = 3

#%%
#  data

path1 = 'C:/Users/m157546/Desktop/flower/anterior/images'    #path of folder of images    
path2 = 'C:/Users/m157546/Desktop/flower/anterior/resized'  #path of folder to save images    

listing = os.listdir(path1) 
num_samples=size(listing)
print (num_samples)

# for file in listing:
#     im = Image.open(path1 + '/' + file)   
#     img = im.resize((img_rows,img_cols))
#     #img = array(img).reshape(1,img_rows,img_cols,1)
#     gray = img.convert('L')
#                 #need to do some more processing here           
#     gray.save(path2 +'/' +  file, "JPEG")
#     img.save(path2 +'/' +  file, "JPEG")

imlist = os.listdir(path2)

#im1 = array(Image.open('resized' + '/'+ imlist[0])) # open one image to get size
im1 = array(Image.open(path2 + '/'+ imlist[0])) # open one image to get size
m,n = im1.shape[0:2] # get the size of the images

imnbr = len(imlist) # get the number of images

# create matrix to store all flattened images
#immatrix = array([array(Image.open('resized'+ '/' + im2)).flatten()
              #for im2 in imlist],'f')
immatrix = array([array(Image.open(path2 + '/' + im2)).flatten()
              for im2 in imlist],'f')    
immatrix=immatrix.reshape(-1,img_rows,img_cols,1)        
label=np.ones((num_samples,),dtype = int)
label[0:853]=0
label[854:1712]=1 # 859
label[1713:]=2 # 503


data,Label = shuffle(immatrix,label, random_state=2)
train_data = [data,Label]

#img=immatrix[167].reshape(img_rows,img_cols)

# plt.imshow(img)
# plt.imshow(img,cmap='gray')
print (train_data[0].shape)
print (train_data[1].shape)

#%%

#batch_size to train
batch_size = 32
# number of output classes
nb_classes = 3
# number of epochs to train
nb_epoch = 20


# number of convolutional filters to use
nb_filters = 32
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3

#%%
(X, Y) = (train_data[0],train_data[1])


# STEP 1: split X and y into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=4)


#X_train = X_train.reshape(X_train.shape[0],  img_rows, img_cols)
X_train = X_train.reshape(-1,  img_rows, img_cols, 1)
#X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)
X_test = X_test.reshape(-1, img_rows, img_cols, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# X_train /= 255
# X_test /= 255

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y = np_utils.to_categorical(Y, nb_classes)
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
print(Y_test)
i = 100
#plt.imshow(X_train[i, 0], interpolation='nearest')
print("label : ", Y_train[i,:])

# Real-time data preprocessing
img_prep = ImagePreprocessing()
img_prep.add_featurewise_zero_center()
img_prep.add_featurewise_stdnorm()

# Real-time data augmentation
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
# Building 'AlexNet'
network = input_data(shape=[None, 227, 227, 1],data_preprocessing=img_prep,
                     data_augmentation=img_aug, name = 'input')
network = conv_2d(network, 96, 11, strides=4, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 256, 5, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 384, 3, activation='relu')
network = conv_2d(network, 256, 3, activation='relu')
network = max_pool_2d(network, 3, strides=2)
network = local_response_normalization(network)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 4096, activation='tanh')
network = dropout(network, 0.5)
network = fully_connected(network, 3, activation='softmax')
network = regression(network, optimizer='adam',
                     loss='categorical_crossentropy',
                      learning_rate=0.001, name='target')

# Training
model = tflearn.DNN(network, checkpoint_path='model_alexnet',
                    max_checkpoints=1, tensorboard_verbose=2)
# model.fit(X, Y, n_epoch=1000, validation_set=0.1, shuffle=True,
#           show_metric=True, batch_size=64, snapshot_step=200,
#           snapshot_epoch=False, run_id='alexnet_oxflowers17')
model.fit({'input': X}, {'target': Y}, n_epoch=10, validation_set=({'input': X_test}, {'target': Y_test}), shuffle=True,
          show_metric=True, batch_size=128, snapshot_step=200,
          snapshot_epoch=True, run_id='alexnet_ILD')

model.save('alexnet_anterior')
# path = 'C:/Users/m157546/Desktop/flower/anterior/test'  #path of folder to save images    
# imlist = os.listdir(path)
# imnbr = len(imlist) # get the number of images
# immatrix = array([array(Image.open(path + '/' + im2)).flatten()
#               for im2 in imlist],'f')   
# immatrix=immatrix.reshape(imnbr,img_rows,img_cols,1)  
# immatrix = immatrix.astype('float32')
# # im = array(Image.open(path + '/'+'test.jpg')) # open one image to get size
# # im = im.astype('float32')
# # im = im.reshape(-1,  img_rows, img_cols, 1)
# # model.load('alexnet_bz.data-00000-of-00001')
# bz = model.predict(immatrix)
# np.savetxt('alex_anterior.txt',bz)

# path = 'C:/Users/m157546/Desktop/flower/lateral/test'  #path of folder to save images    
# imlist = os.listdir(path)
# imnbr = len(imlist) # get the number of images
# immatrix = array([array(Image.open(path + '/' + im1)).flatten()
#               for im1 in imlist],'f')   
# immatrix=immatrix.reshape(imnbr,img_rows,img_cols,1)  
# immatrix = immatrix.astype('float32')
# # im = array(Image.open(path + '/'+'test.jpg')) # open one image to get size
# # im = im.astype('float32')
# # im = im.reshape(-1,  img_rows, img_cols, 1)
# # model.load('alexnet_bz.data-00000-of-00001')
# bz = model.predict(immatrix)
# np.savetxt('alex_lateral.txt',bz)

path = 'C:/Users/m157546/Desktop/flower/anterior/test'  #path of folder to save images    
imlist = os.listdir(path)
imnbr = len(imlist) # get the number of images
immatrix = array([array(Image.open(path + '/' + im2)).flatten()
              for im2 in imlist],'f')                      
immatrix=immatrix.reshape(imnbr,img_rows,img_cols,1)  
immatrix = immatrix.astype('float32')
# im = array(Image.open(path + '/'+'test.jpg')) # open one image to get size
# im = im.astype('float32')
# im = im.reshape(-1,  img_rows, img_cols, 1)
# model.load('alexnet_bz.data-00000-of-00001')
bz = model.predict(immatrix)
np.savetxt('alex_anterior.txt',bz)

# os.system('python alexnet_lateral.py')